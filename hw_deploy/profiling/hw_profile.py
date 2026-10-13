"""
Hardware-oriented profiling: EEGNet student vs Corr-DCRNN teacher (HW Phase 1).

Counts, per 6-s clip (18 ch x 1200 samples @ 200 Hz), what a chip pays for:
  - MACs per layer, hook-driven from the real modules' runtime shapes and
    cross-checked against torch.utils.flop_counter
  - non-MAC ops by type (add/mul, exp, sigmoid, tanh, log, sqrt, div, cmp):
    these need different hardware (LUT / CORDIC / divider), so they are
    tallied separately instead of folded into a single FLOP number
  - weight memory, input-window buffer, peak live activations
  - teacher preprocessing (log-amplitude FFT + correlation graph + top-k +
    dual random-walk matrices), which the student does not need
  - a "reordered" student: firstconv -> BN1 -> depthwise spatial conv -> BN2
    are all linear, so spatial mixing can run BEFORE the temporal filter.
    Exact equivalence is verified numerically (float64) below.

No dataset needed: costs depend only on architecture + shapes. The
equivalence check uses random weights with populated BN statistics; pass
--student_ckpt to also re-check it on a trained student.

Conventions
  - 1 MAC = one multiply-accumulate (conv / linear / matmul).
  - Inference BN is counted as 1 mul + 1 add per element but marked
    "foldable" (merged into the preceding linear op at deployment) and
    excluded from deployment totals.
  - Activation memory counts values (multiply by bytes/value for a given
    precision); the summary reports KB at 8-bit and 16-bit.
  - Teacher memory assumes hardware-friendly step-interleaved execution
    (layer1 -> layer2 per time step), not the PyTorch layer-major loop.

Run (local venv, CPU):
    python hw_deploy/profiling/hw_profile.py
    python hw_deploy/profiling/hw_profile.py --student_ckpt path/to/best.pth.tar
"""

import sys
import json
import math
import argparse
from types import SimpleNamespace
from pathlib import Path
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.flop_counter import FlopCounterMode

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))

import utils
from data.data_utils import keep_topk
from model.model import DCRNNModel_classification
from model.cell import DiffusionGraphConv, DCGRUCell
from model.eegnet import EEGNetStudent

OPS = ["mac", "add", "mul", "exp", "sigmoid", "tanh", "log", "sqrt", "div", "cmp"]


def get_args():
    p = argparse.ArgumentParser("HW profiling: student vs teacher.")
    p.add_argument("--n_chans", type=int, default=18)
    p.add_argument("--fs", type=int, default=200)
    p.add_argument("--win_len", type=int, default=6)
    p.add_argument("--fft_features", type=int, default=100)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--rnn_units", type=int, default=64)
    p.add_argument("--num_rnn_layers", type=int, default=2)
    p.add_argument("--max_diffusion_step", type=int, default=2)
    p.add_argument("--student_ckpt", type=str, default=None,
                   help="Optional trained student ckpt for the equivalence check.")
    p.add_argument("--out_dir", type=str, default=str(REPO / "hw_deploy" / "results"))
    p.add_argument("--seed", type=int, default=2026)
    return p.parse_args()


def row(stage, layer, out_shape="", params=0, act_in=0, act_out=0,
        foldable=False, note="", **ops):
    r = OrderedDict(stage=stage, layer=layer, out_shape=str(out_shape))
    for k in OPS:
        r[k] = int(ops.get(k, 0))
    r.update(params=int(params), act_in=int(act_in), act_out=int(act_out),
             foldable=foldable, note=note)
    return r


def totals(rows, include_foldable=False):
    t = {k: 0 for k in OPS + ["params"]}
    for r in rows:
        if r["foldable"] and not include_foldable:
            continue
        for k in t:
            t[k] += r[k]
    return t


def n_params(mod):
    return sum(p.numel() for p in mod.parameters(recurse=False))


# ─────────────────────────────────────────────────────────────────────────────
# Generic hook-based counter for CNN-style modules (student, both forms)
# ─────────────────────────────────────────────────────────────────────────────

def profile_cnn(model, x, stage):
    """Run one forward pass with hooks on leaf modules; return per-layer rows."""
    rows = []
    names = {m: n for n, m in model.named_modules()}

    def hook(mod, inp, out):
        xi = inp[0]
        name = names[mod]
        ai, ao = xi.numel(), out.numel()
        shape = tuple(out.shape[1:])
        if isinstance(mod, nn.modules.conv._ConvNd):
            k = math.prod(mod.kernel_size)
            macs = ao * (mod.in_channels // mod.groups) * k
            kind = "depthwise" if mod.groups == mod.in_channels and mod.groups > 1 else "conv"
            rows.append(row(stage, f"{name} [{kind} k={tuple(mod.kernel_size)} g={mod.groups}]",
                            shape, n_params(mod), ai, ao,
                            mac=macs, add=ao if mod.bias is not None else 0))
        elif isinstance(mod, nn.Linear):
            rows.append(row(stage, f"{name} [linear {mod.in_features}->{mod.out_features}]",
                            shape, n_params(mod), ai, ao,
                            mac=ao * mod.in_features, add=ao if mod.bias is not None else 0))
        elif isinstance(mod, nn.modules.batchnorm._BatchNorm):
            rows.append(row(stage, f"{name} [BN]", shape, n_params(mod), ai, ao,
                            foldable=True, mul=ao, add=ao, note="folded into prev conv"))
        elif isinstance(mod, nn.ELU):
            # x>0 ? x : exp(x)-1  (worst case: every element takes the exp branch)
            rows.append(row(stage, f"{name} [ELU]", shape, 0, ai, ao,
                            cmp=ao, exp=ao, add=ao))
        elif isinstance(mod, (nn.AvgPool1d, nn.AvgPool2d)):
            ks = mod.kernel_size if isinstance(mod.kernel_size, tuple) else (mod.kernel_size,)
            k = math.prod(ks)
            rows.append(row(stage, f"{name} [avgpool k={k}]", shape, 0, ai, ao,
                            add=ao * (k - 1), mul=ao, note="1/k scale foldable into next linear"))
        # Dropout (eval) / Flatten / containers: zero cost

    hooks = [m.register_forward_hook(hook) for m in model.modules()
             if len(list(m.children())) == 0]
    with torch.no_grad():
        model(x)
    for h in hooks:
        h.remove()
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Reordered (spatial-first, BN-folded) student
# ─────────────────────────────────────────────────────────────────────────────

class EEGNetReordered(nn.Module):
    """Inference-only, BN-folded, spatial-first equivalent of EEGNetStudent.

    Original block 1 (per output map d, temporal filter f = d // D):
        y_d = BN2_d( sum_c w_d[c] * BN1_f( h_f (*) x_c ) )
    Every step is linear, and zero padding commutes with channel mixing, so
        y_d = g_d (*) ( sum_c w_d[c] x_c ) + beta_d
        g_d    = a2_d * a1_f * h_f
        beta_d = a2_d * b1_f * sum_c w_d[c] + b2_d
    -> spatial 1x1 mix (C -> F1*D) on raw samples, then a depthwise temporal
       conv on only F1*D maps instead of F1*C maps.
    """

    def __init__(self, s: EEGNetStudent):
        super().__init__()
        conv1, bn1 = s.firstconv[0], s.firstconv[1]
        dw, bn2 = s.depthwise[0], s.depthwise[1]
        pool1, pool2 = s.depthwise[3], s.separable[4]
        sep_dw, sep_pw, bn3 = s.separable[0], s.separable[1], s.separable[2]

        F1 = conv1.out_channels
        FD = dw.out_channels
        D = FD // F1
        C = dw.kernel_size[0]
        K = conv1.kernel_size[1]

        def affine(bn):
            a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            return a, bn.bias - bn.running_mean * a

        a1, b1 = affine(bn1)
        a2, b2 = affine(bn2)
        a3, b3 = affine(bn3)
        f_of_d = torch.arange(FD) // D
        h = conv1.weight[:, 0, 0, :]          # (F1, K)
        W = dw.weight[:, 0, :, 0]             # (FD, C)

        self.spatial = nn.Conv1d(C, FD, 1, bias=False)
        self.temporal = nn.Conv1d(FD, FD, K, padding=conv1.padding[1], groups=FD, bias=True)
        self.elu1 = nn.ELU()
        self.pool1 = nn.AvgPool1d(pool1.kernel_size[1])
        self.sep_dw = nn.Conv1d(FD, FD, sep_dw.kernel_size[1], padding=sep_dw.padding[1],
                                groups=FD, bias=False)
        self.sep_pw = nn.Conv1d(FD, sep_pw.out_channels, 1, bias=True)
        self.elu2 = nn.ELU()
        self.pool2 = nn.AvgPool1d(pool2.kernel_size[1])
        self.flatten = nn.Flatten()
        self.classify = nn.Linear(s.classify.in_features, s.classify.out_features)

        with torch.no_grad():
            self.spatial.weight.copy_(W[:, :, None])
            g = (a2 * a1[f_of_d])[:, None] * h[f_of_d]               # (FD, K)
            self.temporal.weight.copy_(g[:, None, :])
            self.temporal.bias.copy_(a2 * b1[f_of_d] * W.sum(1) + b2)
            self.sep_dw.weight.copy_(sep_dw.weight[:, :, 0, :])
            self.sep_pw.weight.copy_(sep_pw.weight[:, :, 0, :] * a3[:, None, None])
            self.sep_pw.bias.copy_(b3)
            self.classify.load_state_dict(s.classify.state_dict())

    def forward(self, x):
        x = self.pool1(self.elu1(self.temporal(self.spatial(x))))
        x = self.pool2(self.elu2(self.sep_pw(self.sep_dw(x))))
        return self.classify(self.flatten(x))


def populate_bn_stats(model, n_chans, n_times, scale, n_batches=8):
    """Give BN layers non-trivial running stats (random-weight check only)."""
    for m in model.modules():
        if isinstance(m, nn.modules.batchnorm._BatchNorm):
            m.momentum = None                  # cumulative average
            m.reset_running_stats()
            with torch.no_grad():
                m.weight.uniform_(0.5, 1.5)
                m.bias.uniform_(-0.5, 0.5)
    model.train()
    with torch.no_grad():
        for _ in range(n_batches):
            model(torch.randn(16, n_chans, n_times, dtype=next(model.parameters()).dtype) * scale)
    model.eval()


def check_equivalence(student, n_chans, n_times, scale, label):
    s64 = student.double().eval()
    r64 = EEGNetReordered(s64).double().eval()
    x = torch.randn(32, n_chans, n_times, dtype=torch.float64) * scale
    with torch.no_grad():
        yo, yr = s64(x), r64(x)
    d64 = (yo - yr).abs().max().item()
    s32 = student.float().eval()
    r32 = EEGNetReordered(s32).float().eval()
    with torch.no_grad():
        yo32, yr32 = s32(x.float()), r32(x.float())
    d32 = (yo32 - yr32).abs().max().item()
    ref = yo.abs().max().item()
    print(f"  [{label}] max|logit diff|: float64 {d64:.3e}  float32 {d32:.3e}  "
          f"(max|logit| {ref:.3e})")
    return dict(label=label, max_abs_diff_fp64=d64, max_abs_diff_fp32=d32, max_abs_logit=ref)


# ─────────────────────────────────────────────────────────────────────────────
# Teacher: preprocessing (analytic) + model (hook-driven)
# ─────────────────────────────────────────────────────────────────────────────

def build_supports(raw_tcs, top_k):
    """Same graph path as the dataloader: |corr| -> self-edges -> top-k -> dual RW."""
    T, C, S = raw_tcs.shape
    flat = raw_tcs.transpose(1, 0, 2).reshape(C, -1).astype(np.float64)
    gram = flat @ flat.T
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    adj = np.abs((gram / np.outer(norms, norms)).astype(np.float32))
    np.fill_diagonal(adj, 1.0)
    adj = keep_topk(adj, top_k=top_k, directed=True)
    sup = [utils.calculate_random_walk_matrix(adj).T,
           utils.calculate_random_walk_matrix(adj.T).T]
    return [torch.FloatTensor(s.toarray()) for s in sup], adj


def teacher_preproc_rows(C, T, S, M, top_k, fft_mode):
    """Per-clip preprocessing the teacher needs and the student does not.

    fft_mode: 'fft'  radix estimate for a real 200-pt FFT
              'dft'  direct DFT of only the M kept bins (simplest HW datapath)
    """
    rows = []
    n_fft = T * C
    if fft_mode == "fft":
        # complex radix-2-equivalent: (S/2)log2(S) cmul = 4 real mul + 2 add,
        # halved for real input; plus (S)log2(S) complex adds.
        cm = (S / 2) * math.log2(S) / 2
        mac = 4 * cm
        add = 2 * S * math.log2(S) / 2
        note = f"real {S}-pt FFT, radix estimate (lower bound; {S} not a power of 2)"
    else:
        mac = 2 * M * S          # re/im accumulation of M bins over S samples
        add = 0
        note = f"direct DFT of {M} kept bins (upper bound, trivially pipelined)"
    rows.append(row("teacher-preproc", f"FFT x{n_fft} ({fft_mode})", (T, C, M), 0,
                    T * C * S, T * C * M, mac=n_fft * mac, add=n_fft * add, note=note))
    rows.append(row("teacher-preproc", "|X| + log", (T, C, M), 0, T * C * M, T * C * M,
                    mul=2 * n_fft * M, add=n_fft * M, sqrt=n_fft * M, log=n_fft * M,
                    note="log|X| = 0.5 log(re^2+im^2) can drop the sqrt"))
    rows.append(row("teacher-preproc", "z-norm (scaler)", (T, C, M), 2 * C * M,
                    T * C * M, T * C * M, mac=n_fft * M,
                    note="(x-mu)/sigma as x*s+b; per-(C,M) constants"))
    npair = C * (C + 1) // 2
    rows.append(row("teacher-preproc", "gram X X^T (symmetric)", (C, C), 0,
                    C * T * S, npair, mac=npair * T * S,
                    note="needs the whole 6-s window before DCGRU can start"))
    rows.append(row("teacher-preproc", "corr normalise", (C, C), 0, npair, C * C,
                    sqrt=C, div=C, mul=2 * npair, note="1/||x|| per ch, then 2 mul per pair"))
    rows.append(row("teacher-preproc", f"|.| + top-{top_k} select", (C, C), 0, C * C, C * C,
                    cmp=C * (C - 1) * top_k + C * C, note="partial selection per row"))
    rows.append(row("teacher-preproc", "dual random-walk D^-1 W", (2, C, C), 0, C * C, 2 * C * C,
                    add=2 * C * top_k, div=2 * C, mul=2 * C * (top_k + 1),
                    note="row/col sums, reciprocal, scale nnz"))
    return rows


def profile_teacher(teacher, x, seq_len, supports, nnz_per_support):
    rows = []
    N = teacher.num_nodes
    H = teacher.rnn_units
    call = {"dgc": 0, "cell": 0}
    layer_of = {}
    for li, cell in enumerate(teacher.encoder.encoding_cells):
        layer_of[cell] = li
        layer_of[cell.dconv_gate] = li
        layer_of[cell.dconv_candidate] = li

    def dgc_hook(mod, args, out):
        sups, inputs, state, out_dim = args[0], args[1], args[2], args[3]
        F = mod._input_size
        K = mod._max_diffusion_step
        S = len(sups)
        Mm = S * K + 1
        dense = S * K * N * N * F
        sparse = sum(K * nnz * F for nnz in nnz_per_support[:S])
        which = "gate(r,u)" if out_dim == 2 * H else "cand(c)"
        rows.append(row("teacher-model", f"L{layer_of[mod]+1} {which} diffusion S^k x",
                        (Mm, N, F), 0, N * F, Mm * N * F,
                        mac=sparse, mul=S * (K - 1) * N * F, add=S * (K - 1) * N * F,
                        note=f"dense would be {dense:,} MAC; top-k graph -> sparse"))
        rows.append(row("teacher-model", f"L{layer_of[mod]+1} {which} weight [{F*Mm}x{out_dim}]",
                        (N, out_dim), n_params(mod), Mm * N * F, N * out_dim,
                        mac=N * F * Mm * out_dim, add=N * out_dim))
        rows[-2]["_dense_mac"] = dense
        call["dgc"] += 1

    def cell_hook(mod, args, out):
        NH = N * H
        rows.append(row("teacher-model", f"L{layer_of[mod]+1} GRU gating", (N, H), 0,
                        3 * NH, NH, sigmoid=2 * NH, tanh=NH, mul=3 * NH, add=2 * NH,
                        note="sigma(r,u), tanh(c), r*h, u*h+(1-u)*c"))

    hooks = []
    for m in teacher.modules():
        if isinstance(m, DiffusionGraphConv):
            hooks.append(m.register_forward_hook(dgc_hook))
        elif isinstance(m, DCGRUCell):
            hooks.append(m.register_forward_hook(cell_hook))
    with torch.no_grad():
        teacher(x, seq_len, supports)
    for h in hooks:
        h.remove()

    fc = teacher.fc
    rows.append(row("teacher-model", f"head relu + fc {H}->{fc.out_features} per node",
                    (N, fc.out_features), n_params(fc), N * H, N * fc.out_features,
                    cmp=N * H, mac=N * H * fc.out_features, add=N * fc.out_features))
    rows.append(row("teacher-model", "max-pool over nodes", (1,), 0, N, 1, cmp=N - 1))
    return rows


def collapse_timesteps(rows, T):
    """Merge identical per-timestep rows into one row with x T counts."""
    out = OrderedDict()
    for r in rows:
        key = (r["stage"], r["layer"])
        if key not in out:
            out[key] = dict(r)
            out[key]["_n"] = 1
        else:
            for k in OPS:
                out[key][k] += r[k]
            if "_dense_mac" in r:
                out[key]["_dense_mac"] += r["_dense_mac"]
            out[key]["_n"] += 1
    res = []
    for r in out.values():
        if r["_n"] > 1:
            r["layer"] = f"{r['layer']}  (x{r['_n']} steps)"
        if "_dense_mac" in r:
            r["note"] = f"dense S would be {r['_dense_mac']:,} MAC; top-k graph -> sparse"
        res.append(r)
    return res


# ─────────────────────────────────────────────────────────────────────────────
# Memory model
# ─────────────────────────────────────────────────────────────────────────────

def student_memory(rows, n_chans, n_times):
    """Layer-by-layer execution: live = this layer's input + output."""
    live = [r["act_in"] + r["act_out"] for r in rows if not r["foldable"]]
    peak_i = int(np.argmax(live))
    peak_layer = [r for r in rows if not r["foldable"]][peak_i]["layer"]
    return dict(input_buffer=n_chans * n_times, peak_live=max(live), peak_layer=peak_layer)


def student_streaming_memory(r: "EEGNetReordered", n_chans):
    """Sample-by-sample execution of the reordered student (line buffers only).

    spatial mix is per-sample; each temporal / separable conv keeps a k-tap
    line buffer per map; pools keep one accumulator per map; the classifier
    accumulates its outputs as pooled features arrive. Possible because no
    stage needs the whole window first (the teacher's graph does).
    """
    FD = r.temporal.out_channels
    F2 = r.sep_pw.out_channels
    d = dict(
        input_sample=n_chans,
        temporal_line_buffer=FD * r.temporal.kernel_size[0],
        pool1_acc=FD,
        sep_dw_line_buffer=FD * r.sep_dw.kernel_size[0],
        sep_pw_out=F2,
        pool2_acc=F2,
        classifier_acc=r.classify.out_features,
    )
    return dict(input_buffer=n_chans, peak_live=sum(d.values()), detail=d)


def teacher_memory(C, T, S, M, H, L, top_k, feat_dim_in):
    """Step-interleaved execution after the whole window has been featurised.

    Window phase: 1-s raw segment buffer + all T feature frames + gram accumulators.
    Model phase : feature frames (still being consumed) + L hidden states +
                  diffusion stack of the widest layer + its gate outputs.
    """
    raw_seg = C * S
    feats = T * C * M
    gram = C * (C + 1) // 2
    graph = 2 * C * (top_k + 1) * 2          # 2 supports, nnz values + indices
    F_max = feat_dim_in + H
    Mm = 2 * 2 + 1
    states = L * C * H
    work = C * F_max * Mm + C * 2 * H + C * H
    return dict(input_buffer=feats + raw_seg + gram,
                peak_live=feats + graph + states + work,
                detail=dict(raw_segment=raw_seg, fft_feature_frames=feats, gram_acc=gram,
                            sparse_graph=graph, hidden_states=states, dconv_working_set=work))


# ─────────────────────────────────────────────────────────────────────────────
# Reporting
# ─────────────────────────────────────────────────────────────────────────────

def fmt(n):
    return f"{n:,}" if n else "-"


def md_layer_table(title, rows):
    lines = [f"### {title}", "",
             "| layer | out | MAC | add | mul | exp | sigm | tanh | log | sqrt | div | cmp | params | note |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        lay = r["layer"] + (" *(foldable)*" if r["foldable"] else "")
        lines.append("| " + " | ".join([lay, r["out_shape"]] + [fmt(r[k]) for k in OPS]
                                       + [fmt(r["params"]), r["note"]]) + " |")
    t = totals(rows)
    lines.append("| **total (deploy)** | | " + " | ".join(f"**{fmt(t[k])}**" for k in OPS)
                 + f" | **{fmt(t['params'])}** | |")
    return "\n".join(lines) + "\n"


def main():
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    C, T, S, M = args.n_chans, args.win_len, args.fs, args.fft_features
    n_times = T * S
    H, L = args.rnn_units, args.num_rnn_layers
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("== Student equivalence check (original vs reordered) ==")
    eq = []
    student = EEGNetStudent(n_chans=C, n_times=n_times, n_outputs=2)
    for scale in (1.0, 50.0):
        populate_bn_stats(student, C, n_times, scale)
        eq.append(check_equivalence(student, C, n_times, scale, f"random, input std {scale:g}"))
    if args.student_ckpt:
        st = EEGNetStudent(n_chans=C, n_times=n_times, n_outputs=2)
        st = utils.load_model_checkpoint(args.student_ckpt, st).eval()
        eq.append(check_equivalence(st, C, n_times, 50.0, f"ckpt {Path(args.student_ckpt).name}"))
    student = student.float().eval()

    x_raw = torch.randn(1, C, n_times)
    s_rows = profile_cnn(student, x_raw, "student")
    reordered = EEGNetReordered(student).eval()
    r_rows = profile_cnn(reordered, x_raw, "student-reordered")

    # Teacher
    targs = SimpleNamespace(num_nodes=C, num_rnn_layers=L, rnn_units=H, input_dim=M,
                            max_diffusion_step=args.max_diffusion_step,
                            dcgru_activation="tanh", filter_type="dual_random_walk", dropout=0.0)
    teacher = DCRNNModel_classification(args=targs, num_classes=1, device="cpu").eval()
    raw_tcs = np.random.randn(T, C, S).astype(np.float32)
    supports, adj = build_supports(raw_tcs, args.top_k)
    nnz = [int((s != 0).sum()) for s in supports]
    x_fft = torch.randn(1, T, C, M)
    seq_len = torch.LongTensor([T])
    t_rows = collapse_timesteps(profile_teacher(teacher, x_fft, seq_len, supports, nnz), T)
    p_rows_fft = teacher_preproc_rows(C, T, S, M, args.top_k, "fft")
    p_rows_dft = teacher_preproc_rows(C, T, S, M, args.top_k, "dft")

    # Cross-check MACs with torch's own counter (teacher counted with DENSE supports)
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            student(x_raw)
    fc_student = fc.get_total_flops() // 2
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            reordered(x_raw)
    fc_reord = fc.get_total_flops() // 2
    with FlopCounterMode(display=False) as fc:
        with torch.no_grad():
            teacher(x_fft, seq_len, supports)
    fc_teacher = fc.get_total_flops() // 2
    t_dense = totals(t_rows)["mac"] + sum(r.get("_dense_mac", 0) - r["mac"]
                                          for r in t_rows if "_dense_mac" in r)
    xcheck = dict(
        student=dict(ours=totals(s_rows)["mac"], torch=fc_student),
        student_reordered=dict(ours=totals(r_rows)["mac"], torch=fc_reord),
        teacher_dense=dict(ours=t_dense, torch=fc_teacher),
    )
    print("\n== MAC cross-check vs torch.utils.flop_counter ==")
    for k, v in xcheck.items():
        print(f"  {k:18s} ours {v['ours']:>12,}  torch {v['torch']:>12,}  "
              f"{'OK' if v['ours'] == v['torch'] else 'MISMATCH'}")

    # Memory
    mem = dict(
        student=student_memory(s_rows, C, n_times),
        student_reordered=student_memory(r_rows, C, n_times),
        student_reordered_streaming=student_streaming_memory(reordered, C),
        teacher=teacher_memory(C, T, S, M, H, L, args.top_k, M),
    )

    # Summary
    tt = totals(t_rows)
    pf, pd_ = totals(p_rows_fft), totals(p_rows_dft)
    summary = OrderedDict()
    summary["Teacher model only"] = dict(tt, **mem["teacher"])
    summary["Teacher + preproc (FFT)"] = {k: tt[k] + pf[k] for k in tt}
    summary["Teacher + preproc (DFT)"] = {k: tt[k] + pd_[k] for k in tt}
    summary["Student (original)"] = dict(totals(s_rows), **mem["student"])
    summary["Student (reordered)"] = dict(totals(r_rows), **mem["student_reordered"])
    summary["Student (reordered, streaming)"] = dict(
        totals(r_rows), **{k: v for k, v in mem["student_reordered_streaming"].items()
                           if k != "detail"})
    for k in ("Teacher + preproc (FFT)", "Teacher + preproc (DFT)"):
        summary[k].update(params=tt["params"] + pf["params"], **mem["teacher"])
    for v in summary.values():
        v["onchip_total"] = v["params"] + v["peak_live"]

    print("\n== Per-clip summary (deployment: BN folded) ==")
    hdr = f"  {'':26s}{'MAC':>12s}{'nonlin':>10s}{'add+mul':>10s}{'params':>9s}" \
          f"{'in-buf':>9s}{'peak-act':>10s}{'on-chip':>10s}"
    print(hdr)
    for k, v in summary.items():
        nonlin = sum(v[o] for o in ("exp", "sigmoid", "tanh", "log", "sqrt", "div"))
        print(f"  {k:26s}{v['mac']:>12,}{nonlin:>10,}{v['add']+v['mul']:>10,}"
              f"{v['params']:>9,}{v['input_buffer']:>9,}{v['peak_live']:>10,}{v['onchip_total']:>10,}")

    t_full = summary["Teacher + preproc (FFT)"]
    ratios = OrderedDict()
    for sk in ("Student (original)", "Student (reordered)", "Student (reordered, streaming)"):
        s = summary[sk]
        ratios[sk] = dict(
            params=tt["params"] / s["params"],
            mac_vs_teacher_model=tt["mac"] / s["mac"],
            mac_vs_teacher_pipeline_fft=t_full["mac"] / s["mac"],
            peak_act_vs_teacher=mem["teacher"]["peak_live"] / s["peak_live"],
            onchip_vs_teacher=t_full["onchip_total"] / s["onchip_total"],
        )
    print("\n== Teacher / Student ratios ==")
    for sk, r in ratios.items():
        print(f"  {sk:30s} params {r['params']:.1f}x | MAC {r['mac_vs_teacher_model']:.2f}x "
              f"(model) / {r['mac_vs_teacher_pipeline_fft']:.2f}x (+preproc) | "
              f"peak act {r['peak_act_vs_teacher']:.2f}x | on-chip {r['onchip_vs_teacher']:.2f}x")

    # Write outputs
    def clean(rows):
        return [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]

    result = dict(config=vars(args), equivalence=eq, mac_crosscheck=xcheck,
                  support_nnz=nnz, memory=mem, summary=summary, ratios=ratios,
                  layers=dict(student=clean(s_rows), student_reordered=clean(r_rows),
                              teacher_model=clean(t_rows), teacher_preproc_fft=clean(p_rows_fft),
                              teacher_preproc_dft=clean(p_rows_dft)))
    (out_dir / "hw_profile.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    md = ["# HW profile: Corr-DCRNN teacher vs EEGNet-8,2 student", "",
          f"Per clip: {C} ch x {n_times} samples ({T} s @ {S} Hz). BN folded for deployment totals. "
          f"Top-{args.top_k} graph -> {nnz} nnz per support.", "",
          "## Summary", "",
          "| | MAC | nonlinear (exp/sigm/tanh/log/sqrt/div) | add+mul | cmp | params | input buffer | peak live act | on-chip (params+act) |",
          "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, v in summary.items():
        nonlin = sum(v[o] for o in ("exp", "sigmoid", "tanh", "log", "sqrt", "div"))
        md.append(f"| {k} | {v['mac']:,} | {nonlin:,} | {v['add']+v['mul']:,} | {v['cmp']:,} | "
                  f"{v['params']:,} | {v['input_buffer']:,} | {v['peak_live']:,} | {v['onchip_total']:,} |")
    md += ["", "Memory in values; x1 B (int8) or x2 B (int16) for bytes.", "",
           "## Ratios (teacher / student)", "",
           "| student | params | MAC vs teacher model | MAC vs teacher + preproc | peak act | on-chip |",
           "|---|---:|---:|---:|---:|---:|"]
    for sk, r in ratios.items():
        md.append(f"| {sk} | {r['params']:.1f}x | {r['mac_vs_teacher_model']:.2f}x | "
                  f"{r['mac_vs_teacher_pipeline_fft']:.2f}x | {r['peak_act_vs_teacher']:.2f}x | "
                  f"{r['onchip_vs_teacher']:.2f}x |")
    md += ["", "## Equivalence (original vs reordered student)", ""]
    for e in eq:
        md.append(f"- {e['label']}: max |logit diff| fp64 {e['max_abs_diff_fp64']:.2e}, "
                  f"fp32 {e['max_abs_diff_fp32']:.2e} (max |logit| {e['max_abs_logit']:.2e})")
    md += ["", "## MAC cross-check (torch.utils.flop_counter)", ""]
    for k, v in xcheck.items():
        md.append(f"- {k}: ours {v['ours']:,} / torch {v['torch']:,}")
    md += ["", "## Teacher memory breakdown (values)", ""]
    for k, v in mem["teacher"]["detail"].items():
        md.append(f"- {k}: {v:,}")
    md += ["", "## Reordered student, streaming memory breakdown (values)", ""]
    for k, v in mem["student_reordered_streaming"]["detail"].items():
        md.append(f"- {k}: {v:,}")
    md += ["", "## Per-layer", "",
           md_layer_table("Student (original)", s_rows),
           md_layer_table("Student (reordered, BN-folded)", r_rows),
           md_layer_table("Teacher model (per clip, 6 steps)", t_rows),
           md_layer_table("Teacher preprocessing (FFT estimate)", p_rows_fft),
           md_layer_table("Teacher preprocessing (direct DFT)", p_rows_dft)]
    (out_dir / "hw_profile.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nWrote {out_dir / 'hw_profile.json'} and {out_dir / 'hw_profile.md'}")


if __name__ == "__main__":
    main()
