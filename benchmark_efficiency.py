"""
Efficiency benchmark: EEGNet student vs Corr-DCRNN teacher.

Quantifies the edge-deployment payoff for thesis §5.4 / §6:
  - trainable & total parameters
  - model size (MB, fp32)
  - per-clip inference latency (ms) on the available device AND on CPU
  - throughput (clips/s)
  - FLOPs per clip (best-effort, requires `thop`; skipped if unavailable)

Uses ONE real batch from the same dataloader (return_raw=True) so input
shapes exactly match the experiments — student gets raw (B,C,T), teacher gets
FFT features (B,seq,C,M) + correlation-graph supports.

Run (in the teacher's venv, on Vessl):
    python benchmark_efficiency.py \\
        --hdf5_dir /dataset \\
        --teacher_ckpt      runs/corr_seed2026_topk3_negr3_nodevus_metricF1_szstride1_ep10/train/train-01/best.pth.tar \\
        --teacher_args_json runs/corr_seed2026_topk3_negr3_nodevus_metricF1_szstride1_ep10/train/train-01/args.json \\
        --win_len 6 --fs 200 --top_k 3 --graph_type individual \\
        --rand_seed 2026 --batch_size 1 --n_iters 200
"""

import sys
import json
import time
import logging
import argparse
from types import SimpleNamespace
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))

import utils
from data.dataloader_chbmit_hdf5 import load_dataset_chbmit_hdf5
from model.model import DCRNNModel_classification
from model.eegnet import EEGNetStudent


def get_args():
    p = argparse.ArgumentParser("Efficiency benchmark: student vs teacher.")
    p.add_argument("--hdf5_dir", type=str, required=True)
    p.add_argument("--summary_dir", type=str, default=None)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--teacher_args_json", type=str, required=True)
    # data config (match the experiments)
    p.add_argument("--orig_fs", type=int, default=256)
    p.add_argument("--fs", type=int, default=200)
    p.add_argument("--win_len", type=int, default=6)
    p.add_argument("--time_step_size", type=int, default=1)
    p.add_argument("--fft_features", type=int, default=100)
    p.add_argument("--graph_type", choices=["individual", "combined"], default="individual")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--neg_ratio", type=int, default=3)
    p.add_argument("--seizure_stride", type=float, default=1.0)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--rand_seed", type=int, default=2026)
    # student
    p.add_argument("--drop_prob", type=float, default=0.5)
    # benchmark
    p.add_argument("--batch_size", type=int, default=1,
                   help="1 = single-clip edge latency (recommended).")
    p.add_argument("--n_iters", type=int, default=200)
    p.add_argument("--warmup", type=int, default=20)
    args = p.parse_args()
    args.input_dim = args.fft_features
    args.filter_type = "dual_random_walk" if args.graph_type == "individual" else "laplacian"
    return args


def model_stats(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    size_mb = (sum(p.numel() * p.element_size() for p in model.parameters())
               + sum(b.numel() * b.element_size() for b in model.buffers())) / 1e6
    return total, trainable, size_mb


@torch.no_grad()
def bench_latency(fn, n_iters, warmup, device):
    """Time fn() (one forward pass). Returns mean ms per call."""
    is_cuda = (device == "cuda")
    for _ in range(warmup):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        fn()
    if is_cuda:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n_iters * 1e3  # ms


def try_flops(model, inputs):
    try:
        from thop import profile
        macs, _ = profile(model, inputs=inputs, verbose=False)
        return macs * 2  # FLOPs ≈ 2 × MACs
    except Exception as e:
        return None


def build_teacher(json_path, ckpt_path, num_nodes, device):
    tj = json.load(open(json_path))
    targs = SimpleNamespace(
        num_nodes=num_nodes,
        num_rnn_layers=tj.get("num_rnn_layers", 2),
        rnn_units=tj.get("rnn_units", 64),
        input_dim=tj.get("input_dim", 100),
        max_diffusion_step=tj.get("max_diffusion_step", 2),
        dcgru_activation=tj.get("dcgru_activation", "tanh"),
        filter_type=tj.get("filter_type", "dual_random_walk"),
        dropout=tj.get("dropout", 0.0),
    )
    teacher = DCRNNModel_classification(args=targs, num_classes=tj.get("num_classes", 1),
                                        device=device)
    teacher = utils.load_model_checkpoint(ckpt_path, teacher).to(device)
    teacher.eval()
    return teacher


def main():
    args = get_args()
    # Show dataloader progress (otherwise setup looks silently hung).
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.seed_torch(args.rand_seed)
    print(f"Device: {device}  | batch_size={args.batch_size}  n_iters={args.n_iters}\n", flush=True)

    # ── One real batch (test loader) ────────────────────────────────────────
    # standardize=False → skip the slow streaming-scaler pass; it only rescales
    # input values and does NOT affect params/latency/FLOPs being measured here.
    print("Building dataloader (reading dataset headers; ~1-3 min)…", flush=True)
    dataloaders, datasets, _ = load_dataset_chbmit_hdf5(
        hdf5_dir=args.hdf5_dir, summary_dir=args.summary_dir,
        train_batch_size=args.batch_size, test_batch_size=args.batch_size,
        win_len=args.win_len, time_step_size=args.time_step_size,
        orig_fs=args.orig_fs, fs=args.fs, fft_features=args.fft_features,
        graph_type=args.graph_type, top_k=args.top_k, filter_type=args.filter_type,
        standardize=False, num_workers=0,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.rand_seed,
        undersample_train=True, neg_ratio=args.neg_ratio,
        seizure_stride=args.seizure_stride, inspect_first_file=True, return_raw=True,
    )
    print("Fetching one test batch…", flush=True)
    batch = next(iter(dataloaders["test"]))
    x_fft, _, seq_len, supports, _, _, raw = batch
    num_nodes = datasets["test"].num_nodes
    n_times = raw.shape[-1]
    bs = raw.shape[0]
    print(f"Inputs: raw(student)={tuple(raw.shape)}  fft(teacher)={tuple(x_fft.shape)}  "
          f"supports={len(supports)}×{tuple(supports[0].shape)}\n")

    # ── Build models ────────────────────────────────────────────────────────
    student = EEGNetStudent(n_chans=num_nodes, n_times=n_times, sfreq=args.fs,
                            n_outputs=2, drop_prob=args.drop_prob).to(device).eval()
    teacher = build_teacher(args.teacher_args_json, args.teacher_ckpt, num_nodes, device)

    def to_dev(t):
        return t.to(device)
    raw_d = to_dev(raw)
    x_d = to_dev(x_fft)
    seq_d = seq_len.view(-1).to(device)
    sup_d = [s.to(device) for s in supports]

    student_fwd = lambda: student(raw_d)
    teacher_fwd = lambda: teacher(x_d, seq_d, sup_d)

    # ── Stats ────────────────────────────────────────────────────────────────
    def report(name, model, fwd, flops_inputs):
        total, trainable, size_mb = model_stats(model)
        lat = bench_latency(fwd, args.n_iters, args.warmup, device) / bs
        flops = try_flops(model, flops_inputs)
        print(f"== {name} ==")
        print(f"  params (total)   : {total:,}")
        print(f"  params (trainable): {trainable:,}")
        print(f"  size (fp32)      : {size_mb:.3f} MB")
        print(f"  latency/clip     : {lat:.3f} ms  ({1000.0/lat:.1f} clips/s)  on {device}")
        if flops is not None:
            print(f"  FLOPs/clip       : {flops/1e6:.2f} MFLOPs")
        else:
            print(f"  FLOPs/clip       : (thop not installed → `pip install thop` to measure)")
        return total, size_mb, lat, flops

    print("Benchmarking (this device)…\n")
    s_tot, s_mb, s_lat, s_fl = report("Student (EEGNet-8,2)", student, student_fwd, (raw_d,))
    print()
    t_tot, t_mb, t_lat, t_fl = report("Teacher (Corr-DCRNN)", teacher, teacher_fwd,
                                      (x_d, seq_d, sup_d))

    # ── Ratios (print before optional CPU section so they always show) ──────
    print("\n== Student vs Teacher (efficiency gain) ==")
    print(f"  params : {t_tot/max(s_tot,1):.1f}× fewer")
    print(f"  size   : {t_mb/max(s_mb,1e-9):.1f}× smaller")
    print(f"  latency: {t_lat/max(s_lat,1e-9):.1f}× faster ({device})")
    if s_fl and t_fl:
        print(f"  FLOPs  : {t_fl/max(s_fl,1):.1f}× fewer")

    # ── Optional CPU latency (edge-relevant) ─────────────────────────────────
    if device == "cuda":
        print("\nCPU latency (edge-relevant)…")
        try:
            student_c = student.to("cpu")
            teacher_c = teacher.to("cpu")
            teacher_c._device = "cpu"   # DCRNN caches its device internally
            raw_c = raw.to("cpu"); x_c = x_fft.to("cpu")
            seq_c = seq_len.view(-1).to("cpu"); sup_c = [s.to("cpu") for s in supports]
            s_lat_cpu = bench_latency(lambda: student_c(raw_c),
                                      max(args.n_iters // 4, 20), args.warmup, "cpu") / bs
            t_lat_cpu = bench_latency(lambda: teacher_c(x_c, seq_c, sup_c),
                                      max(args.n_iters // 4, 20), args.warmup, "cpu") / bs
            print(f"  Student CPU latency/clip: {s_lat_cpu:.3f} ms  ({1000.0/s_lat_cpu:.1f} clips/s)")
            print(f"  Teacher CPU latency/clip: {t_lat_cpu:.3f} ms  ({1000.0/t_lat_cpu:.1f} clips/s)")
            print(f"  → Student is {t_lat_cpu/max(s_lat_cpu,1e-9):.1f}× faster on CPU")
        except Exception as e:
            print(f"  (CPU benchmark skipped: {e})")


if __name__ == "__main__":
    main()
