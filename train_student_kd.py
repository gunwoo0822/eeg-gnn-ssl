"""
Train the EEGNet student on CHB-MIT, optionally with online Knowledge
Distillation from a frozen Corr-DCRNN teacher (thesis §4.5 / §4.6).

Design
------
* The student (braindecode EEGNet, see model/eegnet.py) consumes the *raw*
  time-domain clip (B, C, T), e.g. (B, 18, 1200) = 18ch × 6s × 200Hz.
* The teacher (DCRNNModel_classification) consumes the *same clip's* FFT
  features (B, seq, C, 100) + correlation graph supports.
* Both representations come from the SAME dataset item (return_raw=True),
  so teacher soft-targets and student inputs are aligned by construction —
  no offline soft-label cache / clip-id matching is needed (online KD).

KD loss (binary, sigmoid special case of Hinton softmax KD):
    teacher logit z_t (num_classes=1) → 2-class logits [0, z_t]
    soft_t = softmax([0, z_t] / T)                          (frozen, detached)
    L = (1-alpha)·CE(student_logits, y) + alpha·T²·KL(soft_t || soft_s)
Set --alpha 0 to train the identical student WITHOUT distillation (baseline).

To compare fairly with the teacher table, evaluation reuses the teacher's
metric set (AUROC, AUPRC, F1, sensitivity, specificity) and dev F1-based
threshold selection (sweep_threshold from train_chbmit).

Quick-start (KD, paper-faithful preprocessing — match the teacher run):
    python train_student_kd.py \\
        --hdf5_dir     /vessl/data/chbmit_hdf5 \\
        --summary_dir  /vessl/data/chbmit \\
        --save_dir     /vessl/output/student_kd \\
        --teacher_ckpt /vessl/output/teacher/best.pth.tar \\
        --teacher_args_json /vessl/output/teacher/args.json \\
        --win_len 6 --fs 200 --top_k 3 --graph_type individual \\
        --neg_ratio 3 --seizure_stride 1 --rand_seed 2026 \\
        --alpha 0.5 --temperature 2.0 --metric_name f1
"""

import os
import sys
import json
import logging
import argparse
from types import SimpleNamespace
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import utils
from data.dataloader_chbmit_hdf5 import load_dataset_chbmit_hdf5
from model.model import DCRNNModel_classification
from model.eegnet import EEGNetStudent
from train_chbmit import sweep_threshold   # pure (y_true, y_prob) → best F1 threshold

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_args():
    p = argparse.ArgumentParser("Train EEGNet student (+ optional KD) on CHB-MIT.")

    # I/O
    p.add_argument("--hdf5_dir", type=str, required=True,
                   help="Dir with 256 Hz CHB-MIT HDF5 files (same as teacher).")
    p.add_argument("--summary_dir", type=str, default=None,
                   help="Dir with CHB-MIT *-summary.txt files.")
    p.add_argument("--save_dir", type=str, default="./student_runs")

    # KD
    p.add_argument("--alpha", type=float, default=0.5,
                   help="KD weight. 0 = no distillation (baseline student).")
    p.add_argument("--temperature", type=float, default=2.0,
                   help="KD softening temperature T.")
    p.add_argument("--teacher_ckpt", type=str, default=None,
                   help="Frozen teacher checkpoint (required when alpha>0).")
    p.add_argument("--teacher_args_json", type=str, default=None,
                   help="Teacher run's args.json to rebuild teacher arch exactly. "
                        "Falls back to --teacher_* CLI defaults if omitted.")

    # Teacher architecture (used only if --teacher_args_json not given)
    p.add_argument("--teacher_num_rnn_layers", type=int, default=2)
    p.add_argument("--teacher_rnn_units", type=int, default=64)
    p.add_argument("--teacher_max_diffusion_step", type=int, default=2)
    p.add_argument("--teacher_dcgru_activation", choices=["tanh", "relu"], default="tanh")
    p.add_argument("--teacher_dropout", type=float, default=0.0)

    # Data / preprocessing (MUST match the teacher run for valid KD)
    p.add_argument("--orig_fs", type=int, default=256)
    p.add_argument("--fs", type=int, default=200)
    p.add_argument("--win_len", type=int, default=6)
    p.add_argument("--stride", type=int, default=None)
    p.add_argument("--time_step_size", type=int, default=1)
    p.add_argument("--fft_features", type=int, default=100)
    p.add_argument("--graph_type", choices=["individual", "combined"], default="individual")
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--no_undersample", action="store_true", default=False)
    p.add_argument("--neg_ratio", type=int, default=3)
    p.add_argument("--undersample_dev", action="store_true", default=False)
    p.add_argument("--seizure_stride", type=float, default=1.0)
    p.add_argument("--train_ratio", type=float, default=0.70)
    p.add_argument("--val_ratio", type=float, default=0.15)
    p.add_argument("--rand_seed", type=int, default=2026)
    p.add_argument("--num_nodes", type=int, default=0, help="0 = auto-detect.")

    # Student (EEGNet) architecture
    p.add_argument("--drop_prob", type=float, default=0.5)
    p.add_argument("--eeg_f1", type=int, default=None, help="EEGNet F1 (default 8).")
    p.add_argument("--eeg_d", type=int, default=None, help="EEGNet D (default 2).")
    p.add_argument("--eeg_f2", type=int, default=None, help="EEGNet F2 (default 16).")

    # Training
    p.add_argument("--num_epochs", type=int, default=50)
    p.add_argument("--train_batch_size", type=int, default=40)
    p.add_argument("--test_batch_size", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--lr_init", type=float, default=1e-3)
    p.add_argument("--l2_wd", type=float, default=0.0)
    p.add_argument("--max_grad_norm", type=float, default=5.0)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--metric_name", type=str.lower,
                   choices=["auroc", "f1", "acc", "loss", "auprc"], default="f1")

    args = p.parse_args()

    args.input_dim = args.fft_features
    args.max_seq_len = args.win_len // args.time_step_size
    args.maximize_metric = (args.metric_name != "loss")
    args.filter_type = "dual_random_walk" if args.graph_type == "individual" else "laplacian"

    if args.alpha > 0 and not args.teacher_ckpt:
        p.error("--teacher_ckpt is required when --alpha > 0.")

    return args


# ---------------------------------------------------------------------------
# Teacher (frozen)
# ---------------------------------------------------------------------------

def build_teacher(args, num_nodes, device):
    """Rebuild the Corr-DCRNN teacher arch and load its frozen checkpoint."""
    if args.teacher_args_json:
        with open(args.teacher_args_json) as f:
            tj = json.load(f)
        targs = SimpleNamespace(
            num_nodes=num_nodes,
            num_rnn_layers=tj.get("num_rnn_layers", args.teacher_num_rnn_layers),
            rnn_units=tj.get("rnn_units", args.teacher_rnn_units),
            input_dim=tj.get("input_dim", args.input_dim),
            max_diffusion_step=tj.get("max_diffusion_step", args.teacher_max_diffusion_step),
            dcgru_activation=tj.get("dcgru_activation", args.teacher_dcgru_activation),
            filter_type=tj.get("filter_type", args.filter_type),
            dropout=tj.get("dropout", args.teacher_dropout),
        )
        num_classes = tj.get("num_classes", 1)
    else:
        targs = SimpleNamespace(
            num_nodes=num_nodes,
            num_rnn_layers=args.teacher_num_rnn_layers,
            rnn_units=args.teacher_rnn_units,
            input_dim=args.input_dim,
            max_diffusion_step=args.teacher_max_diffusion_step,
            dcgru_activation=args.teacher_dcgru_activation,
            filter_type=args.filter_type,
            dropout=args.teacher_dropout,
        )
        num_classes = 1

    teacher = DCRNNModel_classification(args=targs, num_classes=num_classes, device=device)
    teacher = utils.load_model_checkpoint(args.teacher_ckpt, teacher)
    teacher = teacher.to(device)
    teacher.eval()
    for prm in teacher.parameters():
        prm.requires_grad_(False)
    log.info(f"Loaded frozen teacher from {args.teacher_ckpt} "
             f"({utils.count_parameters(teacher):,} params, frozen).")
    return teacher


def kd_loss(student_logits, teacher_logit, T):
    """Binary KD KL term (returns the raw KL, caller scales by alpha·T²).

    student_logits: (B, 2)            – student 2-class logits
    teacher_logit:  (B,)              – teacher single logit (num_classes=1)
    """
    # teacher single logit z → 2-class logits [0, z]; sigmoid(z)=softmax([0,z])[:,1]
    teacher_2 = torch.stack([torch.zeros_like(teacher_logit), teacher_logit], dim=1)
    soft_t = F.softmax(teacher_2 / T, dim=1).detach()           # (B, 2)
    log_soft_s = F.log_softmax(student_logits / T, dim=1)        # (B, 2)
    return F.kl_div(log_soft_s, soft_t, reduction="batchmean")


# ---------------------------------------------------------------------------
# Evaluation (mirrors train_chbmit.evaluate, but student forward + softmax prob)
# ---------------------------------------------------------------------------

def evaluate_student(student, dataloader, device, threshold=0.5, return_probs=False):
    from sklearn.metrics import confusion_matrix, average_precision_score

    student.eval()
    ce = nn.CrossEntropyLoss().to(device)
    y_true_all, y_prob_all = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in dataloader:
            # return_raw=True → (x_fft, y, seq_len, supports, adj, clip_id, raw)
            y   = batch[1].view(-1).to(device)
            raw = batch[6].to(device)
            logits = student(raw)                       # (B, 2)
            loss = ce(logits, y.long())
            total_loss += loss.item()
            n_batches += 1
            prob1 = F.softmax(logits, dim=1)[:, 1]
            y_prob_all.append(prob1.cpu().numpy())
            y_true_all.append(y.cpu().numpy().astype(int))

    y_prob_all = np.concatenate(y_prob_all)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = (y_prob_all >= threshold).astype(int)

    base_scores, _, _ = utils.eval_dict(
        y_pred=y_pred_all, y=y_true_all, y_prob=y_prob_all, average="binary")

    try:
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception:
        specificity = 0.0
    try:
        auprc = float(average_precision_score(y_true_all, y_prob_all))
    except Exception:
        auprc = float("nan")

    scores = OrderedDict([
        ("acc",         base_scores.get("acc", 0.0)),
        ("F1",          base_scores.get("F1", 0.0)),
        ("precision",   base_scores.get("precision", 0.0)),
        ("recall",      base_scores.get("recall", 0.0)),   # recall == sensitivity
        ("specificity", specificity),
        ("auroc",       base_scores.get("auroc", float("nan"))),
        ("auprc",       auprc),
        ("loss",        total_loss / max(n_batches, 1)),
        ("threshold",   threshold),
    ])
    if return_probs:
        return scores, y_true_all, y_prob_all
    return scores


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(student, teacher, dataloaders, args, device, save_dir):
    ce = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(student.parameters(), lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    saver = utils.CheckpointSaver(save_dir, metric_name=args.metric_name,
                                  maximize_metric=args.maximize_metric, log=log)
    epochs_no_improve = 0
    T = args.temperature

    for epoch in range(1, args.num_epochs + 1):
        student.train()
        log.info(f"Epoch {epoch}/{args.num_epochs}")
        with tqdm(total=len(dataloaders["train"].dataset)) as pbar:
            for batch in dataloaders["train"]:
                y   = batch[1].view(-1).to(device)
                raw = batch[6].to(device)

                optimizer.zero_grad()
                student_logits = student(raw)                # (B, 2)
                loss_ce = ce(student_logits, y.long())

                if args.alpha > 0:
                    x_fft       = batch[0].to(device)
                    seq_lengths = batch[2].view(-1).to(device)
                    supports    = [s.to(device) for s in batch[3]]
                    with torch.no_grad():
                        z_t = teacher(x_fft, seq_lengths, supports).view(-1)
                    loss_kd = kd_loss(student_logits, z_t, T)
                    loss = (1 - args.alpha) * loss_ce + args.alpha * (T * T) * loss_kd
                else:
                    loss = loss_ce

                loss.backward()
                nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                optimizer.step()

                pbar.update(raw.shape[0])
                pbar.set_postfix(epoch=epoch, loss=f"{loss.item():.4f}",
                                 lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        if epoch % args.eval_every == 0:
            scores, dev_true, dev_prob = evaluate_student(
                student, dataloaders["dev"], device, threshold=0.5, return_probs=True)
            log.info("[DEV] threshold=0.50  " +
                     ", ".join(f"{k}={v:.4f}" for k, v in scores.items()))

            if args.metric_name == "f1":
                best_thr, best_f1_scores, _ = sweep_threshold(dev_true, dev_prob)
                metric_val = best_f1_scores["F1"]
                log.info(f"[DEV] threshold-swept best F1={metric_val:.4f} @ thr={best_thr:.2f}")
            else:
                metric_val = {k.lower(): v for k, v in scores.items()}[args.metric_name]
                best_thr = 0.5

            improved = saver.is_best(metric_val)
            saver.save(epoch, student, optimizer, metric_val, extra={
                "best_threshold": best_thr,
                "metric_name": args.metric_name,
                "metric_val": metric_val,
            })
            epochs_no_improve = 0 if improved else epochs_no_improve + 1
            if epochs_no_improve >= args.patience:
                log.info(f"Early stopping at epoch {epoch} "
                         f"(no dev {args.metric_name} improvement for {args.patience} rounds).")
                break

        scheduler.step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    utils.seed_torch(args.rand_seed)

    save_dir = utils.get_save_dir(args.save_dir, training=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(os.path.join(save_dir, "train.log")),
                  logging.StreamHandler(sys.stdout)])
    log.info(f"Device: {device}  | Save dir: {save_dir}")
    log.info(f"KD: alpha={args.alpha}  T={args.temperature}  "
             f"({'DISTILLATION' if args.alpha > 0 else 'BASELINE (no KD)'})")

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Dataset (return_raw=True → both FFT[teacher] and raw[student] per clip) ──
    dataloaders, datasets, _ = load_dataset_chbmit_hdf5(
        hdf5_dir=args.hdf5_dir, summary_dir=args.summary_dir,
        train_batch_size=args.train_batch_size, test_batch_size=args.test_batch_size,
        win_len=args.win_len, stride=args.stride, time_step_size=args.time_step_size,
        orig_fs=args.orig_fs, fs=args.fs, fft_features=args.fft_features,
        graph_type=args.graph_type, top_k=args.top_k, filter_type=args.filter_type,
        standardize=True, num_workers=args.num_workers,
        train_ratio=args.train_ratio, val_ratio=args.val_ratio, seed=args.rand_seed,
        undersample_train=(not args.no_undersample), undersample_dev=args.undersample_dev,
        neg_ratio=args.neg_ratio, seizure_stride=args.seizure_stride,
        inspect_first_file=True, return_raw=True,
    )

    train_ds = datasets["train"]
    num_nodes = train_ds.num_nodes
    n_times = train_ds.seq_len * train_ds.step_samples      # e.g. 6 * 200 = 1200
    log.info(f"Student input: n_chans={num_nodes}  n_times={n_times}  sfreq={args.fs}")

    # ── Student (EEGNet) ────────────────────────────────────────────────────
    eegnet_kwargs = {}
    if args.eeg_f1 is not None:
        eegnet_kwargs["F1"] = args.eeg_f1
    if args.eeg_d is not None:
        eegnet_kwargs["D"] = args.eeg_d
    if args.eeg_f2 is not None:
        eegnet_kwargs["F2"] = args.eeg_f2

    student = EEGNetStudent(
        n_chans=num_nodes, n_times=n_times, sfreq=args.fs,
        n_outputs=2, drop_prob=args.drop_prob, **eegnet_kwargs,
    ).to(device)
    log.info(f"Student EEGNet: {utils.count_parameters(student):,} trainable params.")

    # ── Teacher (frozen) — only needed for KD ───────────────────────────────
    teacher = build_teacher(args, num_nodes, device) if args.alpha > 0 else None

    # ── Train ───────────────────────────────────────────────────────────────
    train(student, teacher, dataloaders, args, device, save_dir)

    best_path = os.path.join(save_dir, "best.pth.tar")
    if os.path.exists(best_path):
        student = utils.load_model_checkpoint(best_path, student).to(device)
        log.info("Loaded best student checkpoint for final evaluation.")

    # ── Dev threshold sweep + final dev/test eval ───────────────────────────
    _, dev_true, dev_prob = evaluate_student(
        student, dataloaders["dev"], device, threshold=0.5, return_probs=True)
    best_thr, best_dev, _ = sweep_threshold(dev_true, dev_prob)
    log.info(f"Best dev threshold (maximise F1): thr={best_thr:.2f}  "
             f"F1={best_dev['F1']:.4f}  precision={best_dev['precision']:.4f}  "
             f"recall={best_dev['recall']:.4f}")

    for split in ["dev", "test"]:
        s05 = evaluate_student(student, dataloaders[split], device, threshold=0.5)
        sbt = evaluate_student(student, dataloaders[split], device, threshold=best_thr)
        log.info(f"[{split.upper()}] threshold=0.50  " +
                 ", ".join(f"{k}={v:.4f}" for k, v in s05.items()))
        log.info(f"[{split.upper()}] threshold={best_thr:.2f}  " +
                 ", ".join(f"{k}={v:.4f}" for k, v in sbt.items()))


if __name__ == "__main__":
    main()
