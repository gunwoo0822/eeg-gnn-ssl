"""
Train DCRNNModel_classification on CHB-MIT using the existing DCRNN implementation.

Supports two data backends (mutually exclusive):
  --npz_dir   : preprocessed NPZ windows from preprocess_chbmit.py (original path)
  --hdf5_dir  : raw 256 Hz CHB-MIT HDF5 files (lazy-loading, memory-efficient)

Quick-start (HDF5 backend – VESSL):
    python train_chbmit.py \\
        --hdf5_dir  /vessl/data/chbmit_hdf5 \\
        --summary_dir /vessl/data/chbmit \\
        --save_dir  /vessl/output/chbmit_run \\
        --do_train \\
        --num_epochs 1 \\
        --train_batch_size 4 \\
        --test_batch_size  4 \\
        --num_workers 0 \\
        --graph_type combined

Quick-start (NPZ backend – original):
    python train_chbmit.py \\
        --npz_dir /data/chbmit_npz \\
        --save_dir /tmp/chbmit_run \\
        --do_train \\
        --num_epochs 1 \\
        --train_batch_size 4 \\
        --test_batch_size  4 \\
        --num_workers 0 \\
        --graph_type combined

Expected tensor shapes at each stage
-------------------------------------
  Raw window (HDF5 or NPZ)  (23, 512)  = (C, T)
  Dataset __getitem__        (2, 23, 256) = (seq_len, num_nodes, input_dim)
  DataLoader batch           (B, 2, 23, 256)
  Inside encoder (transposed)(2, B, 23, 256)
  Encoder hidden             (num_layers, B, 23*64)
  FC output                  (B, 23, 1)
  Max-pool over nodes        (B, 1) → view(-1) → (B,)
  supports[0]                (B, 23, 23)
"""

import os
import sys
import json
import logging
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
from pathlib import Path
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))

import utils
from data.dataloader_chbmit import load_dataset_chbmit
from data.dataloader_chbmit_hdf5 import load_dataset_chbmit_hdf5
from model.model import DCRNNModel_classification

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Arguments
# ---------------------------------------------------------------------------

def get_args():
    parser = argparse.ArgumentParser("Train DCRNN on CHB-MIT.")

    # I/O – data source (exactly one of --npz_dir or --hdf5_dir is required)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--npz_dir", type=str, default=None,
                     help="Dir with per-patient windows.npz (preprocess_chbmit.py output).")
    src.add_argument("--hdf5_dir", type=str, default=None,
                     help="Dir with 256 Hz CHB-MIT HDF5 files (lazy-loaded, VESSL-friendly).")
    parser.add_argument("--summary_dir", type=str, default=None,
                        help="Dir with CHB-MIT *-summary.txt files (needed for seizure "
                             "labels when HDF5 files lack embedded annotations).")
    parser.add_argument("--save_dir", type=str, default="./chbmit_runs",
                        help="Directory for checkpoints and logs.")
    parser.add_argument("--do_train", action="store_true", default=False,
                        help="Run training.")
    parser.add_argument("--load_model_path", type=str, default=None,
                        help="Checkpoint to load (for evaluation or resuming).")

    # Data
    parser.add_argument("--fs", type=int, default=256,
                        help="EEG sampling rate in Hz (default 256).")
    parser.add_argument("--win_len", type=int, default=2,
                        help="Window length in seconds (default 2).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Sliding window stride in seconds (default 1). HDF5 only.")
    parser.add_argument("--time_step_size", type=int, default=1,
                        help="Seconds per DCRNN time step (default 1).")
    parser.add_argument("--graph_type", choices=["individual", "combined"],
                        default="individual",
                        help="'individual'=cross-corr, 'combined'=FC fallback.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top-k neighbours for cross-correlation graph.")
    parser.add_argument("--train_ratio", type=float, default=0.70,
                        help="Fraction of patients used for training.")
    parser.add_argument("--val_ratio",   type=float, default=0.15,
                        help="Fraction of patients used for validation.")
    parser.add_argument("--rand_seed",   type=int, default=123)

    # Model (DCRNN)
    parser.add_argument("--num_nodes", type=int, default=0,
                        help="EEG channel count. 0 = auto-detect from dataset.")
    parser.add_argument("--num_rnn_layers",     type=int, default=2)
    parser.add_argument("--rnn_units",          type=int, default=64)
    parser.add_argument("--max_diffusion_step", type=int, default=2)
    parser.add_argument("--dcgru_activation",   choices=["tanh", "relu"],
                        default="tanh")
    parser.add_argument("--dropout",            type=float, default=0.0)
    parser.add_argument("--num_classes",        type=int, default=1,
                        help="1 for binary seizure detection.")

    # Training
    parser.add_argument("--num_epochs",       type=int,   default=50)
    parser.add_argument("--train_batch_size", type=int,   default=32)
    parser.add_argument("--test_batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--lr_init",          type=float, default=3e-4)
    parser.add_argument("--l2_wd",            type=float, default=5e-4)
    parser.add_argument("--max_grad_norm",    type=float, default=5.0)
    parser.add_argument("--eval_every",       type=int,   default=1)
    parser.add_argument("--patience",         type=int,   default=10)
    parser.add_argument("--metric_name",
                        choices=["auroc", "F1", "acc", "loss"],
                        default="auroc",
                        help="Metric used to select best checkpoint.")

    args = parser.parse_args()

    # Derived fields expected by DCRNNModel_classification
    args.input_dim   = args.fs * args.time_step_size    # raw samples per step (256)
    args.max_seq_len = args.win_len // args.time_step_size  # DCRNN seq_len  (2)
    args.task        = "detection"
    args.model_name  = "dcrnn"
    args.maximize_metric = (args.metric_name != "loss")

    if args.graph_type == "individual":
        args.filter_type = "dual_random_walk"
    else:
        args.filter_type = "laplacian"

    return args


# ---------------------------------------------------------------------------
# Evaluate
# ---------------------------------------------------------------------------

def evaluate(model, dataloader, device, args, debug_shapes=False):
    """Run model on dataloader; return metrics dict."""
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss().to(device)

    y_true_all, y_prob_all = [], []
    total_loss, n_batches  = 0.0, 0

    with torch.no_grad():
        for batch_idx, (x, y, seq_lengths, supports, _, _) in enumerate(dataloader):
            x           = x.to(device)
            y           = y.view(-1).to(device)
            seq_lengths = seq_lengths.view(-1).to(device)
            for i in range(len(supports)):
                supports[i] = supports[i].to(device)

            if debug_shapes and batch_idx == 0:
                print(f"\n[DEBUG eval] x={tuple(x.shape)}  y={tuple(y.shape)}  "
                      f"seq_len={tuple(seq_lengths.shape)}  "
                      f"supports[0]={tuple(supports[0].shape)}")

            logits = model(x, seq_lengths, supports).view(-1)
            loss   = loss_fn(logits, y)
            total_loss += loss.item()
            n_batches  += 1

            y_prob_all.append(torch.sigmoid(logits).cpu().numpy())
            y_true_all.append(y.cpu().numpy().astype(int))

    y_prob_all = np.concatenate(y_prob_all)
    y_true_all = np.concatenate(y_true_all)
    y_pred_all = (y_prob_all >= 0.5).astype(int)

    scores, _, _ = utils.eval_dict(
        y_pred=y_pred_all, y=y_true_all, y_prob=y_prob_all, average="binary")
    scores["loss"] = total_loss / max(n_batches, 1)
    return scores


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(model, dataloaders, args, device, save_dir):
    """Main training loop with early stopping."""
    loss_fn   = nn.BCEWithLogitsLoss().to(device)
    optimizer = optim.Adam(model.parameters(),
                           lr=args.lr_init, weight_decay=args.l2_wd)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    saver     = utils.CheckpointSaver(
        save_dir,
        metric_name=args.metric_name,
        maximize_metric=args.maximize_metric,
        log=log)

    prev_val_loss  = 1e10
    patience_count = 0
    first_debug    = True   # print shapes on the very first training batch

    for epoch in range(1, args.num_epochs + 1):
        model.train()
        log.info(f"Epoch {epoch}/{args.num_epochs}")

        with tqdm(total=len(dataloaders["train"].dataset)) as pbar:
            for x, y, seq_lengths, supports, _, _ in dataloaders["train"]:
                x           = x.to(device)
                y           = y.view(-1).to(device)
                seq_lengths = seq_lengths.view(-1).to(device)
                for i in range(len(supports)):
                    supports[i] = supports[i].to(device)

                # ── shape debug (first batch only) ──────────────────────
                if first_debug:
                    print(f"\n[DEBUG] First training batch shapes:")
                    print(f"  x          : {tuple(x.shape)}"
                          f"  = (batch, seq_len={args.max_seq_len}, "
                          f"num_nodes={args.num_nodes}, "
                          f"input_dim={args.input_dim})")
                    print(f"  y          : {tuple(y.shape)}")
                    print(f"  seq_lengths: {tuple(seq_lengths.shape)}")
                    print(f"  #supports  : {len(supports)}, "
                          f"supports[0]: {tuple(supports[0].shape)}")
                    first_debug = False
                # ────────────────────────────────────────────────────────

                optimizer.zero_grad()
                logits = model(x, seq_lengths, supports).view(-1)
                loss   = loss_fn(logits, y)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()

                pbar.update(x.shape[0])
                pbar.set_postfix(
                    epoch=epoch,
                    loss=f"{loss.item():.4f}",
                    lr=f"{optimizer.param_groups[0]['lr']:.2e}")

        # ── evaluation ──────────────────────────────────────────────────
        if epoch % args.eval_every == 0:
            scores = evaluate(model, dataloaders["dev"], device, args,
                              debug_shapes=(epoch == 1))
            metric_val = scores[args.metric_name]
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
            log.info(f"[Dev] {scores_str}")

            saver.save(epoch, model, optimizer, metric_val)

            if scores["loss"] < prev_val_loss:
                patience_count = 0
            else:
                patience_count += 1
            prev_val_loss = scores["loss"]

            if patience_count >= args.patience:
                log.info("Early stopping triggered.")
                break

        scheduler.step()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = get_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    utils.seed_torch(args.rand_seed)

    # Save directory and logging
    save_dir = utils.get_save_dir(args.save_dir, training=args.do_train)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(save_dir, "train.log")),
            logging.StreamHandler(sys.stdout),
        ])
    log.info(f"Device: {device}")
    log.info(f"Save dir: {save_dir}")

    with open(os.path.join(save_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=2, default=str)

    # ── Dataset ─────────────────────────────────────────────────────────
    if args.hdf5_dir:
        log.info(f"Loading CHB-MIT from HDF5 (lazy) → {args.hdf5_dir}")
        dataloaders, datasets, scaler = load_dataset_chbmit_hdf5(
            hdf5_dir=args.hdf5_dir,
            summary_dir=args.summary_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            win_len=args.win_len,
            stride=args.stride,
            time_step_size=args.time_step_size,
            fs=args.fs,
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            standardize=True,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.rand_seed,
            inspect_first_file=True,
        )
    else:
        log.info(f"Loading CHB-MIT from NPZ → {args.npz_dir}")
        dataloaders, datasets, scaler = load_dataset_chbmit(
            npz_dir=args.npz_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size,
            fs=args.fs,
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            standardize=True,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.rand_seed,
        )

    # Auto-detect num_nodes from the dataset if not specified
    train_ds = datasets["train"]
    if args.num_nodes == 0 or args.num_nodes != train_ds.num_nodes:
        log.info(f"Setting num_nodes: {args.num_nodes} → {train_ds.num_nodes}")
        args.num_nodes = train_ds.num_nodes

    log.info(
        f"DCRNN config: num_nodes={args.num_nodes}  "
        f"seq_len={args.max_seq_len}  input_dim={args.input_dim}  "
        f"filter_type={args.filter_type}"
    )

    # ── Model ────────────────────────────────────────────────────────────
    model = DCRNNModel_classification(
        args=args, num_classes=args.num_classes, device=device)

    if args.load_model_path:
        model = utils.load_model_checkpoint(args.load_model_path, model)
        log.info(f"Loaded checkpoint: {args.load_model_path}")

    log.info(f"Trainable parameters: {utils.count_parameters(model):,}")
    model = model.to(device)

    # ── Train ─────────────────────────────────────────────────────────────
    if args.do_train:
        train(model, dataloaders, args, device, save_dir)
        best_path = os.path.join(save_dir, "best.pth.tar")
        if os.path.exists(best_path):
            model = utils.load_model_checkpoint(best_path, model)
            model = model.to(device)
            log.info("Loaded best checkpoint for final evaluation.")

    # ── Final evaluation ──────────────────────────────────────────────────
    for split in ["dev", "test"]:
        scores     = evaluate(model, dataloaders[split], device, args)
        scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
        log.info(f"[{split.upper()}] {scores_str}")


if __name__ == "__main__":
    main()
