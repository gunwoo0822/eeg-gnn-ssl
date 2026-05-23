"""
Train DCRNNModel_classification on CHB-MIT.

Faithful re-implementation of the Tang et al. ICLR 2022 Dist-DCRNN
seizure-detection pipeline, adapted for CHB-MIT (for use as teacher model
in Knowledge Distillation experiments).

Paper: "Self-Supervised Graph Neural Networks for Improved EEG Seizure Analysis"
       Tang et al., ICLR 2022.  Appendix A / E.

Supports two data backends (mutually exclusive):
  --hdf5_dir  : raw CHB-MIT HDF5 files at 256 Hz (lazy-loaded, paper-faithful)
  --npz_dir   : preprocessed NPZ windows (legacy / debugging)

Quick-start (HDF5, paper defaults):
    python train_chbmit.py \\
        --hdf5_dir    /vessl/data/chbmit_hdf5 \\
        --summary_dir /vessl/data/chbmit \\
        --save_dir    /vessl/output/chbmit_run \\
        --do_train \\
        --graph_type individual

Expected tensor shapes (paper pipeline)
-----------------------------------------
  HDF5 slice                 (C, win_len × 256)   e.g. (23, 3072) @256Hz
  After resample 256→200Hz   (C, win_len × 200)   e.g. (23, 2400) @200Hz
  Reshaped                   (seq_len, C, 200)     e.g. (12, 23, 200)
  After FFT log-amplitude    (seq_len, C, M)       e.g. (12, 23, 100) ← model input
  DataLoader batch           (B, 12, 23, 100)
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
    parser.add_argument("--orig_fs", type=int, default=256,
                        help="Native sampling rate of HDF5 files (default 256 Hz).")
    parser.add_argument("--fs", type=int, default=200,
                        help="Target sampling rate after resampling (paper: 200 Hz).")
    parser.add_argument("--win_len", type=int, default=12,
                        help="Clip length in seconds (paper: 12 for fast, 60 for slow).")
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride in seconds. "
                             "None (default) = non-overlapping (stride = win_len).")
    parser.add_argument("--time_step_size", type=int, default=1,
                        help="Seconds per DCRNN time step (paper: 1 s).")
    parser.add_argument("--fft_features", type=int, default=100,
                        help="Log-amplitude FFT bins per segment (paper: 100 for 200 Hz).")
    parser.add_argument("--graph_type", choices=["individual", "combined"],
                        default="individual",
                        help="'individual'=cross-corr (paper), 'combined'=FC fallback.")
    parser.add_argument("--top_k", type=int, default=3,
                        help="Top-k neighbours for cross-correlation graph (paper: τ=3).")
    parser.add_argument("--no_undersample", action="store_true", default=False,
                        help="Disable negative undersampling on train set entirely.")
    parser.add_argument("--neg_ratio", type=int, default=1,
                        help="Negatives kept per positive during train undersampling "
                             "(default 1 → positive:negative = 1:1). "
                             "E.g. --neg_ratio 3 gives 1:3. "
                             "Ignored when --no_undersample is set. "
                             "dev/test are never affected.")
    parser.add_argument("--undersample_dev", action="store_true", default=False,
                        help="Apply 1:1 negative undersampling to the dev set as well. "
                             "Default OFF (original distribution). "
                             "test set is NEVER undersampled regardless of this flag. "
                             "Threshold sweep uses the (possibly undersampled) dev set.")
    parser.add_argument("--seizure_stride", type=float, default=None,
                        help="Dense stride in seconds for seizure-region oversampling "
                             "on the train set (e.g. 1 or 2). "
                             "None = disabled (non-overlapping everywhere). "
                             "Must be < win_len. dev/test are never affected.")
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

    # Training  (paper defaults: §Appendix E)
    parser.add_argument("--num_epochs",       type=int,   default=100)
    parser.add_argument("--train_batch_size", type=int,   default=40)
    parser.add_argument("--test_batch_size",  type=int,   default=64)
    parser.add_argument("--num_workers",      type=int,   default=4)
    parser.add_argument("--lr_init",          type=float, default=1e-4,
                        help="Initial learning rate (paper detection: 1e-4).")
    parser.add_argument("--l2_wd",            type=float, default=0.0,
                        help="L2 weight decay (paper does not mention WD → 0).")
    parser.add_argument("--max_grad_norm",    type=float, default=5.0)
    parser.add_argument("--eval_every",       type=int,   default=1)
    parser.add_argument("--patience",         type=int,   default=5,
                        help="Early-stopping patience in epochs (paper: 5).")
    parser.add_argument("--metric_name",
                        choices=["auroc", "F1", "acc", "loss"],
                        default="auroc",
                        help="Metric used to select best checkpoint.")

    args = parser.parse_args()

    # Derived fields expected by DCRNNModel_classification
    # input_dim = M = FFT bins per 1-second segment (paper: 100 for 200 Hz)
    args.input_dim   = args.fft_features                  # 100
    args.max_seq_len = args.win_len // args.time_step_size # DCRNN seq_len (12 or 60)
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

def evaluate(model, dataloader, device, args,
             debug_shapes=False, threshold=0.5, return_probs=False):
    """
    Run model on dataloader; return ordered metrics dict.

    Metrics (in display order):
        acc, F1, precision, recall, specificity, auroc, auprc, loss, threshold

    specificity = TN / (TN + FP)   – meaningful for imbalanced datasets
    auprc       = area under precision-recall curve (average_precision_score)
                  threshold-independent; critical when positives are rare
    """
    from sklearn.metrics import confusion_matrix, average_precision_score
    from collections import OrderedDict

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
    y_pred_all = (y_prob_all >= threshold).astype(int)

    # ── Base metrics from utils.eval_dict ────────────────────────────────
    base_scores, _, _ = utils.eval_dict(
        y_pred=y_pred_all, y=y_true_all, y_prob=y_prob_all, average="binary")

    # ── Specificity: TN / (TN + FP) ──────────────────────────────────────
    try:
        cm = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
        tn, fp = int(cm[0, 0]), int(cm[0, 1])
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    except Exception:
        specificity = 0.0

    # ── AUPRC (threshold-independent) ────────────────────────────────────
    try:
        auprc = float(average_precision_score(y_true_all, y_prob_all))
    except Exception:
        auprc = float("nan")

    # ── Assemble in desired display order ─────────────────────────────────
    scores = OrderedDict([
        ("acc",         base_scores.get("acc",       0.0)),
        ("F1",          base_scores.get("F1",        0.0)),
        ("precision",   base_scores.get("precision", 0.0)),
        ("recall",      base_scores.get("recall",    0.0)),
        ("specificity", specificity),
        ("auroc",       base_scores.get("auroc",     float("nan"))),
        ("auprc",       auprc),
        ("loss",        total_loss / max(n_batches, 1)),
        ("threshold",   threshold),
    ])

    if return_probs:
        return scores, y_true_all, y_prob_all
    return scores


def sweep_threshold(y_true, y_prob, thresholds=None):
    """
    Sweep classification thresholds and return best by F1.

    Returns (best_threshold, best_scores, all_rows) where all_rows is a list
    of dicts with keys threshold/F1/precision/recall/acc.
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    best_thresh = 0.5
    best_f1     = -1.0
    best_scores = {}
    all_rows    = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1   = f1_score(y_true, y_pred, average="binary", zero_division=0)
        prec = precision_score(y_true, y_pred, average="binary", zero_division=0)
        rec  = recall_score(y_true, y_pred, average="binary", zero_division=0)
        acc  = accuracy_score(y_true, y_pred)
        row  = dict(threshold=thr, F1=f1, precision=prec, recall=rec, acc=acc)
        all_rows.append(row)
        if f1 > best_f1:
            best_f1     = f1
            best_thresh = thr
            best_scores = row

    return best_thresh, best_scores, all_rows


# ---------------------------------------------------------------------------
# Train
# ---------------------------------------------------------------------------

def train(model, dataloaders, args, device, save_dir):
    """Main training loop with early stopping."""
    loss_fn = nn.BCEWithLogitsLoss().to(device)   # no pos_weight; use undersampling
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
            # Train evaluation (threshold=0.5; helps diagnose under-/over-fitting)
            train_scores = evaluate(model, dataloaders["train"], device, args,
                                    threshold=0.5)
            train_str = ", ".join(f"{k}={v:.4f}" for k, v in train_scores.items())
            log.info(f"[TRAIN] threshold=0.50  {train_str}")

            # Dev evaluation (controls checkpoint saving and early stopping)
            scores = evaluate(model, dataloaders["dev"], device, args,
                              debug_shapes=(epoch == 1))
            metric_val = scores[args.metric_name]
            scores_str = ", ".join(f"{k}={v:.4f}" for k, v in scores.items())
            log.info(f"[DEV]   threshold=0.50  {scores_str}")

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
            orig_fs=args.orig_fs,
            fs=args.fs,
            fft_features=args.fft_features,
            graph_type=args.graph_type,
            top_k=args.top_k,
            filter_type=args.filter_type,
            standardize=True,
            num_workers=args.num_workers,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            seed=args.rand_seed,
            undersample_train=(not args.no_undersample),
            undersample_dev=args.undersample_dev,
            neg_ratio=args.neg_ratio,
            seizure_stride=args.seizure_stride,
            inspect_first_file=True,
        )
    else:
        log.info(f"Loading CHB-MIT from NPZ → {args.npz_dir}")
        log.warning(
            "NPZ backend uses raw EEG (no FFT). "
            "For paper-faithful preprocessing use --hdf5_dir. "
            "Pass --fs 256 --orig_fs 256 if NPZ files are at 256 Hz."
        )
        dataloaders, datasets, scaler = load_dataset_chbmit(
            npz_dir=args.npz_dir,
            train_batch_size=args.train_batch_size,
            test_batch_size=args.test_batch_size,
            time_step_size=args.time_step_size,
            fs=args.orig_fs,   # NPZ uses orig_fs (raw EEG, no resample)
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

    # ── Train set stats ───────────────────────────────────────────────────
    targets  = train_ds.targets()
    n_pos    = int(sum(targets))
    n_neg    = len(targets) - n_pos
    sz_stride_str = (
        f"seizure_stride={args.seizure_stride}s" if args.seizure_stride else "no oversampling"
    )
    undersample_str = (
        f"no undersampling" if args.no_undersample
        else f"1:{args.neg_ratio} undersampling"
    )
    actual_ratio = n_neg / max(n_pos, 1)
    log.info(
        f"Train set final  ({sz_stride_str}, {undersample_str}): "
        f"total={len(targets)} | pos={n_pos} | neg={n_neg} "
        f"(ratio 1:{actual_ratio:.2f}) | "
        f"pos_frac={100*n_pos/max(len(targets),1):.1f}%"
    )

    # ── Dev set stats ─────────────────────────────────────────────────────
    dev_ds = datasets["dev"]
    dev_tgt = dev_ds.targets()
    dev_pos = int(sum(dev_tgt))
    dev_neg = len(dev_tgt) - dev_pos
    if args.undersample_dev:
        dev_actual_ratio = dev_neg / max(dev_pos, 1)
        dev_dist_str = (
            f"undersampled 1:{args.neg_ratio} "
            f"(actual 1:{dev_actual_ratio:.2f})"
        )
    else:
        dev_dist_str = "original distribution"
    log.info(
        f"Dev  set ({dev_dist_str}): "
        f"total={len(dev_tgt)} | pos={dev_pos} ({100*dev_pos/max(len(dev_tgt),1):.2f}%) | neg={dev_neg}"
    )
    log.info(
        f"DCRNN config: num_nodes={args.num_nodes}  "
        f"seq_len={args.max_seq_len}  input_dim={args.input_dim}  "
        f"(FFT features M={args.fft_features})  filter_type={args.filter_type}"
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

    # ── Threshold sweep on dev set ────────────────────────────────────────
    dev_sweep_note = (
        f"undersampled dev (1:{args.neg_ratio})"
        if args.undersample_dev else "original dev distribution"
    )
    log.info(f"Running threshold sweep on dev set [{dev_sweep_note}] …")
    dev_scores_05, dev_true, dev_prob = evaluate(
        model, dataloaders["dev"], device, args,
        threshold=0.5, return_probs=True)
    best_thresh, best_dev_scores, sweep_rows = sweep_threshold(dev_true, dev_prob)
    log.info(
        f"Best threshold ({dev_sweep_note}, maximise dev F1): "
        f"threshold={best_thresh:.2f}  "
        f"F1={best_dev_scores['F1']:.4f}  "
        f"precision={best_dev_scores['precision']:.4f}  "
        f"recall={best_dev_scores['recall']:.4f}"
    )

    # ── Final evaluation ──────────────────────────────────────────────────
    for split in ["dev", "test"]:
        if split == "dev":
            scores_05  = dev_scores_05
            scores_bt, _, _ = evaluate(
                model, dataloaders[split], device, args,
                threshold=best_thresh, return_probs=True)
        else:
            scores_05, _, _ = evaluate(
                model, dataloaders[split], device, args,
                threshold=0.5, return_probs=True)
            scores_bt, _, _ = evaluate(
                model, dataloaders[split], device, args,
                threshold=best_thresh, return_probs=True)

        log.info(f"[{split.upper()}] threshold=0.50  " +
                 ", ".join(f"{k}={v:.4f}" for k, v in scores_05.items()))
        log.info(f"[{split.upper()}] threshold={best_thresh:.2f}  " +
                 ", ".join(f"{k}={v:.4f}" for k, v in scores_bt.items()))


if __name__ == "__main__":
    main()
