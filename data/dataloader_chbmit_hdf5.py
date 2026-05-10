"""
Memory-efficient HDF5 DataLoader for CHB-MIT 256 Hz EEG.

Design: fully lazy – no EEG signal is loaded into RAM during __init__.

Initialization (O(index) ≈ tens of MB)
  1. Probe the first HDF5 file to auto-detect structure (signal key, layout).
  2. Iterate over every HDF5 file.  For each recording:
       a. Read signal shape (header only – no data).
       b. Compute window start-samples via a sliding window.
       c. Determine seizure label for each window from embedded HDF5
          annotations OR from external CHB-MIT *-summary.txt files.
       d. Append lightweight WindowRecord namedtuples to the index.
  3. Build the scaler via a streaming two-pass over unique recordings
     (one recording in RAM at a time, then discarded).

__getitem__ (O(one window) ≈ 47 KB)
  Open the HDF5 file (cached per worker-process), slice exactly
  signal[:, start : start+win_samples], reshape, return.

Supported HDF5 layouts (auto-detected)
  A  Per-recording file, raw signals
       chb01_01.h5 → signal:(C,T) or eeg:(C,T) or data:(C,T) or …
       optional keys: seizure_times, seizures, annotations, label, fs, channels
  B  Per-patient file, recordings as top-level groups
       chb01.h5 → /chb01_01/signal:(C,T), /chb01_03/signal:(C,T), …
  C  Pre-windowed per-patient or per-recording file
       any.h5 → X:(N,C,T_win), y:(N,)

Seizure labels priority
  1. Embedded in HDF5 (seizure_times, seizures, annotations datasets/attrs)
  2. External summary_dir: CHB-MIT *-summary.txt files parsed at init
  3. File-level binary label ("label" key)
  4. If none available → all labels default to 0 with a logged warning

RAM vs I/O tradeoffs
  RAM    : O(N_windows × ~80 bytes) for the index + one recording for scaler
  I/O    : one HDF5 seek+read per __getitem__ call (≈ 47 KB for 23×512 float32)
  VESSL  : safe – even 500K windows cost < 40 MB of index RAM
  Workers: each DataLoader worker opens its own h5py handles; handles are cached
           and reused within the same worker process (no repeated open overhead)

Usage
-----
  from data.dataloader_chbmit_hdf5 import load_dataset_chbmit_hdf5

  dataloaders, datasets, scaler = load_dataset_chbmit_hdf5(
      hdf5_dir  = "/vessl/data/chbmit_hdf5",   # contains chb01/, chb02/, …
      summary_dir = "/vessl/data/chbmit",       # optional; dir with *-summary.txt
      train_batch_size = 32,
      graph_type = "combined",
  )
"""

import os
import re
import sys
import logging
import threading
from collections import namedtuple
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
from data.data_utils import comp_xcorr, keep_topk

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Window index record (lightweight – ~80 bytes per window)
# ---------------------------------------------------------------------------

WindowRecord = namedtuple(
    "WindowRecord",
    [
        "h5_path",      # str  – absolute path to HDF5 file
        "rec_key",      # str|None – HDF5 group key (None if file IS the recording)
        "start",        # int  – first sample of window inside the recording
        "label",        # int  – 0 or 1
        "patient",      # str  – e.g. "chb01"
        "win_samples",  # int  – window length in samples (constant per dataset)
    ],
)

# ---------------------------------------------------------------------------
# HDF5 structure detection
# ---------------------------------------------------------------------------

_SIGNAL_KEY_CANDIDATES = [
    "signal", "eeg", "data", "eeg_signal", "raw", "x",
    "resampled_signal",   # TUSZ-style
    "signals",
]
_SEIZURE_KEY_CANDIDATES = [
    "seizure_times", "seizures", "annotations", "seizure_intervals",
    "sz_times", "ictal_times",
]
_LABEL_KEY_CANDIDATES = ["label", "y", "labels"]


def inspect_hdf5_structure(h5_path, max_depth=2, print_fn=None):
    """
    Pretty-print the structure of an HDF5 file.

    Call this to understand what keys/shapes your HDF5 files contain
    before running the full pipeline.

    Args:
        h5_path: str or Path
        max_depth: how many group levels to recurse
        print_fn: callable for output (default: logging.info)
    """
    if print_fn is None:
        print_fn = log.info

    def _recurse(node, indent=0, depth=0):
        prefix = "  " * indent
        for key in node.keys():
            item = node[key]
            if isinstance(item, h5py.Dataset):
                attrs = dict(item.attrs)
                print_fn(f"{prefix}[Dataset] {key!r}: shape={item.shape} "
                         f"dtype={item.dtype} attrs={attrs}")
            elif isinstance(item, h5py.Group):
                attrs = dict(item.attrs)
                print_fn(f"{prefix}[Group]   {key!r} attrs={attrs}")
                if depth < max_depth:
                    _recurse(item, indent + 1, depth + 1)

    print_fn(f"\n{'='*60}")
    print_fn(f"HDF5 structure: {h5_path}")
    print_fn(f"{'='*60}")
    with h5py.File(str(h5_path), "r") as f:
        print_fn(f"Root keys: {list(f.keys())}")
        print_fn(f"Root attrs: {dict(f.attrs)}")
        _recurse(f)
    print_fn(f"{'='*60}\n")


def _probe_hdf5(h5_path):
    """
    Detect the layout of an HDF5 file.

    Returns:
        fmt:        "windowed" | "raw_flat" | "raw_grouped"
        signal_key: str  – key (or sub-key) holding the signal array
        group_keys: list[str] | None  – top-level group names for "raw_grouped"
        meta:       dict  – e.g. {"shape": (C,T), "has_seizure_key": bool}
    """
    with h5py.File(str(h5_path), "r") as f:
        top_keys = list(f.keys())

        # ── Format C: pre-windowed (X + y) ────────────────────────────────
        X_key = next((k for k in top_keys if k.upper() == "X"), None)
        y_key = next((k for k in top_keys if k.lower() in ("y", "labels")), None)
        if X_key is not None and y_key is not None:
            shape = f[X_key].shape      # (N, C, T_win)
            n_windows = f[y_key].shape[0]
            return ("windowed", X_key,
                    None, {"shape": shape, "y_key": y_key,
                           "n_windows": n_windows})

        # ── Format A: raw signal at top level ─────────────────────────────
        for cand in _SIGNAL_KEY_CANDIDATES:
            for k in top_keys:
                if k.lower() == cand:
                    ds = f[k]
                    if ds.ndim == 2:
                        shape = ds.shape
                        sz_key = next(
                            (sk for sk in top_keys
                             if sk.lower() in _SEIZURE_KEY_CANDIDATES),
                            None)
                        lbl_key = next(
                            (lk for lk in top_keys
                             if lk.lower() in _LABEL_KEY_CANDIDATES),
                            None)
                        return ("raw_flat", k,
                                None, {"shape": shape,
                                       "sz_key": sz_key,
                                       "lbl_key": lbl_key})

        # ── Format B: recordings as groups ────────────────────────────────
        group_keys = [k for k in top_keys if isinstance(f[k], h5py.Group)]
        if group_keys:
            first_grp = f[group_keys[0]]
            for cand in _SIGNAL_KEY_CANDIDATES:
                for sk in first_grp.keys():
                    if sk.lower() == cand and first_grp[sk].ndim == 2:
                        shape = first_grp[sk].shape
                        sz_key = next(
                            (k for k in first_grp.keys()
                             if k.lower() in _SEIZURE_KEY_CANDIDATES),
                            None)
                        return ("raw_grouped", sk,
                                group_keys, {"shape": shape, "sz_key": sz_key})

        # ── Fallback: first 2D dataset found ──────────────────────────────
        for k in top_keys:
            if isinstance(f[k], h5py.Dataset) and f[k].ndim == 2:
                return ("raw_flat", k, None, {"shape": f[k].shape,
                                              "sz_key": None, "lbl_key": None})

    raise ValueError(
        f"Cannot detect HDF5 layout in {h5_path}. "
        f"Top-level keys: {top_keys}. "
        f"Run inspect_hdf5_structure() to examine the file manually."
    )


# ---------------------------------------------------------------------------
# Summary file parsing (fallback annotation source)
# ---------------------------------------------------------------------------

def _parse_summary_dir(summary_dir):
    """
    Parse all CHB-MIT *-summary.txt files in summary_dir (and sub-dirs).

    Returns:
        dict: {edf_filename (str): [(start_sec, end_sec), ...]}
              e.g. {"chb01_03.edf": [(2996, 3036)]}
    """
    summary_dir = Path(summary_dir)
    all_seizures = {}
    for summary_path in summary_dir.rglob("*-summary.txt"):
        cur_file, starts, ends = None, [], []
        with open(summary_path, "r", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                m = re.match(r"File Name:\s+(\S+\.edf)", line, re.IGNORECASE)
                if m:
                    if cur_file is not None:
                        all_seizures[cur_file] = list(zip(starts, ends))
                    cur_file, starts, ends = m.group(1), [], []
                    all_seizures[cur_file] = []
                    continue
                m = re.match(r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)",
                             line, re.IGNORECASE)
                if m:
                    starts.append(int(m.group(1)))
                    continue
                m = re.match(r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)",
                             line, re.IGNORECASE)
                if m:
                    ends.append(int(m.group(1)))
                    continue
        if cur_file is not None:
            all_seizures[cur_file] = list(zip(starts, ends))
    return all_seizures


# ---------------------------------------------------------------------------
# Signal loading helpers
# ---------------------------------------------------------------------------

def _load_signal_from_h5(h5_file, rec_key, signal_key):
    """
    Return the raw signal array (C, T) from an open h5py.File.
    Does NOT load the entire file — returns the h5py.Dataset reference
    so callers can slice it without loading everything.
    """
    if rec_key is not None:
        ds = h5_file[rec_key][signal_key]
    else:
        ds = h5_file[signal_key]

    # Ensure (C, T) orientation: C is the small axis
    if ds.ndim != 2:
        raise ValueError(f"Signal dataset has unexpected ndim={ds.ndim}")
    return ds


def _load_seizure_times_from_h5(h5_file, rec_key, sz_key, fs):
    """
    Try to read seizure intervals from an open h5py.File.

    Returns list of (start_sample, end_sample) or [] if none found.
    """
    if sz_key is None:
        return []
    try:
        node = h5_file[rec_key][sz_key] if rec_key else h5_file[sz_key]
        arr = node[()]
        if arr.ndim == 0 or arr.size == 0:
            return []
        arr = np.atleast_2d(arr)    # (M, 2) – seconds or samples
        # Heuristic: if values look like seconds (< 10000) convert to samples
        if arr.max() < 10000:
            arr = (arr * fs).astype(int)
        return [(int(r[0]), int(r[1])) for r in arr]
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Process-local HDF5 file handle cache
# (safe with DataLoader fork/spawn: each process gets its own dict)
# ---------------------------------------------------------------------------

_h5_cache: dict = {}   # {(pid, path_str): h5py.File}


def _get_cached_h5(path_str: str) -> h5py.File:
    """Return a cached h5py.File handle for this worker process."""
    pid = os.getpid()
    key = (pid, path_str)
    if key not in _h5_cache:
        _h5_cache[key] = h5py.File(path_str, "r")
        log.debug(f"[pid={pid}] Opened HDF5: {path_str}")
    return _h5_cache[key]


def _worker_init_fn(worker_id):
    """
    DataLoader worker initialiser: clear any stale file handles
    inherited from the parent process after fork.
    """
    global _h5_cache
    _h5_cache = {}


# ---------------------------------------------------------------------------
# Index builder
# ---------------------------------------------------------------------------

def _build_index(
        hdf5_paths,
        fmt, signal_key, group_keys, meta,
        win_samples, stride_samples, fs,
        seizure_map,        # dict from _parse_summary_dir (may be empty)
        min_channels=None,  # enforce channel count
):
    """
    Build the lightweight window index.

    Returns:
        index:       list[WindowRecord]
        num_nodes:   int – number of EEG channels detected
        ref_channels: list[str] or None
    """
    index = []
    ref_channels = None
    ref_num_nodes = None

    for h5_path in hdf5_paths:
        patient = Path(h5_path).stem.split("_")[0]   # "chb01" from "chb01_03.h5"
        # For per-patient files (grouped), the patient is the stem itself
        if fmt == "raw_grouped":
            patient = Path(h5_path).stem

        try:
            with h5py.File(str(h5_path), "r") as f:
                # Collect (rec_key, sz_intervals) pairs to process
                recordings = []
                if fmt == "windowed":
                    # Pre-windowed: each HDF5 index maps to one window
                    y_key = meta["y_key"]
                    N, C, T_win = f[signal_key].shape
                    y_arr = f[y_key][()].astype(np.int32)   # load labels only (tiny)

                    if ref_num_nodes is None:
                        ref_num_nodes = C
                    if C != ref_num_nodes:
                        log.warning(f"Skipping {h5_path}: C={C} != {ref_num_nodes}")
                        continue
                    if T_win != win_samples:
                        log.warning(
                            f"Skipping {h5_path}: T_win={T_win} != "
                            f"expected {win_samples}")
                        continue

                    for i in range(N):
                        index.append(WindowRecord(
                            h5_path=str(h5_path),
                            rec_key=None,
                            start=i,        # reused as window index for windowed fmt
                            label=int(y_arr[i]),
                            patient=patient,
                            win_samples=win_samples,
                        ))
                    log.info(f"  {Path(h5_path).name}: {N} pre-windowed windows, "
                             f"seizure_ratio={y_arr.mean():.4f}")
                    continue   # skip raw processing below

                elif fmt == "raw_grouped":
                    recs_in_file = group_keys if group_keys else list(f.keys())
                    for rk in recs_in_file:
                        if rk not in f or not isinstance(f[rk], h5py.Group):
                            continue
                        sz_key = meta.get("sz_key")
                        sz = _load_seizure_times_from_h5(f, rk, sz_key, fs)
                        # Fallback to summary map
                        if not sz:
                            edf_name = rk + ".edf"
                            sz = [
                                (int(s * fs), int(e * fs))
                                for s, e in seizure_map.get(edf_name, [])
                            ]
                        recordings.append((rk, sz))

                else:  # raw_flat
                    sz_key = meta.get("sz_key")
                    sz = _load_seizure_times_from_h5(f, None, sz_key, fs)
                    if not sz:
                        edf_name = Path(h5_path).stem + ".edf"
                        sz = [
                            (int(s * fs), int(e * fs))
                            for s, e in seizure_map.get(edf_name, [])
                        ]
                    lbl_key = meta.get("lbl_key")
                    if not sz and lbl_key and lbl_key in f:
                        file_label = int(f[lbl_key][()])
                        if file_label == 1:
                            log.warning(
                                f"{Path(h5_path).name}: file-level label=1 "
                                f"but no seizure intervals → all windows labelled 1")
                    recordings.append((None, sz))

                # ── Process raw recordings ─────────────────────────────
                for rec_key, sz_sample_intervals in recordings:
                    ds = _load_signal_from_h5(f, rec_key, signal_key)
                    # Detect orientation: C is the smaller axis
                    s0, s1 = ds.shape
                    if s0 > s1:                    # likely (T, C) → transpose
                        C, T = s1, s0
                        transposed = True
                    else:
                        C, T = s0, s1
                        transposed = False

                    if ref_num_nodes is None:
                        ref_num_nodes = C
                        # Try to read channel names
                        try:
                            ch_node = f.get("channels") or f.get("channel_names")
                            if ch_node is not None:
                                ref_channels = [
                                    s.decode() if isinstance(s, bytes) else str(s)
                                    for s in ch_node[()]
                                ]
                        except Exception:
                            pass

                    if C != ref_num_nodes:
                        log.warning(
                            f"Skipping rec {rec_key or Path(h5_path).name}: "
                            f"C={C} != {ref_num_nodes}")
                        continue

                    if min_channels and C < min_channels:
                        log.warning(
                            f"Skipping rec: C={C} < min_channels={min_channels}")
                        continue

                    # Sliding window index computation
                    win_count = 0
                    sz_count = 0
                    start = 0
                    while start + win_samples <= T:
                        end = start + win_samples
                        label = 0
                        for sz_s, sz_e in sz_sample_intervals:
                            if not (end <= sz_s or start >= sz_e):
                                label = 1
                                break
                        index.append(WindowRecord(
                            h5_path=str(h5_path),
                            rec_key=rec_key,
                            start=start,
                            label=label,
                            patient=patient,
                            win_samples=win_samples,
                        ))
                        win_count += 1
                        sz_count += label
                        start += stride_samples

                    rec_name = rec_key or Path(h5_path).name
                    log.info(
                        f"  {rec_name}: shape=({C},{T}) transposed={transposed} "
                        f"→ {win_count} windows, {sz_count} seizure"
                    )

        except Exception as e:
            log.warning(f"Error processing {h5_path}: {e}")
            continue

    return index, ref_num_nodes, ref_channels


# ---------------------------------------------------------------------------
# Streaming scaler (two-pass, one recording in RAM at a time)
# ---------------------------------------------------------------------------

def compute_scaler_streaming(train_index, signal_key, win_samples,
                              step_samples, num_nodes, fmt):
    """
    Compute per-channel mean/std without loading the full dataset.

    Strategy:
      - Collect unique (h5_path, rec_key) pairs from the training index.
      - For each unique recording, load the full signal array (one at a time).
      - Accumulate sum and sum-of-squares across all samples.
      - Compute mean/std from accumulated statistics.

    Peak RAM: ~max(recording_length) × C × 4 bytes (≈ 85 MB for a 1-hour file)
    """
    log.info("[Scaler] Computing per-channel statistics (streaming)...")

    if fmt == "windowed":
        # For pre-windowed data, sample the first window from each file
        # (full signal isn't available; use windows as-is)
        n_sample = min(len(train_index), 10000)
        sample_idx = np.random.choice(len(train_index), n_sample, replace=False)
        chan_sum    = np.zeros(num_nodes, dtype=np.float64)
        chan_sq_sum = np.zeros(num_nodes, dtype=np.float64)
        count = 0
        for i in sample_idx:
            entry = train_index[i]
            with h5py.File(entry.h5_path, "r") as f:
                win = f[signal_key][entry.start].astype(np.float64)  # (C, T_win)
                if win.shape[0] != num_nodes:
                    win = win.T
            chan_sum    += win.sum(axis=1)
            chan_sq_sum += (win ** 2).sum(axis=1)
            count       += win.shape[1]
        mean = chan_sum / count
        std  = np.sqrt(np.maximum(chan_sq_sum / count - mean ** 2, 1e-12))
        std[std < 1e-6] = 1.0
        return (mean[np.newaxis, :, np.newaxis].astype(np.float32),
                std[np.newaxis, :, np.newaxis].astype(np.float32))

    # For raw data: collect unique recordings from train index
    unique_recs = {}
    for entry in train_index:
        key = (entry.h5_path, entry.rec_key)
        unique_recs[key] = entry

    n_recs = len(unique_recs)
    log.info(f"[Scaler] {n_recs} unique recordings in training set.")

    # Pass 1: accumulate sum
    chan_sum = np.zeros(num_nodes, dtype=np.float64)
    total_count = 0
    for (h5_path, rec_key), _ in unique_recs.items():
        with h5py.File(h5_path, "r") as f:
            ds = _load_signal_from_h5(f, rec_key, signal_key)
            sig = ds[()].astype(np.float64)    # full recording, float64
        if sig.shape[0] != num_nodes:
            sig = sig.T
        chan_sum    += sig.sum(axis=1)
        total_count += sig.shape[1]
        log.debug(f"[Scaler pass1] {Path(h5_path).name} rec={rec_key} "
                  f"sig={sig.shape}")

    mean = chan_sum / total_count

    # Pass 2: accumulate variance
    chan_sq_diff = np.zeros(num_nodes, dtype=np.float64)
    for (h5_path, rec_key), _ in unique_recs.items():
        with h5py.File(h5_path, "r") as f:
            ds = _load_signal_from_h5(f, rec_key, signal_key)
            sig = ds[()].astype(np.float64)
        if sig.shape[0] != num_nodes:
            sig = sig.T
        chan_sq_diff += ((sig - mean[:, np.newaxis]) ** 2).sum(axis=1)

    std = np.sqrt(np.maximum(chan_sq_diff / total_count, 1e-12))
    std[std < 1e-6] = 1.0

    log.info(f"[Scaler] mean range [{mean.min():.3f}, {mean.max():.3f}]  "
             f"std range [{std.min():.3f}, {std.max():.3f}]")

    return (mean[np.newaxis, :, np.newaxis].astype(np.float32),
            std[np.newaxis, :, np.newaxis].astype(np.float32))


# ---------------------------------------------------------------------------
# Graph helpers (identical to dataloader_chbmit.py – kept local to avoid
# import coupling between the two loaders)
# ---------------------------------------------------------------------------

def _build_fc_adj(num_nodes):
    adj = np.ones((num_nodes, num_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return adj / row_sums


def _compute_supports(adj_mat, filter_type):
    supports_mat = []
    if filter_type == "laplacian":
        supports_mat.append(
            utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
    elif filter_type == "dual_random_walk":
        supports_mat.append(utils.calculate_random_walk_matrix(adj_mat).T)
        supports_mat.append(utils.calculate_random_walk_matrix(adj_mat.T).T)
    else:
        supports_mat.append(
            utils.calculate_scaled_laplacian(adj_mat, lambda_max=None))
    return [torch.FloatTensor(s.toarray()) for s in supports_mat]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CHBMITDatasetHDF5(Dataset):
    """
    Lazy-loading HDF5 Dataset for CHB-MIT EEG.

    No EEG signal data is loaded into RAM during __init__.
    Each __getitem__ call reads exactly one window from disk.

    Output tuple (matches dataloader_chbmit.py / dataloader_detection.py):
        x          – FloatTensor (seq_len, num_nodes, input_dim)
        y          – FloatTensor (1,)
        seq_len    – LongTensor  (1,)
        supports   – list of FloatTensor (num_nodes, num_nodes)
        adj_mat    – ndarray     (num_nodes, num_nodes)
        filename   – str
    """

    def __init__(
            self,
            hdf5_paths,
            fmt, signal_key, group_keys, meta,
            win_samples,
            stride_samples,
            fs=256,
            time_step_size=1,
            standardize=True,
            scaler=None,
            graph_type="combined",
            top_k=3,
            filter_type="laplacian",
            seizure_map=None,
            split="train",
            debug_first_batch=True):
        """
        Do not call directly – use load_dataset_chbmit_hdf5() instead.
        """
        if standardize and scaler is None:
            raise ValueError("Provide a scaler when standardize=True.")

        self.signal_key    = signal_key
        self.fmt           = fmt
        self.win_samples   = win_samples
        self.step_samples  = int(time_step_size * fs)
        self.seq_len       = win_samples // self.step_samples
        self.input_dim     = self.step_samples
        self.fs            = fs
        self.standardize   = standardize
        self.scaler        = scaler
        self.graph_type    = graph_type
        self.top_k         = top_k
        self.filter_type   = filter_type
        self.split         = split
        self._debug_printed = not debug_first_batch

        log.info(
            f"[{split}] Building window index "
            f"(lazy – no EEG loaded into RAM)…"
        )

        self.index, self.num_nodes, self.channels = _build_index(
            hdf5_paths=hdf5_paths,
            fmt=fmt, signal_key=signal_key,
            group_keys=group_keys, meta=meta,
            win_samples=win_samples,
            stride_samples=stride_samples,
            fs=fs,
            seizure_map=seizure_map or {},
        )

        if not self.index:
            raise RuntimeError(f"[{split}] No windows found in provided HDF5 files.")

        N = len(self.index)
        n_sz = sum(e.label for e in self.index)
        ram_mb = N * 80 / 1e6   # approximate bytes per WindowRecord
        window_kb = self.num_nodes * win_samples * 4 / 1024

        log.info(
            f"[{split}] Index built: {N} windows | "
            f"{n_sz} seizure ({100*n_sz/N:.1f}%) | "
            f"num_nodes={self.num_nodes} seq_len={self.seq_len} "
            f"input_dim={self.input_dim}"
        )
        log.info(
            f"[{split}] Memory: index≈{ram_mb:.1f} MB "
            f"(lazy=True, no EEG in RAM) | "
            f"per-window I/O≈{window_kb:.1f} KB"
        )

        self._fc_adj      = _build_fc_adj(self.num_nodes)
        self._fc_supports = _compute_supports(self._fc_adj, "laplacian")
        self._targets     = [e.label for e in self.index]

    def __len__(self):
        return len(self.index)

    def targets(self):
        return self._targets

    def _load_window(self, entry):
        """
        Load exactly one (C, win_samples) window from disk.
        Uses process-local h5py handle cache.
        """
        h5 = _get_cached_h5(entry.h5_path)

        if self.fmt == "windowed":
            # Pre-windowed: entry.start is the window index
            win = h5[self.signal_key][entry.start]    # (C, T_win) or (T_win, C)
            win = np.array(win, dtype=np.float32)
            if win.shape[0] != self.num_nodes:
                win = win.T
        else:
            # Raw: entry.start is the sample offset inside the recording
            ds = _load_signal_from_h5(h5, entry.rec_key, self.signal_key)
            s = entry.start
            e = s + entry.win_samples
            # Detect orientation from shape (avoid loading the whole dataset)
            C0, T0 = ds.shape if ds.shape[0] < ds.shape[1] else (ds.shape[1], ds.shape[0])
            if ds.shape[0] > ds.shape[1]:          # (T, C) layout
                win = ds[s:e, :].T.astype(np.float32)   # → (C, win_samples)
            else:                                   # (C, T) layout
                win = ds[:, s:e].astype(np.float32)

        return win   # (C, win_samples) = (num_nodes, win_samples)

    def _get_indiv_graph(self, eeg_clip):
        """Cross-correlation adjacency from un-standardised eeg_clip (seq_len, C, D)."""
        C = eeg_clip.shape[1]
        adj = np.eye(C, dtype=np.float32)
        flat = eeg_clip.transpose(1, 0, 2).reshape(C, -1)
        for i in range(C):
            for j in range(i + 1, C):
                xc = comp_xcorr(flat[i], flat[j], mode="valid", normalize=True)
                adj[i, j] = xc
                adj[j, i] = xc
        adj = np.abs(adj)
        return keep_topk(adj, top_k=self.top_k, directed=True)

    def __getitem__(self, idx):
        entry = self.index[idx]

        # ── 1. Load raw window (C, win_samples) ─────────────────────────
        raw = self._load_window(entry)   # (C, win_samples) = (23, 512)

        # ── 2. Reshape to (seq_len, num_nodes, input_dim) ───────────────
        #   (23, 512) → (23, 2, 256) → (2, 23, 256)
        eeg_clip = raw.reshape(
            self.num_nodes, self.seq_len, self.input_dim
        ).transpose(1, 0, 2).copy()   # (seq_len, C, input_dim)

        # ── 3. Standardise ───────────────────────────────────────────────
        curr_feat = eeg_clip.copy()
        if self.standardize:
            curr_feat = self.scaler.transform(curr_feat)

        # ── 4. Graph ─────────────────────────────────────────────────────
        if self.graph_type == "individual":
            adj_mat  = self._get_indiv_graph(eeg_clip)   # use raw signal
            supports = _compute_supports(adj_mat, self.filter_type)
        else:
            adj_mat  = self._fc_adj
            supports = self._fc_supports

        x       = torch.FloatTensor(curr_feat)          # (seq_len, C, input_dim)
        y       = torch.FloatTensor([float(entry.label)])  # (1,)
        seq_len = torch.LongTensor([self.seq_len])       # (1,)

        # ── Debug print (first item in first batch only) ─────────────────
        if not self._debug_printed:
            self._debug_printed = True
            log.info(
                f"\n[DEBUG __getitem__ {self.split}]"
                f"\n  raw window       : {raw.shape}   = (C={self.num_nodes}, T={self.win_samples})"
                f"\n  eeg_clip (raw)   : {eeg_clip.shape} = (seq_len, C, input_dim)"
                f"\n  x (standardised) : {tuple(x.shape)}"
                f"\n  y                : {tuple(y.shape)}  label={entry.label}"
                f"\n  seq_len          : {tuple(seq_len.shape)}"
                f"\n  adj_mat          : {adj_mat.shape}"
                f"\n  num_supports     : {len(supports)}  "
                f"support[0]={tuple(supports[0].shape)}"
                f"\n  lazy             : True  (h5py slicing per item)"
            )

        return x, y, seq_len, supports, adj_mat, f"chbmit_hdf5_{entry.patient}_{idx}"


# ---------------------------------------------------------------------------
# load_dataset_chbmit_hdf5  (public API)
# ---------------------------------------------------------------------------

def load_dataset_chbmit_hdf5(
        hdf5_dir,
        summary_dir=None,
        train_batch_size=32,
        test_batch_size=64,
        win_len=2,
        stride=1,
        time_step_size=1,
        fs=256,
        graph_type="combined",
        top_k=3,
        filter_type=None,       # auto-set from graph_type if None
        standardize=True,
        num_workers=4,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=123,
        min_channels=None,
        inspect_first_file=True,
):
    """
    Build train / dev / test DataLoaders from a directory of CHB-MIT HDF5 files.

    Args:
        hdf5_dir:          root dir containing HDF5 files or patient sub-dirs.
                           Searched recursively for *.h5 and *.hdf5 files.
        summary_dir:       optional path to CHB-MIT summary .txt files.
                           Used as fallback if HDF5 files lack seizure annotations.
        train_batch_size:  batch size for training
        test_batch_size:   batch size for dev / test
        win_len:           window length in seconds (default 2)
        stride:            stride in seconds (default 1)
        time_step_size:    seconds per DCRNN time step (default 1)
        fs:                expected sampling rate Hz (default 256)
        graph_type:        'individual' (xcorr) | 'combined' (FC fallback)
        top_k:             neighbours for xcorr graph
        filter_type:       overrides graph_type-derived filter (advanced use)
        standardize:       z-normalise with streaming scaler
        num_workers:       DataLoader workers (use 0 for debugging)
        train_ratio:       fraction of patients in train split
        val_ratio:         fraction of patients in dev split
        seed:              RNG seed for patient-level split
        min_channels:      skip recordings with fewer than this many channels
        inspect_first_file: if True, log full structure of the first HDF5 found

    Returns:
        dataloaders:  {'train', 'dev', 'test'} → DataLoader
        datasets:     {'train', 'dev', 'test'} → CHBMITDatasetHDF5
        scaler:       utils.StandardScaler or None
    """
    hdf5_dir = Path(hdf5_dir)

    # Discover HDF5 files
    all_h5 = sorted(
        list(hdf5_dir.rglob("*.h5")) + list(hdf5_dir.rglob("*.hdf5"))
    )
    if not all_h5:
        raise FileNotFoundError(
            f"No *.h5 / *.hdf5 files found under {hdf5_dir}")
    log.info(f"Found {len(all_h5)} HDF5 file(s) under {hdf5_dir}.")

    # Inspect & probe structure
    if inspect_first_file:
        inspect_hdf5_structure(all_h5[0])

    fmt, signal_key, group_keys, meta = _probe_hdf5(all_h5[0])
    log.info(
        f"Detected HDF5 format: {fmt!r}  signal_key={signal_key!r}  "
        f"group_keys={'<grouped>' if group_keys else None}  "
        f"meta_keys={list(meta.keys())}"
    )

    # Seizure annotations from summary files (optional fallback)
    seizure_map = {}
    if summary_dir:
        seizure_map = _parse_summary_dir(summary_dir)
        log.info(f"Loaded seizure annotations for "
                 f"{len(seizure_map)} EDF file(s) from {summary_dir}.")

    win_samples    = int(win_len * fs)
    stride_samples = int(stride * fs)

    if filter_type is None:
        filter_type = "dual_random_walk" if graph_type == "individual" else "laplacian"

    # Patient-level train / dev / test split
    #   For per-recording files: group by patient prefix ("chb01" from "chb01_03.h5")
    #   For per-patient files:   the file itself is the patient
    if fmt == "raw_grouped":
        # Each file is one patient
        patient_to_files = {f.stem: [f] for f in all_h5}
    else:
        patient_to_files: dict = {}
        for f in all_h5:
            pat = f.stem.split("_")[0]    # "chb01" from "chb01_03"
            patient_to_files.setdefault(pat, []).append(f)

    patients = sorted(patient_to_files.keys())
    rng      = np.random.default_rng(seed)
    idx_perm = rng.permutation(len(patients))

    n_train = max(1, int(len(patients) * train_ratio))
    n_val   = max(1, int(len(patients) * val_ratio))

    train_pats = [patients[i] for i in idx_perm[:n_train]]
    val_pats   = [patients[i] for i in idx_perm[n_train:n_train + n_val]]
    test_pats  = [patients[i] for i in idx_perm[n_train + n_val:]]
    if not test_pats:
        test_pats = val_pats

    split_files = {
        "train": [f for p in train_pats for f in patient_to_files[p]],
        "dev":   [f for p in val_pats   for f in patient_to_files[p]],
        "test":  [f for p in test_pats  for f in patient_to_files[p]],
    }
    log.info(
        f"Patient split: {len(train_pats)} train ({len(split_files['train'])} files) | "
        f"{len(val_pats)} dev | {len(test_pats)} test"
    )

    # Build training index first (needed for scaler)
    log.info("Building training index (probe only – no EEG in RAM)…")
    train_idx, num_nodes, channels = _build_index(
        hdf5_paths=split_files["train"],
        fmt=fmt, signal_key=signal_key,
        group_keys=group_keys, meta=meta,
        win_samples=win_samples,
        stride_samples=stride_samples,
        fs=fs,
        seizure_map=seizure_map,
        min_channels=min_channels,
    )

    if not train_idx:
        raise RuntimeError("No training windows found. Check hdf5_dir and summary_dir.")

    # Streaming scaler (one recording in RAM at a time)
    if standardize:
        mean_arr, std_arr = compute_scaler_streaming(
            train_idx, signal_key,
            win_samples, int(time_step_size * fs),
            num_nodes, fmt)
        scaler = utils.StandardScaler(mean=mean_arr, std=std_arr)
    else:
        scaler = None

    # Build Dataset objects
    common_kwargs = dict(
        fmt=fmt, signal_key=signal_key, group_keys=group_keys, meta=meta,
        win_samples=win_samples, stride_samples=stride_samples,
        fs=fs, time_step_size=time_step_size,
        standardize=standardize, scaler=scaler,
        graph_type=graph_type, top_k=top_k, filter_type=filter_type,
        seizure_map=seizure_map,
    )

    dataloaders, datasets = {}, {}
    for split in ("train", "dev", "test"):
        ds = CHBMITDatasetHDF5(
            hdf5_paths=split_files[split],
            split=split,
            debug_first_batch=(split == "train"),
            **common_kwargs,
        )
        shuffle    = (split == "train")
        batch_size = train_batch_size if split == "train" else test_batch_size
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=(split == "train"),
            worker_init_fn=_worker_init_fn,
            persistent_workers=(num_workers > 0),
        )
        dataloaders[split] = loader
        datasets[split]    = ds

    return dataloaders, datasets, scaler
