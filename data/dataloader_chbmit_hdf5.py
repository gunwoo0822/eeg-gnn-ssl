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
from scipy.signal import resample as sp_resample
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


_SEIZURE_DESCRIPTIONS = {"sz", "seizure", "seiz", "ictal"}


def _parse_annotation_node(node, fs):
    """
    Parse an HDF5 node that contains seizure annotations.

    Handles two formats:
      1. Dataset (M, 2): each row is (start_sec_or_sample, end_sec_or_sample)
      2. Group with sub-datasets onset / duration / description:
             onset       – (M,) seconds
             duration    – (M,) seconds
             description – (M,) bytes/str, keep where value in SEIZURE_DESCRIPTIONS

    Returns list of (start_sample, end_sample) or [] if nothing found.
    """
    import h5py

    if isinstance(node, h5py.Dataset):
        arr = node[()]
        if arr.ndim == 0 or arr.size == 0:
            return []
        arr = np.atleast_2d(arr)
        if arr.max() < 10000:
            arr = (arr * fs).astype(int)
        return [(int(r[0]), int(r[1])) for r in arr]

    if isinstance(node, h5py.Group):
        if "onset" not in node or "duration" not in node:
            return []
        onsets    = node["onset"][()]
        durations = node["duration"][()]

        # Filter by description if present
        if "description" in node:
            raw_desc = node["description"][()]
            keep = []
            for i, d in enumerate(raw_desc):
                if isinstance(d, (bytes, np.bytes_)):
                    d = d.decode("utf-8", errors="replace")
                if str(d).strip().lower() in _SEIZURE_DESCRIPTIONS:
                    keep.append(i)
            if not keep:
                return []
            onsets    = onsets[keep]
            durations = durations[keep]

        intervals = []
        for onset, dur in zip(onsets, durations):
            start = int(float(onset) * fs)
            end   = int((float(onset) + float(dur)) * fs)
            if end > start:
                intervals.append((start, end))
        return intervals

    return []


def _load_seizure_times_from_h5(h5_file, rec_key, sz_key, fs):
    """
    Try to read seizure intervals from an open h5py.File.

    Checks sz_key first; if that is None or fails, falls back to an
    'annotations' group at the recording level (or top level).

    Returns list of (start_sample, end_sample) or [] if none found.
    """
    import h5py

    def _try_key(parent, key):
        try:
            node = parent[key]
            return _parse_annotation_node(node, fs)
        except Exception:
            return None

    # Determine the parent group (recording level or top level)
    try:
        parent = h5_file[rec_key] if rec_key else h5_file
    except Exception:
        return []

    # 1. Try the explicitly detected sz_key
    if sz_key is not None:
        result = _try_key(parent, sz_key)
        if result is not None:
            return result

    # 2. Fallback: look for an 'annotations' group / dataset
    for fallback in ("annotations", "seizure_times", "seizures"):
        if fallback in parent:
            result = _try_key(parent, fallback)
            if result is not None:
                return result

    # 3. Last resort: check at the top-level HDF5 root if rec_key was set
    if rec_key:
        for fallback in ("annotations", "seizure_times", "seizures"):
            if fallback in h5_file:
                result = _try_key(h5_file, fallback)
                if result is not None:
                    return result

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
                              step_samples, num_nodes, fmt,
                              orig_fs=256, target_fs=200, fft_features=100):
    """
    Compute per-channel mean/std of log-amplitude FFT features (paper §A).

    Strategy (raw formats):
      - Collect unique (h5_path, rec_key) recordings from the training index.
      - Load each full recording once, resample orig_fs→target_fs, compute FFT
        features, accumulate two-pass statistics.
      - Peak RAM ≈ one recording (C × T × 4 bytes ≈ 85 MB for a 1-hour file).

    Scaler output shape: (1, C, 1) — per-channel scalar, broadcast over
    (seq_len, C, fft_features) via StandardScaler.transform().
    """
    log.info(
        f"[Scaler] Computing per-channel FFT-feature statistics (streaming) "
        f"orig_fs={orig_fs}→target_fs={target_fs}, fft_features={fft_features}…"
    )

    def _recording_to_fft(sig_raw):
        """(C, T_orig) → (total_steps, C, M) FFT features."""
        sig = _resample_signal(sig_raw.astype(np.float32), orig_fs, target_fs)
        C, T_res = sig.shape
        n_steps = T_res // target_fs
        if n_steps == 0:
            return None
        sig = sig[:, :n_steps * target_fs]                      # trim tail
        seg = sig.reshape(C, n_steps, target_fs).transpose(1, 0, 2)  # (n_steps,C,200)
        return _apply_fft_features(seg, fft_features)            # (n_steps,C,M)

    if fmt == "windowed":
        # Pre-windowed: sample windows, resample, FFT
        n_sample = min(len(train_index), 10000)
        sample_idx = np.random.choice(len(train_index), n_sample, replace=False)
        chan_sum    = np.zeros(num_nodes, dtype=np.float64)
        chan_sq_sum = np.zeros(num_nodes, dtype=np.float64)
        count = 0
        for i in sample_idx:
            entry = train_index[i]
            with h5py.File(entry.h5_path, "r") as f:
                raw = f[signal_key][entry.start].astype(np.float32)  # (C, T_win)
                if raw.shape[0] != num_nodes:
                    raw = raw.T
            feats = _recording_to_fft(raw)   # (n_steps, C, M)
            if feats is None:
                continue
            chan_sum    += feats.sum(axis=(0, 2))          # (C,)
            chan_sq_sum += (feats ** 2).sum(axis=(0, 2))
            count       += feats.shape[0] * feats.shape[2]
        mean = chan_sum / max(count, 1)
        std  = np.sqrt(np.maximum(chan_sq_sum / max(count, 1) - mean ** 2, 1e-12))
        std[std < 1e-6] = 1.0
        return (mean[np.newaxis, :, np.newaxis].astype(np.float32),
                std[np.newaxis, :, np.newaxis].astype(np.float32))

    # Raw formats: collect unique recordings
    unique_recs = {}
    for entry in train_index:
        key = (entry.h5_path, entry.rec_key)
        unique_recs[key] = entry

    n_recs = len(unique_recs)
    log.info(f"[Scaler] {n_recs} unique recordings in training set.")

    # Pass 1: accumulate sum of FFT features
    chan_sum  = np.zeros(num_nodes, dtype=np.float64)
    total_cnt = 0
    for (h5_path, rec_key), _ in unique_recs.items():
        with h5py.File(h5_path, "r") as f:
            ds  = _load_signal_from_h5(f, rec_key, signal_key)
            raw = ds[()].astype(np.float32)
        if raw.shape[0] != num_nodes:
            raw = raw.T
        feats = _recording_to_fft(raw)           # (n_steps, C, M) or None
        if feats is None:
            continue
        chan_sum  += feats.sum(axis=(0, 2))      # sum over (steps, freq_bins)
        total_cnt += feats.shape[0] * feats.shape[2]
        log.debug(f"[Scaler pass1] {Path(h5_path).name} feats={feats.shape}")

    mean = chan_sum / max(total_cnt, 1)

    # Pass 2: accumulate squared deviations
    chan_sq_diff = np.zeros(num_nodes, dtype=np.float64)
    for (h5_path, rec_key), _ in unique_recs.items():
        with h5py.File(h5_path, "r") as f:
            ds  = _load_signal_from_h5(f, rec_key, signal_key)
            raw = ds[()].astype(np.float32)
        if raw.shape[0] != num_nodes:
            raw = raw.T
        feats = _recording_to_fft(raw)
        if feats is None:
            continue
        diff = feats - mean[np.newaxis, :, np.newaxis]    # (n_steps,C,M)
        chan_sq_diff += (diff ** 2).sum(axis=(0, 2))

    std = np.sqrt(np.maximum(chan_sq_diff / max(total_cnt, 1), 1e-12))
    std[std < 1e-6] = 1.0

    log.info(f"[Scaler] mean range [{mean.min():.3f}, {mean.max():.3f}]  "
             f"std range [{std.min():.3f}, {std.max():.3f}]")

    return (mean[np.newaxis, :, np.newaxis].astype(np.float32),
            std[np.newaxis, :, np.newaxis].astype(np.float32))


# ---------------------------------------------------------------------------
# Signal processing helpers (paper §Appendix A)
# ---------------------------------------------------------------------------

def _resample_signal(signal, orig_fs, target_fs):
    """
    Resample signal (C, T) or (C,) from orig_fs → target_fs along last axis.
    No-op when orig_fs == target_fs.
    """
    if orig_fs == target_fs:
        return signal.astype(np.float32)
    n_out = int(round(signal.shape[-1] * target_fs / orig_fs))
    return sp_resample(signal, n_out, axis=-1).astype(np.float32)


def _apply_fft_features(eeg_clip, fft_features=100):
    """
    Convert a raw EEG clip to log-amplitude FFT features (paper §Appendix A).

    Pipeline (per time step, per channel):
      1. FFT the 1-second segment
      2. Take non-negative frequency components (skip DC at index 0)
      3. Log amplitude: log(|FFT|[1 : fft_features+1] + ε)

    Args:
        eeg_clip:    (T, C, step_samples) float32  –  raw signal after resample
        fft_features: M = number of frequency bins to keep (default 100 for 200 Hz)

    Returns:
        fft_clip:    (T, C, M) float32
    """
    T, C, S = eeg_clip.shape
    fft_clip = np.zeros((T, C, fft_features), dtype=np.float32)
    for t in range(T):
        for c in range(C):
            spec = np.fft.rfft(eeg_clip[t, c])          # (S//2 + 1,) complex
            log_amp = np.log(np.abs(spec[1: fft_features + 1]) + 1e-8)
            fft_clip[t, c] = log_amp
    return fft_clip


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
            orig_fs=256,
            target_fs=200,
            fft_features=100,
            time_step_size=1,
            standardize=True,
            scaler=None,
            graph_type="combined",
            top_k=3,
            filter_type="laplacian",
            seizure_map=None,
            split="train",
            undersample=False,
            undersample_seed=42,
            debug_first_batch=True):
        """
        Do not call directly – use load_dataset_chbmit_hdf5() instead.

        Args:
            orig_fs:      Native sampling rate of the HDF5 files (e.g. 256 Hz).
            target_fs:    Target sampling rate after resampling (paper: 200 Hz).
            fft_features: M – number of log-amplitude FFT bins kept per segment.
                          (target_fs // 2 = 100 for 200 Hz, 1-second window.)
            undersample:  If True (train split only), downsample negative clips
                          so that the training set is ~50% positive (paper §5).
        """
        if standardize and scaler is None:
            raise ValueError("Provide a scaler when standardize=True.")

        self.signal_key    = signal_key
        self.fmt           = fmt
        self.win_samples   = win_samples         # in orig_fs samples (for HDF5 slice)
        self.orig_fs       = orig_fs
        self.target_fs     = target_fs
        self.fft_features  = fft_features
        # After resampling: 1 second = target_fs samples per DCRNN time step
        self.step_samples  = target_fs            # = time_step_size(1s) × target_fs
        # seq_len = win_len seconds (each 1-second segment is one DCRNN step)
        self.seq_len       = int(round(win_samples * target_fs / orig_fs)) // target_fs
        self.input_dim     = fft_features         # M log-amplitude FFT bins
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
            fs=orig_fs,                           # index in native-fs samples
            seizure_map=seizure_map or {},
        )

        if not self.index:
            raise RuntimeError(f"[{split}] No windows found in provided HDF5 files.")

        # ── Undersampling (train only) ──────────────────────────────────────
        N_orig = len(self.index)
        n_sz_orig = sum(e.label for e in self.index)
        if undersample and n_sz_orig > 0:
            pos_idx = [i for i, e in enumerate(self.index) if e.label == 1]
            neg_idx = [i for i, e in enumerate(self.index) if e.label == 0]
            n_pos, n_neg = len(pos_idx), len(neg_idx)
            rng_us = np.random.default_rng(undersample_seed)
            if n_neg > n_pos:
                neg_idx = rng_us.choice(neg_idx, n_pos, replace=False).tolist()
            kept = sorted(pos_idx + neg_idx)
            self.index = [self.index[i] for i in kept]
            log.info(
                f"[{split}] Undersampling: {N_orig} → {len(self.index)} windows | "
                f"orig pos={n_pos} neg={n_neg} | "
                f"after pos={len(pos_idx)} neg={len(neg_idx)}"
            )

        N = len(self.index)
        n_sz = sum(e.label for e in self.index)
        ram_mb     = N * 80 / 1e6
        window_kb  = self.num_nodes * win_samples * 4 / 1024

        log.info(
            f"[{split}] Index built: {N} windows | "
            f"{n_sz} seizure ({100*n_sz/N:.1f}%) | "
            f"num_nodes={self.num_nodes} seq_len={self.seq_len} "
            f"input_dim={self.input_dim} (FFT features)"
        )
        log.info(
            f"[{split}] Memory: index≈{ram_mb:.1f} MB "
            f"(lazy=True, no EEG in RAM) | "
            f"per-window I/O≈{window_kb:.1f} KB"
        )

        self._fc_adj      = _build_fc_adj(self.num_nodes)
        self._fc_supports = _compute_supports(self._fc_adj, "laplacian")
        self._targets     = [e.label for e in self.index]
        self._graph_debug_printed = False   # print graph shapes on first call

    def __len__(self):
        return len(self.index)

    def targets(self):
        return self._targets

    def _load_window(self, entry):
        """
        Load one raw window (C, win_samples_orig) then resample to target_fs.

        Returns: (C, win_samples_resampled) at target_fs.
        """
        h5 = _get_cached_h5(entry.h5_path)

        if self.fmt == "windowed":
            win = np.array(h5[self.signal_key][entry.start], dtype=np.float32)
            if win.shape[0] != self.num_nodes:
                win = win.T
        else:
            ds = _load_signal_from_h5(h5, entry.rec_key, self.signal_key)
            s, e = entry.start, entry.start + entry.win_samples
            if ds.shape[0] > ds.shape[1]:          # (T, C) layout
                win = ds[s:e, :].T.astype(np.float32)
            else:                                   # (C, T) layout
                win = ds[:, s:e].astype(np.float32)

        # Resample orig_fs → target_fs (paper: 256 → 200 Hz)
        win = _resample_signal(win, self.orig_fs, self.target_fs)
        return win   # (C, seq_len * target_fs)

    def _get_indiv_graph(self, eeg_clip):
        """
        Compute per-clip correlation adjacency matrix (paper §2, Corr-DCRNN).

        Method (vectorised):
          1. Concatenate all T time steps along the temporal axis:
               (T, C, step_samples) → (C, T×step_samples)
          2. Compute normalised zero-lag cross-correlation for every channel pair
             via the gram matrix:
               adj[i,j] = dot(xi, xj) / (||xi|| × ||xj||)
             This is mathematically identical to scipy.signal.correlate(mode='valid')
             with normalization, but returns guaranteed scalar (C,C) float32 values
             and is O(C²) faster than the nested loop.
          3. Take absolute value, restore self-edges to 1.
          4. Apply top-k sparsification (paper: τ=3 per node).

        Args:
            eeg_clip: (seq_len, C, step_samples) – raw time-domain signal at
                      target_fs (200 Hz), BEFORE FFT.

        Returns:
            adj_mat: (C, C) float32, directed top-k sparse.
        """
        T, C, S = eeg_clip.shape

        # (T, C, S) → (C, T*S): concatenate all time steps per channel
        flat = eeg_clip.transpose(1, 0, 2).reshape(C, -1).astype(np.float64)

        # ── Debug (first call only) ──────────────────────────────────────
        if not self._graph_debug_printed:
            self._graph_debug_printed = True
            log.info(
                f"\n[Graph debug] eeg_clip input  : {eeg_clip.shape}"
                f"  = (seq_len={T}, C={C}, step_samples={S})"
                f"\n[Graph debug] flat (concat)   : {flat.shape}"
                f"  = (C={C}, T×S={T*S})"
            )

        # ── Vectorised normalised cross-correlation ──────────────────────
        # gram[i,j] = dot(flat[i], flat[j])
        gram = flat @ flat.T                                # (C, C) float64

        # L2 norms: sqrt of diagonal of gram matrix
        norms = np.sqrt(np.maximum(np.diag(gram), 0.0))    # (C,)  ||flat[i]||

        # Outer product of norms = normalisation denominator
        norm_prod = np.outer(norms, norms)                  # (C, C)
        norm_prod[norm_prod < 1e-12] = 1.0                  # guard against zero

        adj = (gram / norm_prod).astype(np.float32)         # (C, C) ∈ [-1, 1]

        # ── Debug: intermediate correlation stats ────────────────────────
        if not getattr(self, '_graph_debug2_printed', False):
            self._graph_debug2_printed = True
            off = adj[~np.eye(C, dtype=bool)]
            log.info(
                f"[Graph debug] corr matrix      : shape={adj.shape}"
                f"  min={adj.min():.3f}  max={adj.max():.3f}"
                f"  off-diag mean={off.mean():.3f}  std={off.std():.3f}"
            )

        # ── Top-k sparsification ─────────────────────────────────────────
        adj = np.abs(adj)
        np.fill_diagonal(adj, 1.0)
        adj = keep_topk(adj, top_k=self.top_k, directed=True)

        # ── Debug: final adjacency ────────────────────────────────────────
        if not getattr(self, '_graph_debug3_printed', False):
            self._graph_debug3_printed = True
            nnz = np.count_nonzero(adj)
            log.info(
                f"[Graph debug] final adj        : shape={adj.shape}"
                f"  nnz={nnz}"
                f"  (self-edges={C} + top-{self.top_k} per node"
                f" ≤ {C + C * self.top_k})"
            )

        return adj

    def __getitem__(self, idx):
        entry = self.index[idx]

        # ── 1. Load & resample: (C, seq_len * target_fs) ────────────────
        raw = self._load_window(entry)          # (C, T_resampled) at target_fs

        # ── 2. Reshape to (seq_len, C, step_samples) ────────────────────
        #   e.g. (23, 2400) → (23, 12, 200) → (12, 23, 200)
        n_steps = self.seq_len
        eeg_clip = raw.reshape(
            self.num_nodes, n_steps, self.step_samples
        ).transpose(1, 0, 2).copy()            # (seq_len, C, target_fs)

        # ── 3. FFT features: log-amplitude of non-negative frequencies ───
        #   (seq_len, C, target_fs) → (seq_len, C, fft_features)
        fft_clip = _apply_fft_features(eeg_clip, self.fft_features)

        # ── 4. Standardise in FFT feature space ──────────────────────────
        curr_feat = fft_clip.copy()
        if self.standardize:
            curr_feat = self.scaler.transform(curr_feat)  # (seq_len, C, M)

        # ── 5. Graph (computed on raw time-domain signal) ────────────────
        if self.graph_type == "individual":
            adj_mat  = self._get_indiv_graph(eeg_clip)   # uses raw (seq_len,C,200)
            supports = _compute_supports(adj_mat, self.filter_type)
        else:
            adj_mat  = self._fc_adj
            supports = self._fc_supports

        x       = torch.FloatTensor(curr_feat)              # (seq_len, C, M)
        y       = torch.FloatTensor([float(entry.label)])   # (1,)
        seq_len = torch.LongTensor([self.seq_len])          # (1,)

        # ── Debug print (first item only) ────────────────────────────────
        if not self._debug_printed:
            self._debug_printed = True
            log.info(
                f"\n[DEBUG __getitem__ {self.split}]"
                f"\n  raw window (resampled) : {raw.shape}"
                f"  = (C={self.num_nodes}, T={raw.shape[-1]}) @{self.target_fs}Hz"
                f"\n  eeg_clip (time-domain) : {eeg_clip.shape}"
                f"  = (seq_len={n_steps}, C, step_samples={self.step_samples})"
                f"\n  fft_clip (log-amp)     : {fft_clip.shape}"
                f"  = (seq_len, C, M={self.fft_features})"
                f"\n  x (standardised)       : {tuple(x.shape)}"
                f"\n  y                      : {tuple(y.shape)}  label={entry.label}"
                f"\n  supports               : {len(supports)} × {tuple(supports[0].shape)}"
                f"\n  lazy                   : True (h5py slicing per item)"
            )

        return x, y, seq_len, supports, adj_mat, f"chbmit_hdf5_{entry.patient}_{idx}"


# ---------------------------------------------------------------------------
# load_dataset_chbmit_hdf5  (public API)
# ---------------------------------------------------------------------------

def load_dataset_chbmit_hdf5(
        hdf5_dir,
        summary_dir=None,
        train_batch_size=40,
        test_batch_size=64,
        win_len=12,
        stride=None,            # None → non-overlapping (stride = win_len)
        time_step_size=1,
        orig_fs=256,            # native HDF5 sampling rate
        fs=200,                 # target sampling rate after resampling (paper)
        fft_features=100,       # M log-amplitude FFT bins (paper: fs//2)
        graph_type="individual",
        top_k=3,
        filter_type=None,       # auto-set from graph_type if None
        standardize=True,
        num_workers=4,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=123,
        min_channels=None,
        undersample_train=True,  # 50/50 neg undersampling on train (paper §5)
        inspect_first_file=True,
):
    """
    Build train / dev / test DataLoaders from a directory of CHB-MIT HDF5 files.

    Args:
        hdf5_dir:          root dir containing HDF5 files or patient sub-dirs.
                           Searched recursively for *.h5 and *.hdf5 files.
        summary_dir:       optional CHB-MIT summary .txt dir (fallback labels).
        train_batch_size:  batch size for training (paper: 40)
        test_batch_size:   batch size for dev / test
        win_len:           clip length in seconds (paper: 12 or 60)
        stride:            stride in seconds; None → non-overlapping (= win_len)
        time_step_size:    seconds per DCRNN time step (always 1 for paper)
        orig_fs:           native HDF5 sampling rate (CHB-MIT: 256 Hz)
        fs:                target sampling rate after resampling (paper: 200 Hz)
        fft_features:      M – log-amplitude FFT bins per segment (paper: 100)
        graph_type:        'individual' (xcorr) | 'combined' (FC fallback)
        top_k:             neighbours for xcorr graph (paper: τ=3)
        filter_type:       overrides graph_type-derived filter (advanced use)
        standardize:       z-normalise FFT features with streaming scaler
        num_workers:       DataLoader workers (use 0 for debugging)
        train_ratio:       fraction of patients in train split
        val_ratio:         fraction of patients in dev split
        seed:              RNG seed for patient-level split
        min_channels:      skip recordings with fewer than this many channels
        undersample_train: if True, downsample negatives to 50/50 (paper §5)
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

    if stride is None:
        stride = win_len                         # non-overlapping (paper default)

    win_samples    = int(win_len * orig_fs)      # in native-fs samples
    stride_samples = int(stride * orig_fs)

    fft_features = fft_features or (fs // 2)    # default: 100 for 200 Hz

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
        fs=orig_fs,                              # index uses native fs
        seizure_map=seizure_map,
        min_channels=min_channels,
    )

    if not train_idx:
        raise RuntimeError("No training windows found. Check hdf5_dir and summary_dir.")

    n_pos_train = sum(e.label for e in train_idx)
    n_neg_train = len(train_idx) - n_pos_train
    log.info(
        f"[train] Original: {len(train_idx)} windows | "
        f"pos={n_pos_train} ({100*n_pos_train/len(train_idx):.1f}%) | "
        f"neg={n_neg_train}"
    )

    # Streaming scaler in FFT feature space (one recording in RAM at a time)
    if standardize:
        mean_arr, std_arr = compute_scaler_streaming(
            train_idx, signal_key,
            win_samples, int(time_step_size * orig_fs),
            num_nodes, fmt,
            orig_fs=orig_fs, target_fs=fs,
            fft_features=fft_features)
        scaler = utils.StandardScaler(mean=mean_arr, std=std_arr)
    else:
        scaler = None

    # Build Dataset objects
    common_kwargs = dict(
        fmt=fmt, signal_key=signal_key, group_keys=group_keys, meta=meta,
        win_samples=win_samples, stride_samples=stride_samples,
        orig_fs=orig_fs, target_fs=fs, fft_features=fft_features,
        time_step_size=time_step_size,
        standardize=standardize, scaler=scaler,
        graph_type=graph_type, top_k=top_k, filter_type=filter_type,
        seizure_map=seizure_map,
    )

    dataloaders, datasets = {}, {}
    for split in ("train", "dev", "test"):
        ds = CHBMITDatasetHDF5(
            hdf5_paths=split_files[split],
            split=split,
            undersample=(split == "train" and undersample_train),
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
