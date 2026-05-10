"""
Preprocess CHB-MIT raw EDF files into windowed NPZ arrays.

Input layout:
    edf_dir/
        chb01/
            chb01-summary.txt
            chb01_01.edf
            chb01_02.edf
            ...
        chb02/
            ...

Output layout:
    out_dir/
        chb01/
            windows.npz   # X:(N,C,T) float32, y:(N,) int32, channels:(C,) str
        chb02/
            ...

Usage:
    python data/preprocess_chbmit.py \\
        --edf_dir /path/to/physionet.org/files/chbmit/1.0.0 \\
        --out_dir /path/to/chbmit_npz

    # Process specific patients only:
    python data/preprocess_chbmit.py \\
        --edf_dir /path/to/chbmit \\
        --out_dir /path/to/chbmit_npz \\
        --patients chb01 chb02 chb03
"""

import os
import re
import sys
import argparse
import logging
import numpy as np
from pathlib import Path

try:
    import pyedflib
except ImportError:
    raise ImportError("pyedflib required: pip install pyedflib")

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


def parse_summary(summary_path):
    """
    Parse a CHB-MIT *-summary.txt file.

    Handles both single-seizure format:
        Seizure Start Time: 2996 seconds
    and multi-seizure format:
        Seizure 1 Start Time: 1015 seconds

    Returns:
        dict: {edf_filename (str): [(start_sec, end_sec), ...]}
    """
    seizures = {}
    cur_file = None
    starts, ends = [], []

    with open(summary_path, "r", errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.strip()

            m = re.match(r"File Name:\s+(\S+\.edf)", line, re.IGNORECASE)
            if m:
                if cur_file is not None:
                    seizures[cur_file] = list(zip(starts, ends))
                cur_file = m.group(1)
                starts, ends = [], []
                seizures[cur_file] = []
                continue

            # "Number of Seizures in File: N" – informational, not used directly
            if re.match(r"Number of Seizures", line, re.IGNORECASE):
                continue

            m = re.match(
                r"Seizure(?:\s+\d+)?\s+Start\s+Time:\s+(\d+)",
                line, re.IGNORECASE)
            if m:
                starts.append(int(m.group(1)))
                continue

            m = re.match(
                r"Seizure(?:\s+\d+)?\s+End\s+Time:\s+(\d+)",
                line, re.IGNORECASE)
            if m:
                ends.append(int(m.group(1)))
                continue

    # flush last file
    if cur_file is not None:
        seizures[cur_file] = list(zip(starts, ends))

    return seizures


def _read_edf(edf_path, target_labels=None):
    """
    Open an EDF file and read signals.

    Args:
        edf_path: Path to .edf file
        target_labels: list of uppercase channel names to extract (order preserved).
                       If None, read all channels.

    Returns:
        (signals, labels, fs) or None if the file is invalid / channels missing.
            signals: np.float32 (C, T)
            labels:  list[str] – channel names in order
            fs:      int – sampling rate (Hz)
    """
    try:
        f = pyedflib.EdfReader(str(edf_path))
    except Exception as e:
        log.warning(f"Cannot open {edf_path.name}: {e}")
        return None

    raw_labels = [lbl.strip().upper() for lbl in f.getSignalLabels()]
    fs_list = list(f.getSampleFrequencies())
    n_sigs = f.signals_in_file

    if target_labels is not None:
        missing = [l for l in target_labels if l not in raw_labels]
        if missing:
            log.warning(
                f"Skipping {edf_path.name}: missing channels {missing[:5]}"
                f"{' ...' if len(missing) > 5 else ''}")
            f._close()
            return None
        indices = [raw_labels.index(l) for l in target_labels]
    else:
        indices = list(range(n_sigs))

    # Verify uniform sampling rate across selected channels
    expected_fs = int(fs_list[indices[0]])
    for i in indices:
        if int(fs_list[i]) != expected_fs:
            log.warning(f"Skipping {edf_path.name}: inconsistent fs across channels")
            f._close()
            return None

    signals = []
    for i in indices:
        try:
            signals.append(f.readSignal(i).astype(np.float32))
        except Exception as e:
            log.warning(f"Error reading signal {i} in {edf_path.name}: {e}")
            f._close()
            return None

    f._close()
    signals = np.stack(signals, axis=0)  # (C, T)
    labels = [raw_labels[i] for i in indices]
    return signals, labels, expected_fs


def _window_and_label(signals, seizure_intervals_sec, fs,
                      win_samples, stride_samples):
    """
    Sliding-window extraction with binary seizure labelling.

    A window is labelled 1 if it overlaps (even by 1 sample) with any
    seizure interval.

    Args:
        signals:                (C, T) float32
        seizure_intervals_sec:  list of (start_sec, end_sec) from summary
        fs:                     sampling rate (Hz)
        win_samples:            window length in samples
        stride_samples:         stride in samples

    Returns:
        X: (N, C, win_samples) float32  or None if no windows
        y: (N,)                int32
    """
    C, T = signals.shape
    sz_sample_iv = [(int(s * fs), int(e * fs)) for s, e in seizure_intervals_sec]

    X_list, y_list = [], []
    start = 0
    while start + win_samples <= T:
        end = start + win_samples
        window = signals[:, start:end]

        label = 0
        for sz_s, sz_e in sz_sample_iv:
            if not (end <= sz_s or start >= sz_e):
                label = 1
                break

        X_list.append(window)
        y_list.append(label)
        start += stride_samples

    if not X_list:
        return None, None

    X = np.stack(X_list, axis=0)           # (N, C, win_samples)
    y = np.array(y_list, dtype=np.int32)   # (N,)
    return X, y


def process_patient(patient_dir, out_dir, win_len=2, stride=1, target_fs=256):
    """
    Preprocess all EDF files for one patient.

    Canonical channel list is determined from the first valid EDF found.
    All subsequent EDF files must expose exactly those channels; others are
    skipped with a log warning.
    """
    patient = patient_dir.name

    # Locate summary file
    summary_files = list(patient_dir.glob("*-summary.txt"))
    if not summary_files:
        log.warning(f"{patient}: no *-summary.txt found, skipping.")
        return
    seizure_map = parse_summary(summary_files[0])

    edf_files = sorted(patient_dir.glob("*.edf"))
    if not edf_files:
        log.warning(f"{patient}: no EDF files found, skipping.")
        return

    log.info(f"Processing {patient}: {len(edf_files)} EDF file(s) found.")

    # Determine canonical channels from the first readable EDF at target_fs
    canonical_labels = None
    for edf_path in edf_files:
        result = _read_edf(edf_path, target_labels=None)
        if result is None:
            continue
        _, labels, fs = result
        if int(fs) != target_fs:
            log.warning(f"  {edf_path.name}: fs={fs} != {target_fs}, "
                        f"skipping for channel detection.")
            continue
        canonical_labels = labels
        log.info(f"  Canonical channels ({len(canonical_labels)}) "
                 f"from {edf_path.name}: {canonical_labels[:4]} ...")
        break

    if canonical_labels is None:
        log.warning(f"{patient}: no valid EDF at {target_fs} Hz, skipping.")
        return

    win_samples = int(win_len * target_fs)
    stride_samples = int(stride * target_fs)

    all_X, all_y = [], []
    for edf_path in edf_files:
        result = _read_edf(edf_path, target_labels=canonical_labels)
        if result is None:
            continue
        signals, _, fs = result

        if int(fs) != target_fs:
            log.warning(f"  Skipping {edf_path.name}: fs={fs}")
            continue

        sz_intervals = seizure_map.get(edf_path.name, [])
        X, y = _window_and_label(signals, sz_intervals, target_fs,
                                  win_samples, stride_samples)
        if X is None:
            continue

        all_X.append(X)
        all_y.append(y)
        sz_count = int(y.sum())
        log.info(f"  {edf_path.name}: {len(X)} windows, "
                 f"{sz_count} seizure, {len(X) - sz_count} non-seizure")

    if not all_X:
        log.warning(f"{patient}: no windows extracted.")
        return

    X_all = np.concatenate(all_X, axis=0)   # (N, C, T)
    y_all = np.concatenate(all_y, axis=0)   # (N,)

    out_patient_dir = out_dir / patient
    out_patient_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_patient_dir / "windows.npz"

    np.savez_compressed(
        str(out_path),
        X=X_all,
        y=y_all,
        channels=np.array(canonical_labels, dtype=object),
    )

    log.info(
        f"Saved {patient}: X={X_all.shape}, y={y_all.shape}, "
        f"seizure_ratio={y_all.mean():.4f}  →  {out_path}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess CHB-MIT EDF files into windowed NPZ arrays.")
    parser.add_argument("--edf_dir", type=str, required=True,
                        help="Root of CHB-MIT dataset (contains chb01/, chb02/, ...).")
    parser.add_argument("--out_dir", type=str, required=True,
                        help="Output directory for NPZ files.")
    parser.add_argument("--win_len", type=int, default=2,
                        help="Window length in seconds (default: 2).")
    parser.add_argument("--stride", type=int, default=1,
                        help="Stride in seconds (default: 1).")
    parser.add_argument("--fs", type=int, default=256,
                        help="Expected sampling rate in Hz (default: 256).")
    parser.add_argument("--patients", nargs="+", default=None,
                        help="Patient IDs to process, e.g. chb01 chb02. "
                             "Default: all chb* subdirectories.")
    args = parser.parse_args()

    edf_dir = Path(args.edf_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.patients:
        patient_dirs = [edf_dir / p for p in args.patients]
    else:
        patient_dirs = sorted(
            d for d in edf_dir.iterdir()
            if d.is_dir() and d.name.lower().startswith("chb"))

    log.info(f"Found {len(patient_dirs)} patient directories.")
    for patient_dir in patient_dirs:
        if not patient_dir.exists():
            log.warning(f"Not found: {patient_dir}")
            continue
        process_patient(patient_dir, out_dir,
                        win_len=args.win_len,
                        stride=args.stride,
                        target_fs=args.fs)

    log.info("Preprocessing complete.")


if __name__ == "__main__":
    main()
