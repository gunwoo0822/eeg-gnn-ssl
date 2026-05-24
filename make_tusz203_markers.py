"""
make_tusz203_markers.py  –  TUSZ 2.0.3 clip-level marker generator.

Labelling rules (per non-overlapping clip of length --clip_len seconds):
  overlap >= min_pos_overlap  →  positive  (written to *_sz.txt,   label=1)
  overlap == 0.0              →  negative  (written to *_nosz.txt,  label=0)
  0 < overlap < min_pos_overlap → EXCLUDED (ambiguous; not written to any file)

Annotation source: TUSZ 2.0.3 .csv_bi files
  label == "bckg"  → background (ignored)
  label != "bckg"  → seizure    (kept)

Split detection (from EDF file path):
  path contains "/dev/"  → dev
  path contains "/eval/" → test
  otherwise              → train

Output files (one pair per split):
  {out_dir}/trainSet_seq2seq_{clip_len}s_sz.txt
  {out_dir}/trainSet_seq2seq_{clip_len}s_nosz.txt
  {out_dir}/devSet_seq2seq_{clip_len}s_sz.txt
  {out_dir}/devSet_seq2seq_{clip_len}s_nosz.txt
  {out_dir}/testSet_seq2seq_{clip_len}s_sz.txt
  {out_dir}/testSet_seq2seq_{clip_len}s_nosz.txt

Each line format:
  <edf_basename>.edf_<clip_idx>.h5,<label>

Usage:
    python make_tusz203_markers.py \\
        --raw_dir /dataset/edf \\
        --h5_dir  /root/data/TUSZ_resampled \\
        --out_dir data/file_markers_detection \\
        --clip_len 12 \\
        --min_pos_overlap 6.0
"""

import csv
import argparse
from pathlib import Path

import h5py


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate TUSZ 2.0.3 clip-level seizure detection markers.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--raw_dir", required=True,
                   help="Root dir of TUSZ EDF files (must contain train/, dev/, eval/ sub-dirs).")
    p.add_argument("--h5_dir", required=True,
                   help="Dir with per-recording resampled HDF5 files (one *.h5 per EDF).")
    p.add_argument("--out_dir", default="data/file_markers_detection",
                   help="Output dir for marker txt files.")
    p.add_argument("--clip_len", type=int, default=12,
                   help="Non-overlapping clip length in seconds.")
    p.add_argument("--min_pos_overlap", type=float, default=6.0,
                   help="Minimum seizure overlap (seconds) for positive label. "
                        "Clips with 0 < overlap < this value are excluded as ambiguous.")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Annotation parsing
# ---------------------------------------------------------------------------

def read_seizure_intervals(csv_bi: Path):
    """
    Parse a TUSZ 2.0.3 .csv_bi file.

    CSV columns (after comment lines / header row):
        channel, start_sec, stop_sec, label, confidence

    Returns list of (start_sec, stop_sec) for all non-background segments.
    """
    intervals = []
    try:
        with open(csv_bi, newline="") as f:
            # Strip comment lines (start with '#')
            data_lines = [line for line in f if not line.startswith("#")]
        reader = csv.reader(data_lines)
        next(reader, None)      # skip header row
        for row in reader:
            if len(row) < 4:
                continue
            label = row[3].strip().lower()
            if label != "bckg":
                intervals.append((float(row[1]), float(row[2])))
    except Exception:
        pass
    return intervals


# ---------------------------------------------------------------------------
# Overlap calculation & labelling
# ---------------------------------------------------------------------------

def seizure_overlap_secs(clip_start: float, clip_end: float, intervals) -> float:
    """Total overlap in seconds between [clip_start, clip_end) and seizure intervals."""
    total = 0.0
    for s, e in intervals:
        total += max(0.0, min(clip_end, e) - max(clip_start, s))
    return total


def classify_clip(overlap: float, min_pos: float) -> str:
    """
    Returns one of:
      'pos'  – overlap >= min_pos  → label 1
      'neg'  – overlap == 0        → label 0
      'excl' – 0 < overlap < min_pos  → ambiguous, excluded
    """
    if overlap == 0.0:
        return "neg"
    if overlap >= min_pos:
        return "pos"
    return "excl"


# ---------------------------------------------------------------------------
# Split detection
# ---------------------------------------------------------------------------

def get_split(edf_path: Path) -> str:
    parts = set(edf_path.parts)
    if "dev"  in parts:
        return "dev"
    if "eval" in parts:
        return "test"
    return "train"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    raw_dir    = Path(args.raw_dir)
    h5_dir     = Path(args.h5_dir)
    out_dir    = Path(args.out_dir)
    clip_len   = args.clip_len
    min_pos_ov = args.min_pos_overlap

    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  clip_len        = {clip_len} s")
    print(f"  min_pos_overlap = {min_pos_ov} s  "
          f"(= {100 * min_pos_ov / clip_len:.0f}% of clip)")
    print(f"  raw_dir         = {raw_dir}")
    print(f"  h5_dir          = {h5_dir}")
    print(f"  out_dir         = {out_dir}")
    print("=" * 60)

    # Storage: lines to write and counts per split
    lines  = {s: {"sz": [], "nosz": []} for s in ("train", "dev", "test")}
    counts = {s: {"sz": 0, "nosz": 0, "excl": 0} for s in ("train", "dev", "test")}

    edf_files = sorted(raw_dir.rglob("*.edf"))
    print(f"Found {len(edf_files)} EDF file(s) under {raw_dir}\n")

    for edf in edf_files:
        split  = get_split(edf)
        base   = edf.stem                       # e.g. "00004892_s001_t001"
        h5     = h5_dir / f"{base}.h5"
        csv_bi = edf.with_suffix(".csv_bi")

        # Skip if resampled HDF5 or annotation file is missing
        if not h5.exists() or not csv_bi.exists():
            continue

        # Read signal duration from HDF5 (no data loading)
        try:
            with h5py.File(h5, "r") as hf:
                freq      = int(hf["resample_freq"][()])
                n_samples = hf["resampled_signal"].shape[1]
                duration  = n_samples / freq       # seconds
        except Exception:
            continue

        intervals = read_seizure_intervals(csv_bi)
        n_clips   = int(duration // clip_len)

        for idx in range(n_clips):
            clip_start = idx * clip_len
            clip_end   = clip_start + clip_len
            overlap    = seizure_overlap_secs(clip_start, clip_end, intervals)
            kind       = classify_clip(overlap, min_pos_ov)

            entry = f"{base}.edf_{idx}.h5"
            if kind == "pos":
                lines[split]["sz"].append(f"{entry},1\n")
                counts[split]["sz"] += 1
            elif kind == "neg":
                lines[split]["nosz"].append(f"{entry},0\n")
                counts[split]["nosz"] += 1
            else:                               # excl
                counts[split]["excl"] += 1

    # Write output files and print per-split summary
    split_prefix = {"train": "trainSet", "dev": "devSet", "test": "testSet"}
    for split in ("train", "dev", "test"):
        for kind, suffix in (("sz", "sz"), ("nosz", "nosz")):
            fname = out_dir / f"{split_prefix[split]}_seq2seq_{clip_len}s_{suffix}.txt"
            fname.write_text("".join(lines[split][kind]))

        c     = counts[split]
        total = c["sz"] + c["nosz"] + c["excl"]
        print(
            f"[{split:5s}]  "
            f"total={total:7d}  "
            f"positive={c['sz']:6d}  "
            f"negative={c['nosz']:7d}  "
            f"excluded(ambiguous)={c['excl']:6d}"
        )

    print(f"\nDone. Marker files written to: {out_dir}")


if __name__ == "__main__":
    main()
