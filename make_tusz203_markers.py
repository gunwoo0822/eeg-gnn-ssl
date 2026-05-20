import os, csv, random, h5py
from pathlib import Path

RAW_DIR = Path("/dataset/edf")
H5_DIR = Path("/root/data/TUSZ_resampled")
OUT_DIR = Path("data/file_markers_detection")
CLIP_LEN = 12
SEED = 123
random.seed(SEED)

def read_seizure_times(csv_bi):
    times = []
    with open(csv_bi) as f:
        reader = csv.reader(line for line in f if not line.startswith("#"))
        next(reader, None)
        for row in reader:
            if len(row) < 5:
                continue
            label = row[3].strip().lower()
            if label != "bckg":
                times.append((float(row[1]), float(row[2])))
    return times

def has_seizure(start, end, seizure_times):
    for s, e in seizure_times:
        if not (end <= s or start >= e):
            return True
    return False

items = {"train": {"sz": [], "nosz": []},
         "dev": {"sz": [], "nosz": []},
         "test": {"sz": [], "nosz": []}}

edf_files = sorted(RAW_DIR.rglob("*.edf"))

for edf in edf_files:
    split = "dev" if "/dev/" in str(edf) else "test" if "/eval/" in str(edf) else "train"
    base = edf.stem
    h5 = H5_DIR / f"{base}.h5"
    csv_bi = edf.with_suffix(".csv_bi")

    if not h5.exists() or not csv_bi.exists():
        continue

    try:
        with h5py.File(h5, "r") as hf:
            sig = hf["resampled_signal"]
            freq = int(hf["resample_freq"][()])
            duration = sig.shape[1] / freq
    except Exception:
        continue

    seizure_times = read_seizure_times(csv_bi)
    n_clips = int(duration // CLIP_LEN)

    for idx in range(n_clips):
        start = idx * CLIP_LEN
        end = start + CLIP_LEN
        label = 1 if has_seizure(start, end, seizure_times) else 0
        line = f"{base}.edf_{idx}.h5,{label}\n"
        items[split]["sz" if label else "nosz"].append(line)

for split in ["train", "dev", "test"]:
    for kind in ["sz", "nosz"]:
        random.shuffle(items[split][kind])
        out = OUT_DIR / f"{split}Set_seq2seq_12s_{kind}.txt"
        out.write_text("".join(items[split][kind]))
        print(out, len(items[split][kind]))
