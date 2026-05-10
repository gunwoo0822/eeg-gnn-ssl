"""
DataLoader for CHB-MIT preprocessed NPZ windows.

Each item returned by CHBMITDataset.__getitem__:
    x          – FloatTensor  (seq_len, num_nodes, input_dim)
                  seq_len  = win_len // time_step_size   (default 2)
                  num_nodes = number of EEG channels     (default 23)
                  input_dim = time_step_size * fs         (default 256)
    y          – FloatTensor  (1,)  binary label
    seq_len    – LongTensor   (1,)
    supports   – list of FloatTensor (num_nodes, num_nodes)
                  1 tensor  for filter_type='laplacian'
                  2 tensors for filter_type='dual_random_walk'
    adj_mat    – ndarray (num_nodes, num_nodes)
    filename   – str

After DataLoader collation:
    x          → (batch, seq_len, num_nodes, input_dim)
    y          → (batch, 1)
    seq_len    → (batch, 1)
    supports   → list of (batch, num_nodes, num_nodes)
"""

import sys
import logging
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent))

import utils
from data.data_utils import comp_xcorr, keep_topk

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Graph helpers
# ---------------------------------------------------------------------------

def _build_fc_adj(num_nodes):
    """
    Normalised fully-connected adjacency matrix (CHB-MIT fallback graph).
    Used when graph_type='combined' and no pre-computed distance graph exists.
    """
    adj = np.ones((num_nodes, num_nodes), dtype=np.float32)
    np.fill_diagonal(adj, 0.0)
    row_sums = adj.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return adj / row_sums


def _compute_supports(adj_mat, filter_type):
    """
    Compute support tensors from an adjacency matrix.

    filter_type='laplacian'        → 1 scaled Laplacian support
    filter_type='dual_random_walk' → 2 bidirectional random-walk supports

    Returns list of dense FloatTensor (num_nodes, num_nodes).
    These are collated by DataLoader into (batch, num_nodes, num_nodes).
    """
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
# Scaler
# ---------------------------------------------------------------------------

def compute_chbmit_scaler(npz_paths, fs=256, time_step_size=1):
    """
    Compute per-channel mean/std from training windows.

    Returns utils.StandardScaler with mean/std shape (1, num_nodes, 1),
    compatible with transform(eeg_clip) where eeg_clip is (seq_len, C, input_dim).
    """
    step_samples = int(time_step_size * fs)
    all_clips = []
    for npz_path in npz_paths:
        data = np.load(str(npz_path), allow_pickle=True)
        X = data["X"].astype(np.float32)   # (N, C, T)
        N, C, T = X.shape
        seq_len = T // step_samples
        # (N, seq_len, C, step_samples)
        clips = X.reshape(N, C, seq_len, step_samples).transpose(0, 2, 1, 3)
        all_clips.append(clips.reshape(-1, C, step_samples))

    all_clips = np.concatenate(all_clips, axis=0)   # (N_total, C, step_samples)
    mean = all_clips.mean(axis=(0, 2))              # (C,)
    std  = all_clips.std(axis=(0, 2))               # (C,)
    std[std < 1e-6] = 1.0                           # avoid division by zero

    # Shape (1, C, 1) for broadcast over (seq_len, C, input_dim)
    mean = mean[np.newaxis, :, np.newaxis]
    std  = std[np.newaxis, :, np.newaxis]
    return utils.StandardScaler(mean=mean, std=std)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class CHBMITDataset(Dataset):
    """
    PyTorch Dataset for CHB-MIT windowed EEG.

    Loads NPZ files produced by preprocess_chbmit.py.  Reshapes each raw
    window (C, T) into DCRNN time-step format (seq_len, C, input_dim) and
    computes a per-sample or shared graph adjacency matrix.
    """

    def __init__(
            self,
            npz_paths,
            time_step_size=1,
            fs=256,
            standardize=True,
            scaler=None,
            graph_type="individual",
            top_k=3,
            filter_type="dual_random_walk",
            split="train"):
        """
        Args:
            npz_paths:      list of Path / str to windows.npz files
            time_step_size: seconds per DCRNN time step (default 1)
            fs:             sampling rate in Hz (default 256)
            standardize:    z-normalise with scaler if True
            scaler:         utils.StandardScaler (required when standardize=True)
            graph_type:     'individual' = per-sample cross-correlation graph
                            'combined'   = normalised fully-connected fallback
            top_k:          k nearest neighbours for individual graph
            filter_type:    'dual_random_walk' (individual) or 'laplacian' (combined)
            split:          'train' | 'dev' | 'test'  (used for logging)
        """
        if standardize and scaler is None:
            raise ValueError("Provide a scaler when standardize=True.")

        self.time_step_size = time_step_size
        self.fs = fs
        self.step_samples = int(time_step_size * fs)   # samples per DCRNN step
        self.standardize = standardize
        self.scaler = scaler
        self.graph_type = graph_type
        self.top_k = top_k
        self.filter_type = filter_type
        self.split = split

        all_X, all_y = [], []
        num_channels = None
        for npz_path in npz_paths:
            data = np.load(str(npz_path), allow_pickle=True)
            X = data["X"].astype(np.float32)   # (N, C, T)
            y = data["y"].astype(np.int32)     # (N,)

            if num_channels is None:
                num_channels = X.shape[1]
                log.info(f"[{split}] Reference channel count: {num_channels}")
            elif X.shape[1] != num_channels:
                log.warning(f"Skipping {npz_path}: C={X.shape[1]} != {num_channels}")
                continue

            # Validate that T is divisible by step_samples
            T = X.shape[2]
            if T % self.step_samples != 0:
                log.warning(
                    f"Skipping {npz_path}: T={T} not divisible by "
                    f"step_samples={self.step_samples}")
                continue

            all_X.append(X)
            all_y.append(y)

        if not all_X:
            raise RuntimeError(f"No data loaded for split={split}")

        self.X = np.concatenate(all_X, axis=0)   # (N, C, T)
        self.y = np.concatenate(all_y, axis=0)   # (N,)

        N, C, T = self.X.shape
        self.num_nodes = C
        self.seq_len   = T // self.step_samples
        self.input_dim = self.step_samples

        log.info(
            f"[{split}] {N} windows | C={C} T={T} → "
            f"seq_len={self.seq_len} input_dim={self.input_dim} | "
            f"seizure_ratio={self.y.mean():.4f}"
        )

        # Pre-build FC adjacency and supports for the 'combined' graph type.
        # These are shared across all samples (same tensor every call).
        self._fc_adj      = _build_fc_adj(self.num_nodes)
        self._fc_supports = _compute_supports(self._fc_adj, "laplacian")

        self._targets = self.y.tolist()

    def __len__(self):
        return len(self.X)

    def targets(self):
        return self._targets

    def _get_indiv_graph(self, eeg_clip):
        """
        Build per-sample cross-correlation adjacency.

        Args:
            eeg_clip: (seq_len, num_nodes, input_dim) raw (un-standardised) signal
        Returns:
            adj_mat: (num_nodes, num_nodes) float32
        """
        num_nodes = eeg_clip.shape[1]
        adj_mat = np.eye(num_nodes, dtype=np.float32)

        # Flatten time steps: (num_nodes, seq_len * input_dim)
        flat = eeg_clip.transpose(1, 0, 2).reshape(num_nodes, -1)

        for i in range(num_nodes):
            for j in range(i + 1, num_nodes):
                xcorr = comp_xcorr(flat[i], flat[j], mode="valid", normalize=True)
                adj_mat[i, j] = xcorr
                adj_mat[j, i] = xcorr

        adj_mat = np.abs(adj_mat)
        adj_mat = keep_topk(adj_mat, top_k=self.top_k, directed=True)
        return adj_mat

    def __getitem__(self, idx):
        raw   = self.X[idx]          # (C, T)
        label = float(self.y[idx])
        C, T  = raw.shape

        # Reshape raw → (seq_len, num_nodes, input_dim)
        # C=23, T=512 → (23, 2, 256) → (2, 23, 256)
        eeg_clip = raw.reshape(C, self.seq_len, self.input_dim)
        eeg_clip = eeg_clip.transpose(1, 0, 2).copy()   # (seq_len, C, input_dim)

        # Standardise a copy; compute graph on raw signal (mirrors TUSZ behaviour)
        curr_feature = eeg_clip.copy()
        if self.standardize:
            curr_feature = self.scaler.transform(curr_feature)

        if self.graph_type == "individual":
            adj_mat  = self._get_indiv_graph(eeg_clip)
            supports = _compute_supports(adj_mat, self.filter_type)
        else:  # 'combined' → normalised fully-connected fallback
            adj_mat  = self._fc_adj
            supports = self._fc_supports

        x       = torch.FloatTensor(curr_feature)     # (seq_len, num_nodes, input_dim)
        y       = torch.FloatTensor([label])           # (1,)
        seq_len = torch.LongTensor([self.seq_len])     # (1,)

        return x, y, seq_len, supports, adj_mat, f"chbmit_idx{idx}"


# ---------------------------------------------------------------------------
# load_dataset_chbmit
# ---------------------------------------------------------------------------

def load_dataset_chbmit(
        npz_dir,
        train_batch_size=32,
        test_batch_size=64,
        time_step_size=1,
        fs=256,
        graph_type="individual",
        top_k=3,
        filter_type="dual_random_walk",
        standardize=True,
        num_workers=4,
        train_ratio=0.70,
        val_ratio=0.15,
        seed=123):
    """
    Load CHB-MIT NPZ dataset.

    Splits patients (not individual windows) into train / dev / test to avoid
    data leakage across the same recording session.

    Args:
        npz_dir:           directory containing per-patient folders with windows.npz
        train_batch_size:  batch size for training loader
        test_batch_size:   batch size for dev / test loaders
        time_step_size:    seconds per DCRNN time step (default 1)
        fs:                sampling rate (default 256)
        graph_type:        'individual' or 'combined'
        top_k:             k neighbours for individual graph
        filter_type:       'dual_random_walk' or 'laplacian'
        standardize:       z-normalise signals
        num_workers:       DataLoader worker processes
        train_ratio:       fraction of patients for training
        val_ratio:         fraction of patients for validation
        seed:              random seed for patient-level split

    Returns:
        dataloaders: dict {'train', 'dev', 'test'} → DataLoader
        datasets:    dict {'train', 'dev', 'test'} → CHBMITDataset
        scaler:      utils.StandardScaler (or None)
    """
    npz_dir = Path(npz_dir)
    all_npz = sorted(npz_dir.glob("*/windows.npz"))
    if not all_npz:
        raise FileNotFoundError(f"No windows.npz files found under {npz_dir}")

    rng     = np.random.default_rng(seed)
    indices = rng.permutation(len(all_npz))

    n_train = max(1, int(len(all_npz) * train_ratio))
    n_val   = max(1, int(len(all_npz) * val_ratio))

    train_paths = [all_npz[i] for i in indices[:n_train]]
    val_paths   = [all_npz[i] for i in indices[n_train:n_train + n_val]]
    test_paths  = [all_npz[i] for i in indices[n_train + n_val:]]
    if not test_paths:
        test_paths = val_paths   # fallback for small datasets

    log.info(
        f"CHB-MIT patient split: {len(train_paths)} train | "
        f"{len(val_paths)} dev | {len(test_paths)} test"
    )

    scaler = compute_chbmit_scaler(train_paths, fs=fs,
                                   time_step_size=time_step_size) \
             if standardize else None

    split_paths = {"train": train_paths, "dev": val_paths, "test": test_paths}
    dataloaders, datasets = {}, {}

    for split, paths in split_paths.items():
        dataset = CHBMITDataset(
            npz_paths=paths,
            time_step_size=time_step_size,
            fs=fs,
            standardize=standardize,
            scaler=scaler,
            graph_type=graph_type,
            top_k=top_k,
            filter_type=filter_type,
            split=split,
        )
        shuffle    = (split == "train")
        batch_size = train_batch_size if split == "train" else test_batch_size
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            drop_last=(split == "train"),
        )
        dataloaders[split] = loader
        datasets[split]    = dataset

    return dataloaders, datasets, scaler
