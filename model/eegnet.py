"""
EEGNet student model for Knowledge Distillation on CHB-MIT.

Thin wrapper around braindecode's EEGNet (EEGNet-8,2: F1=8, D=2, F2=16),
the lightweight CNN student distilled from the Corr-DCRNN teacher
(see thesis §4.5).

The student consumes the *raw* time-domain clip, shape (B, C, T)
  e.g. (B, 18, 1200)  = 18 channels × (6 s × 200 Hz),
NOT the FFT features or correlation graph used by the teacher.

The ``forward`` signature mirrors the teacher's
``model(x, seq_lengths, supports)`` so the same training/eval plumbing can
drive either model; the student simply ignores ``seq_lengths`` and
``supports``.

Output: logits of shape (B, n_outputs). For binary seizure detection we use
n_outputs=2 (softmax) following the working braindecode notebook; the
positive-class probability is ``softmax(logits)[:, 1]``.
"""

import torch.nn as nn

from braindecode.models import EEGNet


class EEGNetStudent(nn.Module):
    """braindecode EEGNet wrapped with the teacher's call signature.

    Args:
        n_chans:   EEG channel count (paper: 18).
        n_times:   samples per clip (paper: 6 s × 200 Hz = 1200).
        sfreq:     sampling rate in Hz (paper: 200).
        n_outputs: output classes (2 → softmax, binary).
        drop_prob: dropout probability (tuned on dev).
        **eegnet_kwargs: forwarded to braindecode EEGNet
                         (e.g. F1, D, F2, kernel_length). Defaults give
                         EEGNet-8,2 (F1=8, D=2, F2=16).
    """

    def __init__(self, n_chans, n_times, sfreq,
                 n_outputs=2, drop_prob=0.5, **eegnet_kwargs):
        super().__init__()
        self.n_outputs = n_outputs
        self.net = EEGNet(
            n_chans=n_chans,
            n_outputs=n_outputs,
            n_times=n_times,
            sfreq=sfreq,
            drop_prob=drop_prob,
            **eegnet_kwargs,
        )

    def forward(self, x, seq_lengths=None, supports=None):
        """x: (B, C, T) raw signal. seq_lengths/supports ignored (teacher-only)."""
        return self.net(x)            # (B, n_outputs) logits
