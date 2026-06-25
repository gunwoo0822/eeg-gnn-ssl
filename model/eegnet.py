"""
EEGNet student model for Knowledge Distillation on CHB-MIT.

Self-contained EEGNet-8,2 implementation (Lawhern et al. 2018, thesis ref [11])
in pure PyTorch — no braindecode / torchaudio dependency, so it runs in the
teacher's existing venv (torch 2.5.1+cu118) with no extra installs.

The student consumes the *raw* time-domain clip, shape (B, C, T)
  e.g. (B, 18, 1200) = 18 channels × (6 s × 200 Hz),
NOT the FFT features or correlation graph used by the teacher.

``forward`` mirrors the teacher's ``model(x, seq_lengths, supports)`` signature
so the same training/eval plumbing can drive either model; the student ignores
``seq_lengths`` and ``supports``.

Output: logits of shape (B, n_outputs). For binary seizure detection we use
n_outputs=2 (softmax); the positive-class probability is softmax(logits)[:, 1].

Architecture (EEGNet-8,2 defaults: F1=8, D=2, F2=16):
  Block1: temporal Conv2d(1→F1, (1,kernel_length)) + BN
          depthwise Conv2d(F1→F1*D, (C,1), groups=F1) + BN + ELU + AvgPool(1,p1) + Drop
  Block2: separable [depthwise (1,16) + pointwise (1,1)] (→F2) + BN + ELU + AvgPool(1,p2) + Drop
  Head:   Flatten + Linear(→n_outputs)
"""

import torch
import torch.nn as nn


class EEGNetStudent(nn.Module):
    def __init__(self, n_chans, n_times, sfreq=None,
                 n_outputs=2, drop_prob=0.5,
                 F1=8, D=2, F2=16, kernel_length=64,
                 pool1=4, pool2=8, **_unused):
        super().__init__()
        self.n_outputs = n_outputs

        # ── Block 1: temporal conv (band-pass filters) ──────────────────────
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel_length),
                      padding=(0, kernel_length // 2), bias=False),
            nn.BatchNorm2d(F1),
        )
        # depthwise spatial conv: one spatial filter per temporal feature map
        self.depthwise = nn.Sequential(
            nn.Conv2d(F1, F1 * D, (n_chans, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D),
            nn.ELU(),
            nn.AvgPool2d((1, pool1)),
            nn.Dropout(drop_prob),
        )
        # ── Block 2: separable conv (depthwise temporal + pointwise mix) ─────
        self.separable = nn.Sequential(
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8),
                      groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, (1, 1), bias=False),
            nn.BatchNorm2d(F2),
            nn.ELU(),
            nn.AvgPool2d((1, pool2)),
            nn.Dropout(drop_prob),
        )
        # ── Classifier (LazyLinear infers flattened dim on first forward) ───
        self.flatten = nn.Flatten()
        self.classify = nn.LazyLinear(n_outputs)

        # Materialise the LazyLinear immediately via a dummy forward, so that
        # parameter counting, optimizer construction, and checkpoint load/save
        # all work BEFORE the first real batch. eval() avoids touching BN
        # running stats; no_grad() avoids building a graph.
        self.eval()
        with torch.no_grad():
            self.forward(torch.zeros(1, n_chans, n_times))
        self.train()

    def forward(self, x, seq_lengths=None, supports=None):
        """x: (B, C, T) raw signal. seq_lengths/supports ignored (teacher-only)."""
        if x.dim() == 3:                       # (B, C, T) → (B, 1, C, T)
            x = x.unsqueeze(1)
        x = self.firstconv(x)
        x = self.depthwise(x)
        x = self.separable(x)
        x = self.flatten(x)
        return self.classify(x)                # (B, n_outputs) logits
