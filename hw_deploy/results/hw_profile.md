# HW profile: Corr-DCRNN teacher vs EEGNet-8,2 student

Per clip: 18 ch x 1200 samples (6 s @ 200 Hz). BN folded for deployment totals. Top-3 graph -> [72, 72] nnz per support.

## Summary

| | MAC | nonlinear (exp/sigm/tanh/log/sqrt/div) | add+mul | cmp | params | input buffer | peak live act | on-chip (params+act) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Teacher model only | 31,284,864 | 41,472 | 362,898 | 1,169 | 280,769 | 14,571 | 31,608 | 312,377 |
| Teacher + preproc (FFT) | 31,665,971 | 63,144 | 560,999 | 2,411 | 284,369 | 14,571 | 31,608 | 315,977 |
| Teacher + preproc (DFT) | 35,820,864 | 63,144 | 395,892 | 2,411 | 284,369 | 14,571 | 31,608 | 315,977 |
| Student (original) | 11,569,600 | 24,032 | 47,970 | 24,032 | 2,498 | 21,600 | 194,544 | 197,042 |
| Student (reordered) | 1,730,720 | 24,032 | 72,002 | 24,032 | 3,042 | 21,600 | 40,800 | 43,842 |
| Student (reordered, streaming) | 1,730,720 | 24,032 | 72,002 | 24,032 | 3,042 | 18 | 1,348 | 4,390 |

Memory in values; x1 B (int8) or x2 B (int16) for bytes.

## Ratios (teacher / student)

| student | params | MAC vs teacher model | MAC vs teacher + preproc | peak act | on-chip |
|---|---:|---:|---:|---:|---:|
| Student (original) | 112.4x | 2.70x | 2.74x | 0.16x | 1.60x |
| Student (reordered) | 92.3x | 18.08x | 18.30x | 0.77x | 7.21x |
| Student (reordered, streaming) | 92.3x | 18.08x | 18.30x | 23.45x | 71.98x |

## Equivalence (original vs reordered student)

- random, input std 1: max |logit diff| fp64 2.36e-08, fp32 2.09e-07 (max |logit| 4.92e-01)
- random, input std 50: max |logit diff| fp64 4.25e-08, fp32 1.79e-07 (max |logit| 6.08e-01)

## MAC cross-check (torch.utils.flop_counter)

- student: ours 11,569,600 / torch 11,569,600
- student_reordered: ours 1,730,720 / torch 1,730,720
- teacher_dense: ours 34,816,896 / torch 34,816,896

## Teacher memory breakdown (values)

- raw_segment: 3,600
- fft_feature_frames: 10,800
- gram_acc: 171
- sparse_graph: 288
- hidden_states: 2,304
- dconv_working_set: 18,216

## Reordered student, streaming memory breakdown (values)

- input_sample: 18
- temporal_line_buffer: 1,024
- pool1_acc: 16
- sep_dw_line_buffer: 256
- sep_pw_out: 16
- pool2_acc: 16
- classifier_acc: 2

## Per-layer

### Student (original)

| layer | out | MAC | add | mul | exp | sigm | tanh | log | sqrt | div | cmp | params | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| firstconv.0 [conv k=(1, 64) g=1] | (8, 18, 1201) | 11,068,416 | - | - | - | - | - | - | - | - | - | 512 |  |
| firstconv.1 [BN] *(foldable)* | (8, 18, 1201) | - | 172,944 | 172,944 | - | - | - | - | - | - | - | 16 | folded into prev conv |
| depthwise.0 [depthwise k=(18, 1) g=8] | (16, 1, 1201) | 345,888 | - | - | - | - | - | - | - | - | - | 288 |  |
| depthwise.1 [BN] *(foldable)* | (16, 1, 1201) | - | 19,216 | 19,216 | - | - | - | - | - | - | - | 32 | folded into prev conv |
| depthwise.2 [ELU] | (16, 1, 1201) | - | 19,216 | - | 19,216 | - | - | - | - | - | 19,216 | - |  |
| depthwise.3 [avgpool k=4] | (16, 1, 300) | - | 14,400 | 4,800 | - | - | - | - | - | - | - | - | 1/k scale foldable into next linear |
| separable.0 [depthwise k=(1, 16) g=16] | (16, 1, 301) | 77,056 | - | - | - | - | - | - | - | - | - | 256 |  |
| separable.1 [conv k=(1, 1) g=1] | (16, 1, 301) | 77,056 | - | - | - | - | - | - | - | - | - | 256 |  |
| separable.2 [BN] *(foldable)* | (16, 1, 301) | - | 4,816 | 4,816 | - | - | - | - | - | - | - | 32 | folded into prev conv |
| separable.3 [ELU] | (16, 1, 301) | - | 4,816 | - | 4,816 | - | - | - | - | - | 4,816 | - |  |
| separable.4 [avgpool k=8] | (16, 1, 37) | - | 4,144 | 592 | - | - | - | - | - | - | - | - | 1/k scale foldable into next linear |
| classify [linear 592->2] | (2,) | 1,184 | 2 | - | - | - | - | - | - | - | - | 1,186 |  |
| **total (deploy)** | | **11,569,600** | **42,578** | **5,392** | **24,032** | **-** | **-** | **-** | **-** | **-** | **24,032** | **2,498** | |

### Student (reordered, BN-folded)

| layer | out | MAC | add | mul | exp | sigm | tanh | log | sqrt | div | cmp | params | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| spatial [conv k=(1,) g=1] | (16, 1200) | 345,600 | - | - | - | - | - | - | - | - | - | 288 |  |
| temporal [depthwise k=(64,) g=16] | (16, 1201) | 1,229,824 | 19,216 | - | - | - | - | - | - | - | - | 1,040 |  |
| elu1 [ELU] | (16, 1201) | - | 19,216 | - | 19,216 | - | - | - | - | - | 19,216 | - |  |
| pool1 [avgpool k=4] | (16, 300) | - | 14,400 | 4,800 | - | - | - | - | - | - | - | - | 1/k scale foldable into next linear |
| sep_dw [depthwise k=(16,) g=16] | (16, 301) | 77,056 | - | - | - | - | - | - | - | - | - | 256 |  |
| sep_pw [conv k=(1,) g=1] | (16, 301) | 77,056 | 4,816 | - | - | - | - | - | - | - | - | 272 |  |
| elu2 [ELU] | (16, 301) | - | 4,816 | - | 4,816 | - | - | - | - | - | 4,816 | - |  |
| pool2 [avgpool k=8] | (16, 37) | - | 4,144 | 592 | - | - | - | - | - | - | - | - | 1/k scale foldable into next linear |
| classify [linear 592->2] | (2,) | 1,184 | 2 | - | - | - | - | - | - | - | - | 1,186 |  |
| **total (deploy)** | | **1,730,720** | **66,610** | **5,392** | **24,032** | **-** | **-** | **-** | **-** | **-** | **24,032** | **3,042** | |

### Teacher model (per clip, 6 steps)

| layer | out | MAC | add | mul | exp | sigm | tanh | log | sqrt | div | cmp | params | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| L1 gate(r,u) diffusion S^k x  (x6 steps) | (5, 18, 164) | 283,392 | 35,424 | 35,424 | - | - | - | - | - | - | - | - | dense S would be 1,275,264 MAC; top-k graph -> sparse |
| L1 gate(r,u) weight [820x128]  (x6 steps) | (18, 128) | 11,335,680 | 13,824 | - | - | - | - | - | - | - | - | 105,088 |  |
| L1 cand(c) diffusion S^k x  (x6 steps) | (5, 18, 164) | 283,392 | 35,424 | 35,424 | - | - | - | - | - | - | - | - | dense S would be 1,275,264 MAC; top-k graph -> sparse |
| L1 cand(c) weight [820x64]  (x6 steps) | (18, 64) | 5,667,840 | 6,912 | - | - | - | - | - | - | - | - | 52,544 |  |
| L1 GRU gating  (x6 steps) | (18, 64) | - | 13,824 | 20,736 | - | 13,824 | 6,912 | - | - | - | - | - | sigma(r,u), tanh(c), r*h, u*h+(1-u)*c |
| L2 gate(r,u) diffusion S^k x  (x6 steps) | (5, 18, 128) | 221,184 | 27,648 | 27,648 | - | - | - | - | - | - | - | - | dense S would be 995,328 MAC; top-k graph -> sparse |
| L2 gate(r,u) weight [640x128]  (x6 steps) | (18, 128) | 8,847,360 | 13,824 | - | - | - | - | - | - | - | - | 82,048 |  |
| L2 cand(c) diffusion S^k x  (x6 steps) | (5, 18, 128) | 221,184 | 27,648 | 27,648 | - | - | - | - | - | - | - | - | dense S would be 995,328 MAC; top-k graph -> sparse |
| L2 cand(c) weight [640x64]  (x6 steps) | (18, 64) | 4,423,680 | 6,912 | - | - | - | - | - | - | - | - | 41,024 |  |
| L2 GRU gating  (x6 steps) | (18, 64) | - | 13,824 | 20,736 | - | 13,824 | 6,912 | - | - | - | - | - | sigma(r,u), tanh(c), r*h, u*h+(1-u)*c |
| head relu + fc 64->1 per node | (18, 1) | 1,152 | 18 | - | - | - | - | - | - | - | 1,152 | 65 |  |
| max-pool over nodes | (1,) | - | - | - | - | - | - | - | - | - | 17 | - |  |
| **total (deploy)** | | **31,284,864** | **195,282** | **167,616** | **-** | **27,648** | **13,824** | **-** | **-** | **-** | **1,169** | **280,769** | |

### Teacher preprocessing (FFT estimate)

| layer | out | MAC | add | mul | exp | sigm | tanh | log | sqrt | div | cmp | params | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FFT x108 (fft) | (6, 18, 100) | 165,107 | 165,107 | - | - | - | - | - | - | - | - | - | real 200-pt FFT, radix estimate (lower bound; 200 not a power of 2) |
| |X| + log | (6, 18, 100) | - | 10,800 | 21,600 | - | - | - | 10,800 | 10,800 | - | - | - | log|X| = 0.5 log(re^2+im^2) can drop the sqrt |
| z-norm (scaler) | (6, 18, 100) | 10,800 | - | - | - | - | - | - | - | - | - | 3,600 | (x-mu)/sigma as x*s+b; per-(C,M) constants |
| gram X X^T (symmetric) | (18, 18) | 205,200 | - | - | - | - | - | - | - | - | - | - | needs the whole 6-s window before DCGRU can start |
| corr normalise | (18, 18) | - | - | 342 | - | - | - | - | 18 | 18 | - | - | 1/||x|| per ch, then 2 mul per pair |
| |.| + top-3 select | (18, 18) | - | - | - | - | - | - | - | - | - | 1,242 | - | partial selection per row |
| dual random-walk D^-1 W | (2, 18, 18) | - | 108 | 144 | - | - | - | - | - | 36 | - | - | row/col sums, reciprocal, scale nnz |
| **total (deploy)** | | **381,107** | **176,015** | **22,086** | **-** | **-** | **-** | **10,800** | **10,818** | **54** | **1,242** | **3,600** | |

### Teacher preprocessing (direct DFT)

| layer | out | MAC | add | mul | exp | sigm | tanh | log | sqrt | div | cmp | params | note |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| FFT x108 (dft) | (6, 18, 100) | 4,320,000 | - | - | - | - | - | - | - | - | - | - | direct DFT of 100 kept bins (upper bound, trivially pipelined) |
| |X| + log | (6, 18, 100) | - | 10,800 | 21,600 | - | - | - | 10,800 | 10,800 | - | - | - | log|X| = 0.5 log(re^2+im^2) can drop the sqrt |
| z-norm (scaler) | (6, 18, 100) | 10,800 | - | - | - | - | - | - | - | - | - | 3,600 | (x-mu)/sigma as x*s+b; per-(C,M) constants |
| gram X X^T (symmetric) | (18, 18) | 205,200 | - | - | - | - | - | - | - | - | - | - | needs the whole 6-s window before DCGRU can start |
| corr normalise | (18, 18) | - | - | 342 | - | - | - | - | 18 | 18 | - | - | 1/||x|| per ch, then 2 mul per pair |
| |.| + top-3 select | (18, 18) | - | - | - | - | - | - | - | - | - | 1,242 | - | partial selection per row |
| dual random-walk D^-1 W | (2, 18, 18) | - | 108 | 144 | - | - | - | - | - | 36 | - | - | row/col sums, reciprocal, scale nnz |
| **total (deploy)** | | **4,536,000** | **10,908** | **22,086** | **-** | **-** | **-** | **10,800** | **10,818** | **54** | **1,242** | **3,600** | |
