# Benchmark results (v1.0.0 development, reference machine)

Output of `python benchmarks/bench_hotpaths.py` after memoizing the FFT time-origin
phase (before: FFT 533 us / transform, GNLSE step 8.08 ms, mode-locked round trip 11.8 ms).

Python 3.11.15, NumPy 2.4.6, SciPy 1.17.1, 4 CPUs, x86_64

| hot path | size | time | unit |
|---|---|---|---|
| FFT (numerics.fft.spectrum) | N = 8192 | 198 | us / transform |
| GNLSE RK4IP step (Raman + shock) | N = 8192 | 3.3 | ms / step |
| SSFM step (Kerr + beta2) | N = 8192 | 0.691 | ms / step |
| Laser rate equations, RK4 (Python loop) | 8000 steps | 7.41 | us / step |
| 2x2 MIMO CMA update (Python loop) | 15 taps | 10.3 | us / update |
| Mode-locked laser round trip | N = 1024, 20 fiber steps | 5.61 | ms / round trip |
| Coherent QPSK link, full run | 32767 symbols x 4 sps | 0.961 | s / run |
| Coherent sweep 4 OSNR x 2 trials, serial | shared cache | 3.14 | s |
|   same with 4 process workers | speedup 0.90x | 3.48 | s |
| Coherent Monte Carlo 16 trials, serial | shared cache | 6.76 | s |
|   same with 4 process workers | speedup 1.42x | 4.75 | s |
| Tiny sweep (reference, 3200 runs), serial |  | 0.91 | s |
|   same with 4 process workers | speedup 0.52x | 1.74 | s |
