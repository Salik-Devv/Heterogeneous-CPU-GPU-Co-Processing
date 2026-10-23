# Heterogeneous CPU-GPU Co-Processing for Data-Parallel Workloads

> Empirical study of when splitting a single data-parallel workload across CPU and GPU actually
> helps — and when it doesn't.

---

## What this is

Four data-parallel workloads — **GEMM**, **Sobel convolution**, **parallel prefix sum**, and
**SpMV** — each implemented as CPU-serial, CPU-OpenMP, GPU-CUDA, and a heterogeneous CPU+GPU
variant, swept across CPU/GPU partition ratios (0–100% CPU fraction, 10% steps) on two platforms
with different GPU memory technologies:

| Platform | GPU | Memory | Role |
|---|---|---|---|
| **Local** | NVIDIA RTX 4060 Laptop (Ada) | GDDR6, 272 GB/s | Primary — full NCU profiling |
| **Kaggle** | NVIDIA Tesla P100 (Pascal) | HBM2, 732 GB/s | Cloud validation |

GPU kernels on the local platform are additionally profiled with **NVIDIA Nsight Compute (NCU)**
to trace wall-time behaviour back to specific microarchitectural causes. NCU is blocked on Kaggle,
so that platform is timing-only.

## Key findings

- **GPU-only wins everywhere on the local RTX 4060 platform** — co-processing never beats the
  GPU-only baseline for any of the four workloads there.
- **Three failure modes** explain why: *compute imbalance* (GEMM, convolution — GPU/CPU
  throughput ratio 138–171×), *mandatory inter-pass synchronisation* (prefix sum — algorithmic,
  not fixable with faster hardware), and *PCIe transfer domination* (SpMV — H2D/D2H is 94–97%
  of GPU wall time).
- **Three co-processing wins on the Kaggle P100**, where the GPU/CPU throughput ratio for the
  bottleneck resource drops to ≲6×: SpMV (**5.8×**, via H2D elimination), convolution (**2.5×**,
  via HBM2 bandwidth), prefix sum (**14%**, partial CPU-GPU overlap).
- **NCU isolates a distinct bottleneck per kernel:** register pressure (GEMM, 33% occupancy),
  warp-latency starvation (convolution, 0.72 eligible warps/scheduler), shared-memory bank
  conflicts (prefix sum, 75%+ of accesses serialised), uncoalesced global memory access (SpMV,
  65% of fetched sectors wasted).
- Co-processing benefit turns out to be a property of the **hardware throughput ratio**, not the
  algorithm — the same SpMV kernel is transfer-dominated and neutral on the laptop GPU, but
  strongly beneficial on the KVM-constrained cloud instance.

## Repository structure

The local (RTX 4060) platform lives at the repo root; Kaggle gets its own subfolder since it's
the validation run, not a parallel copy of the whole project.

```
.
├── README.md
├── workloads/                    # One folder per workload, one subfolder per implementation
│   ├── gemm/
│   │   ├── cpu/                   # CMakeLists.txt → gemm_cpu, gemm_cpu_omp (build/bin/)
│   │   ├── gpu/                   # naive / tiled / cuBLAS CUDA kernels
│   │   └── hetero/                # CPU+GPU split variant
│   ├── convolution/
│   │   ├── cpu/                   # CMakeLists.txt → conv_cpu, conv_cpu_omp (build/bin/)
│   │   ├── gpu/
│   │   └── hetero/
│   ├── prefix_sum/
│   │   ├── cpu/
│   │   ├── gpu/
│   │   └── hetero/
│   └── spmv/
│       ├── cpu/
│       ├── gpu/
│       └── hetero/
├── benchmarks/
│   ├── report/                    # Raw NCU reports (--set full)
│   │   ├── gemm_report_ncu.pdf
│   │   ├── conv_report_ncu.pdf
│   │   ├── prefix_sum_gpu_report_ncu.pdf
│   │   └── spmv_report.pdf
│   ├── ncu_roofline.png
│   ├── ncu_throughput.png
│   └── specs.png
├── results/
│   ├── *_results.csv              # Raw timing data per workload (written by the executables)
│   ├── *_table.py                 # Table generation scripts
│   └── *_table.png
├── analysis/
│   ├── *_analysis.py              # Plotting scripts (GFLOPS, bandwidth, split-ratio sweeps)
│   └── *.png                      # Generated figures (Figs. 1–6 in the paper)
├── environment/                   # Setup notes / dependency list
└── kaggle/                        # Tesla P100 — cloud validation; mirrors the layout above, minus NCU
```

*The `gemm/cpu/` and `convolution/cpu/` layouts above are confirmed from the CMakeLists.txt and
build log — `gpu/` and `hetero/` are shown as the same per-variant pattern; adjust if yours differ.*

## Platforms in detail

| Component | Local — Ubuntu 24.04 (Primary) | Kaggle Notebook (Validation) |
|---|---|---|
| CPU | AMD Ryzen 7 7840HS (Zen 4), 8C/16T, 3.8–5.1 GHz, L3 16 MB, ~1305 GFLOPS FP32 (AVX-512) | Intel Xeon @ 2.00 GHz (KVM guest), 2C/4T, Skylake-SP/Cascade Lake, AVX-512 |
| RAM | 16 GB DDR5-5600 dual-ch, theoretical 89.6 GB/s, STREAM Triad 30.09 GB/s native (27.77 GB/s under WSL2) | ~29 GB DDR4-3200 (est.), theoretical ~51 GB/s, STREAM unavailable (KVM env.) |
| GPU | NVIDIA RTX 4060 Laptop (Ada), 24 SMs, 3072 CUDA cores, 8 GB GDDR6 @ 272 GB/s, 15,670 GFLOPS FP32 | NVIDIA Tesla P100-PCIE (Pascal), 56 SMs, 3584 CUDA cores, 16 GB HBM2 @ 732 GB/s, ~9,300 GFLOPS FP32 |
| Interconnect | PCIe Gen 4 x8, ~15–16 GB/s measured | PCIe (P100-PCIE, no NVLink) |
| OMP threads | 16 (8C × 2T) | 4 (2C × 2T, KVM-limited) |
| NCU profiling | Available (`--set full`) | Blocked by platform policy |
| OS / Compiler | Ubuntu 24.04 LTS native, GCC 12.4, NVCC CUDA 12.x | Kaggle Docker (Ubuntu base), GCC + NVCC CUDA 13.0 |

## Workloads

| Workload | Description | Arithmetic intensity |
|---|---|---|
| **GEMM** | C = A×B, N×N single-precision, naive/tiled (16×16)/cuBLAS GPU kernels; hetero splits rows of C between OpenMP and GPU-tiled | ~N/6 FLOP/byte — compute-bound at N≥512 |
| **Sobel convolution** | 3×3 Gx/Gy edge detection on a 5824×3264 grayscale image; tiled GPU kernel with shared-memory halo | ~0.3 FLOP/byte |
| **Parallel prefix sum** | Inclusive scan over 32-bit floats; three-pass Blelloch-style GPU scan with a mandatory CPU round-trip for block-total correction | ~0.12 FLOP/byte |
| **SpMV** | y = Ax, CSR format, 32 nnz/row (synthetic uniform sparsity); warp-per-row GPU kernel | ~0.15 FLOP/byte |

## Methodology

- n = 7–8 runs per configuration, one warm-up run discarded.
- GPU kernel timing via CUDA events (`cudaEventElapsedTime`); PCIe transfers timed on a
  dedicated stream. Heterogeneous wall time via `CLOCK_MONOTONIC`, from concurrent CPU-thread +
  GPU-stream launch to D→H completion. Results reported as mean ± 1 SD.
- Sweep ranges: GEMM N ∈ {512, 1024, 2048}; convolution fixed at 5824×3264; prefix sum
  N ∈ {64K, 1M, 4M, 16M, 64M, 128M}; SpMV M ∈ {64K×64K (2M nnz), 128K×128K (4M nnz),
  256K×256K (8M nnz)}. CPU fraction α swept 0.0–1.0 in 0.1 steps for all heterogeneous runs.

## Reproducing the results

Each workload/implementation folder has its own `CMakeLists.txt` and builds out-of-source into
`build/`, with executables landing in `build/bin/`:

```bash
cd workloads/<workload>/cpu          # or gpu/, hetero/
mkdir -p build && cd build
cmake .. && make

# Each executable takes a size argument and appends its own result
# to the shared CSV at the project root (e.g. results/gemm_results.csv)
./bin/<workload>_cpu <size>            # e.g. ./bin/gemm_cpu 1024
./bin/<workload>_cpu_omp <size>
```

```bash
# Profile with NCU (local platform only)
ncu --set full -o benchmarks/report/<workload>_report_ncu ./bin/<workload>_gpu_tiled <size>

# Generate tables and figures
cd results && python <workload>_table.py
cd ../analysis && python <workload>_analysis.py
```

## Requirements

- CMake 3.18+
- CUDA Toolkit 12.x+ (NVCC), matching GCC (12.4 tested locally)
- cuBLAS (bundled with the CUDA Toolkit)
- OpenMP (`-fopenmp`)
- NVIDIA Nsight Compute (NCU) — local platform only, for kernel profiling
- POSIX pthreads — for concurrent CPU/GPU launch in the heterogeneous variants
- Python 3 with `numpy`, `pandas`, `matplotlib` — for the `analysis/` and `results/*_table.py` scripts

## Citation

Paper is currently under review; citation details will be updated on acceptance.

```
Mohammad Salik Dev, "Performance and Bottleneck Analysis of Heterogeneous CPU-GPU Co-Processing
for Data-Parallel Workloads," in submission, ICMACC 2026.
```

## License

No license file yet. Add one (MIT/BSD-3-Clause/Apache-2.0 are common choices for research code)
before making the repo public — without one, GitHub's default all-rights-reserved terms apply.
