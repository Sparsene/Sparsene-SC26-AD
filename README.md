# Sparsene-SC26-AD

Artifact Description for the SC'26 paper: **Sparsene: A Format-Driven Automated Optimization Framework for SpMM on Modern GPUs**.

---

## Overview of Contributions and Artifacts

Sparsene is a format-driven automated optimization framework for SpMM that systematically derives kernel optimizations from a hierarchical sparse format DSL, achieving state-of-the-art performance on modern GPUs without manual rewriting.

| Contribution | Description |
|---|---|
| **C1** – Hierarchical Sparse Format Representation | A sparse format DSL with declarative format-child primitives and dynamic transformation primitives that systematically lower user-defined hierarchical sparse formats into hardware-aligned physical layouts. |
| **C2** – Automatic Pipeline Construction | Pipeline plans over the pGraph abstraction, efficiently identified through dependency-aware sandwich profiling, a hardware-aware pipeline simulator, and a two-stage hybrid search algorithm. |
| **C3** – Format-Aware Input Load Balancing | Multi-binding and strict load balancing strategies automatically selected based on the format's workload distribution. |

### Artifact–Contribution Mapping

| Artifact ID | Sub ID | Contributions | Related Paper Elements |
|---|---|---|---|
| A1 | A1.1 | C1 | Figure 7, Table I |
| A1 | A1.2 | C2 | Table III, Figure 11, Table IV, Figure 14 |
| A1 | A1.3 | C3 | Figure 13 |
| A1 | A1.4 | C2, C3 | Table V |

---

## Artifact Setup

### 0. Clone This Repository

```bash
git clone https://github.com/Sparsene/Sparsene-SC26-AD.git
cd Sparsene-SC26-AD
```

### 1. Hardware Requirements

We evaluate performance on three NVIDIA GPUs:

- **A100** – Ampere architecture, Compute Capability 8.0, 80 GB global memory
- **H100** – Hopper architecture, Compute Capability 9.0, 80 GB global memory
- **RTX 4090** – Ada architecture, Compute Capability 8.6, 24 GB global memory

### 2. Software Dependencies

- Python 3.10, setuptools 59.6
- NVCC 12.6 (compiled with `-O3`; all experiments use FP32 precision)
- Python packages:

```bash
pip install tqdm pandas matplotlib
```

### 3. Baselines

| Baseline | Source |
|---|---|
| Sputnik | https://github.com/google-research/sputnik |
| cuSPARSE v12 | https://developer.nvidia.com/cuda-12-6-0-download-archive |
| DTC-SpMM | https://github.com/HPMLL/DTC-SpMM_ASPLOS24 |
| Acc-SpMM | https://github.com/Hyaloid/AccSpMM |
| SparseTIR | https://github.com/uwsampl/SparseTIR |
| FlashSparse | https://github.com/ParCIS/FlashSparse |

### 4. Datasets / Inputs

We use sparse matrices from SuiteSparse, as well as datasets from TC-GNN, SNAP, DGL, and OGB. All 26 matrices used in experiments are listed in Table II of the paper.

**Download the dataset:**

Download `sparsene_sc26_dataset.tar.gz` from Google Drive and extract it into the `Sparsene/dataset` directory:

https://drive.google.com/drive/folders/1f5KP5H1jxJ98tB_cFiRvJZcKTp6_E94q?usp=sharing

```bash
tar -xzf sparsene_sc26_dataset.tar.gz -C Sparsene/dataset/
```

**Convert matrices for DTC-SpMM and FlashSparse:**

```bash
# Convert .mtx to .npz for DTC-SpMM
python mtx2npz_selected.py --input_dir ./dataset/mtx/ --output_dir ./selected_npz/

# Convert .npz to FlashSparse format
python flashsparse_convert_parallel.py --input_dir ./selected_npz/ --output_dir ./flashsparse_npz/
```

---

## Installation and Deployment

### Sparsene and Baselines Installation (~20 min)

We provide an `install.sh` script for Sparsene and for each baseline in their respective directories:

```bash
# Install Sparsene
cd Sparsene && bash install.sh && cd ..

# Install DTC-SpMM
cd DTC-SpMM && bash install.sh && cd ..

# Install Acc-SpMM
cd Acc-SpMM && bash install.sh && cd ..

# Install SparseTIR
cd SparseTIR && bash install.sh && cd ..

# Install FlashSparse
cd FlashSparse && bash install.sh && cd ..
```

> **Note:** Sputnik and cuSPARSE are included in the Sparsene and FlashSparse scripts and do not require separate installation.

### End-to-End Experiment Baselines

The end-to-end experiments (T2) require three additional baselines (PyG, DGL, DTC-SpMM end2end) with separate install scripts. See `Sparsene/end2end/` for details.

### Quick Test (~5 min)

After installation, run a quick test to verify correctness of Sparsene and all baselines:

```bash
cd Sparsene && bash quick_test.sh
```

---

## Artifact Execution

The full execution flow is: **T1 → T2 → T3 → T4 → T5 → T7**, or use **T6 → T7** for quick reproduction from pre-collected data.

### T1: Run Sparsene and All Baselines (~300 min)

```bash
cd Sparsene   && bash run_sparsene_kernel.sh    && cd ..
cd DTC-SpMM   && bash run_dtc_spmm_kernel.sh    && cd ..
cd Acc-SpMM   && bash run_acc_spmm_kernel.sh    && cd ..
cd SparseTIR  && bash run_sparsetir_kernel.sh   && cd ..
cd FlashSparse && bash run_flashsparse_kernel.sh && cd ..
```

> cuSPARSE and Sputnik results are collected within the Sparsene and FlashSparse scripts.

### T2: Run End-to-End Experiments (~240 min)

```bash
cd Sparsene/end2end
bash run_sparsene_end2end.sh
bash run_flashsparse_end2end.sh
bash run_dtc_spmm_end2end.sh
bash run_dgl_end2end.sh
bash run_pyg_end2end.sh
cd ../..
```

### T3: Run Load Balance Experiments (~5 min)

```bash
cd Sparsene/load_balance
bash build.sh          # Build the load balance operators
bash run_load_balance.sh
cd ../..
```

### T4: Run Hardware-Aware Pipeline Simulation (~600 min)

```bash
cd Sparsene/simulator
bash run_simulator_accuracy.sh
cd ../..
```

### T5: Run Search Convergence Experiments (~180 min)

```bash
cd Sparsene/search_convergence
bash run_search_convergence.sh
cd ../..
```

### T6: Quick Reproduction from Pre-collected Data

We provide all pre-collected experimental results in `Sparsene_AD/results_precomputed/`. To skip T1–T5:

```bash
bash quick_reproduce.sh
```

This copies the pre-collected data into `Sparsene_AD/results/`.

### T7: Generate Figures and Tables

```bash
cd Sparsene
bash generate_figures_tables.sh
cd ..
```

This produces all figures and tables in the paper: Figure 11, Figure 13, Figure 14, Table III, Table IV, and Table V.

---

## Expected Results

### A1.1 – Hierarchical Format DSL (C1)

The DSL successfully expresses all SOTA hierarchical sparse formats shown in Figure 7 (BIT-TCF, BIT-BSR, ME-TCF, SR-BCRS). Generated kernels produce numerically correct results matching handwritten implementations across all three GPU platforms.

### A1.2 – Automatic Pipeline Construction (C2)

- **Overall Performance (Table III, Figure 11):** Sparsene consistently outperforms all six baselines across A100, H100, and RTX 4090.
- **Simulator Accuracy (Table IV):** The simulator achieves reasonable ranking accuracy (P@k) with small candidate pools and reliably identifies promising candidates as the pool grows.
- **Search Convergence (Figure 14):** The simulator-guided search converges significantly faster than the structure-only variant, reaching near-optimal performance with fewer on-device evaluations.

### A1.3 – Format-Aware Load Balancing (C3)

- **mip1:** Multi-binding delivers significant speedup by distributing the straggler row window across multiple thread blocks.
- **ddi:** Strict load balance achieves consistent improvement by evenly distributing irregular workloads across all SMs.

### A1.4 – End-to-End GCN Training (C2, C3)

Sparsene achieves significant speedups over PyG, DGL, and DTC-SpMM in end-to-end GCN training. The gains over FlashSparse are modest but approach the Amdahl's law theoretical ceiling (~48% SpMM time share).

---

## Estimated Reproduction Time

Taking A100 GPU as an example:

| Task | Time |
|---|---|
| T1 – Kernel benchmarks | ~300 min |
| T2 – End-to-end experiments | ~240 min |
| T3 – Load balance | ~5 min |
| T4 – Simulator accuracy | ~600 min |
| T5 – Search convergence | ~180 min |
| **Total** | **~1325 min** |

Use **T6 → T7** for instant reproduction from pre-collected data.