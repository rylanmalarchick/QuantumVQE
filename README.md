# Quantum VQE for H2 Molecule

Variational Quantum Eigensolver (VQE) implementation for computing the ground state energy of molecular hydrogen (H2) using PennyLane, optimized for HPC clusters with multi-GPU support.

## Overview

This project computes the potential energy surface of the H2 molecule with the Variational Quantum Eigensolver. It runs the same scan with a serial PennyLane baseline, an Optax + Catalyst JIT version, a `lightning.gpu` version, and an MPI version on the ERAU Vega HPC cluster (4× NVIDIA H100). The write-up is arXiv:2601.09951 (v2 corrects the speedup claims of v1).

### The Model

The VQE algorithm finds the ground state energy by optimizing a parameterized quantum circuit (ansatz) to minimize the expectation value of the molecular Hamiltonian:

$$E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle$$

where $|\psi(\theta)\rangle$ is the quantum state prepared by our ansatz and $H$ is the molecular Hamiltonian.

**System Details:**
- **Molecule**: H2 (hydrogen dimer)
- **Qubits**: 4 (scaling study up to 26 qubits)
- **Ansatz**: DoubleExcitation gate with Hartree-Fock initialization
- **Optimizer**: Optax Adam with JAX JIT compilation
- **Device**: PennyLane Lightning (CPU/GPU backends)
- **Basis Set**: STO-3G
- **Method**: DHF (built-in Hartree-Fock)

**Computational Workload:**
- Bond lengths scanned: 100 (from 0.35 to 3.0 Angstroms)
- VQE iterations per bond: 300 fixed (`main.py`), or at most 300 with early stopping (Optax versions)

## Setup

### Environment Configurations

This project provides multiple environment configurations in `configs/`:

1. **`environment.yml`** - Basic serial implementation (CPU only)
2. **`vqe-mpi.yml`** - CPU-based MPI parallelization 
3. **`vqe-gpu.yml`** - GPU-accelerated with CUDA support for HPC clusters
4. **`vqe-lightning-gpu.yml`** - Lightning GPU backend

### Local Setup (CPU)

For local testing and development:

```bash
# Create environment for serial/basic testing
conda env create -f configs/environment.yml
conda activate quantumvqe

# OR for MPI testing (requires OpenMPI)
conda env create -f configs/vqe-mpi.yml
conda activate vqe-openmpi
```

### HPC Cluster Setup (GPU)

For running on GPU-enabled HPC clusters:

```bash
# Create GPU-enabled environment
conda env create -f configs/vqe-gpu.yml
conda activate vqe-gpu

# Verify GPU access
python -c "import jax; print(f'GPUs available: {jax.devices()}')"
```

**Key Dependencies:**
- Python 3.12
- PennyLane 0.43.1 (quantum computing framework)
- JAX 0.6.2 (with CUDA 11.8 support for GPU version)
- PennyLane Catalyst 0.13.0 (JIT compilation)
- Optax 0.2.6 (optimization)
- OpenMPI + mpi4py (for distributed computing)
- NumPy, SciPy, Matplotlib (scientific computing)

## Running the Code

### Local Execution

**Serial implementation:**
```bash
# Quick test (5 bond lengths, 50 iterations)
python src/main_study/test_main.py

# Full run
python src/main_study/main.py
```

**JIT-compiled version with Optax** (recommended):
```bash
python src/main_study/vqe_serial_optax.py
```

**GPU-accelerated version**:
```bash
python src/main_study/vqe_gpu.py
```

**MPI parallel version** (requires OpenMPI):
```bash
# Run with 4 MPI processes
mpirun -np 4 python src/main_study/vqe_mpi.py
```

### Reproducibility

The serial study starts every optimization from zero-initialized parameters (`np.zeros`) on PennyLane Lightning's analytic backend, so there is no random number generator to seed. Repeated runs of the same configuration agree to within floating-point reduction order (~1e-6 Ha), set by the BLAS/OpenMP thread count rather than any RNG; fix the thread count (e.g. `OMP_NUM_THREADS=1`) for bit-identical results.

Convergence to FCI requires roughly 50 optimization steps. `main.py` prints a warning when `MAX_STEPS` is set below this floor (`CONVERGENCE_MIN_STEPS` in `vqe_params.py`); the energy returned at smaller budgets is meaningful but not converged.

### HPC Cluster Execution

Submit jobs using PBS scheduler:

```bash
# Submit serial baseline job
qsub pbs_scripts/run_serial.sh

# Submit GPU-accelerated job
qsub pbs_scripts/run_gpu.sh

# Submit MPI parallel job
qsub pbs_scripts/run_mpi_template.sh

# Run scaling study
qsub pbs_scripts/run_scaling_study.sh

# Run multi-GPU benchmark
qsub pbs_scripts/run_comprehensive_gpu_benchmark.sh

# Check job status
qstat -u $USER
```

**Results:** Output plots and data are saved in `results/` directory.

## Performance Results

Benchmarked on ERAU Vega HPC cluster with AMD EPYC 9654 (192 cores) and 4× NVIDIA H100 GPUs (80GB each).

### H2 scan, 100 bond lengths

| Implementation | Optimizer | Stopping rule | Start per bond | Iterations | Runtime | Time per iteration |
|---|---|---|---|---|---|---|
| `main.py` (CPU) | PennyLane Adam, lr 0.01 | fixed 300 | zeros | 30,000 | 593.95 s | 0.0198 s |
| `vqe_serial_optax.py` (CPU, Catalyst JIT) | Optax Adam, lr 0.05 | \|ΔE\| < 1e-8 or 300 | previous bond | 7,033 | 143.80 s | 0.0204 s |
| `vqe_gpu.py` (`lightning.gpu`) | Optax Adam, lr 0.05 | \|ΔE\| < 1e-8 or 300 | zeros | 11,345 | 164.91 s | 0.0145 s |

Time per iteration is the total runtime divided by the iteration count. It includes the per-bond Hamiltonian construction, so it does not isolate the cost of the compiled circuit. The 4.13× runtime difference between `main.py` and `vqe_serial_optax.py` tracks the 4.3× difference in iteration count.

### MPI runs

| Ranks | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Runtime (rank 0) | 8.45 s | 6.07 s | 5.48 s | 5.06 s | 5.04 s |

`vqe_mpi.py` records the elapsed time on rank 0 only, before the gather. The serial jobs ran with `ppn=1`. A balanced split of the serial per-bond times gives 73.5 s for rank 0 at 2 ranks, against 8.45 s measured. The MPI jobs and the serial jobs therefore ran with different resources, and the ratio of their runtimes is not a parallel speedup.

### CPU vs GPU scaling study (4 to 26 qubits)

Transverse-field Ising Hamiltonian (J = 1.0, h = 0.5), hardware-efficient ansatz with 2 layers.

| Qubits | CPU time | GPU time | CPU/GPU time ratio |
|---|---|---|---|
| 4 | 8.33 s | 0.79 s | 10.5 |
| 20 | 46.77 s | 1.08 s | 43.2 |
| 26 | 1425.06 s | 17.71 s | 80.5 |

The two paths use different software. The CPU path uses `lightning.qubit`, the JAX interface, `jax.jit`, Optax Adam, with JAX in its default 32-bit mode. The GPU path uses `lightning.gpu`, the autograd interface, adjoint differentiation, and PennyLane Adam. The ratio includes these differences and is not a hardware-only comparison.

### Multi-GPU runs (4× H100)

All three runs use a sum-of-Pauli-Z Hamiltonian, `lightning.gpu` with adjoint differentiation, and gradient descent. The two multi-problem runs use 1 MPI rank per GPU.

- Largest state vector on one H100: 29 qubits. 30 qubits failed with an out-of-memory error.
- 8 independent 22-qubit problems (15 iterations each), 2 per GPU: the longest per-GPU time is 8.04 s. The sum of the per-GPU times is 31.99 s. No single-GPU run of the same 8 problems was made.
- 12 independent 20-qubit problems, 3 per GPU: the longest per-GPU time is 12.27 s.

## Code Structure

```
QuantumVQE/
├── src/
│   ├── main_study/          # H2 molecule VQE implementations
│   │   ├── main.py              # Serial VQE (baseline)
│   │   ├── test_main.py         # Quick test version
│   │   ├── vqe_serial_optax.py  # JIT-compiled with Optax
│   │   ├── vqe_gpu.py           # GPU-accelerated
│   │   ├── vqe_mpi.py           # MPI parallel
│   │   ├── vqe_qjit.py          # Catalyst JIT
│   │   └── vqe_params.py        # Shared parameters
│   └── scaling_study/       # GPU scaling benchmarks
│       ├── run_scaling_study.py
│       ├── comprehensive_gpu_benchmark.py
│       └── hamiltonians.py
├── pbs_scripts/             # PBS job submission scripts
├── scripts/                 # Analysis and plotting scripts
├── results/                 # Output plots and data
│   ├── main_study/          # H2 VQE results
│   ├── scaling_study/       # CPU vs GPU scaling results
│   └── multi_gpu/           # Multi-GPU benchmark results
├── configs/                 # Conda environment files
│   ├── environment.yml
│   ├── vqe-mpi.yml
│   ├── vqe-gpu.yml
│   └── vqe-lightning-gpu.yml
├── deliverables/            # arXiv preprint source
└── logs/                    # HPC job logs
```

## License

Academic project for MA453 - High Performance Computing, Fall 2025

Ashton Steed and Rylan Malarchick, Embry-Riddle Aeronautical University
