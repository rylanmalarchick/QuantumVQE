# Quantum VQE for H2 Molecule

Variational Quantum Eigensolver (VQE) code that computes the ground-state energy of molecular hydrogen (H2) with PennyLane.

## Overview

This project computes the potential energy surface of H2 with VQE. It runs the same scan in four versions: a serial PennyLane baseline, an Optax + Catalyst JIT version, a `lightning.gpu` version, and an MPI version. The runs used the ERAU Vega HPC cluster (4× NVIDIA H100). The write-up is arXiv:2601.09951. Version 2 of the preprint corrects the speedup claims of version 1.

### The Model

VQE finds the ground-state energy. It optimizes a parameterized quantum circuit (the ansatz) to minimize the expectation value of the molecular Hamiltonian:

$$E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle$$

Here $|\psi(\theta)\rangle$ is the state that the ansatz prepares, and $H$ is the molecular Hamiltonian.

**System Details:**
- **Molecule**: H2
- **Qubits**: 4 (the scaling study goes up to 26 qubits)
- **Ansatz**: DoubleExcitation gate on the Hartree-Fock state
- **Optimizer**: PennyLane Adam (`main.py`) or Optax Adam (the other versions)
- **Device**: PennyLane Lightning (`lightning.qubit` on CPU, `lightning.gpu` on GPU)
- **Basis Set**: STO-3G
- **Method**: DHF (PennyLane's built-in Hartree-Fock)

**Computational Workload:**
- Bond lengths: 100, from 0.35 to 3.0 Angstrom
- VQE iterations per bond length: 300 fixed (`main.py`), or at most 300 with early stopping (Optax versions)

## Setup

### Environment Configurations

The `configs/` directory has four conda environments:

1. **`environment.yml`**: serial version, CPU only
2. **`vqe-mpi.yml`**: MPI version, CPU
3. **`vqe-gpu.yml`**: Catalyst and JAX with CUDA, for the cluster
4. **`vqe-lightning-gpu.yml`**: `lightning.gpu` backend

### Local Setup (CPU)

```bash
# Serial version
conda env create -f configs/environment.yml
conda activate quantumvqe

# MPI version (needs OpenMPI)
conda env create -f configs/vqe-mpi.yml
conda activate vqe-openmpi
```

### HPC Cluster Setup (GPU)

```bash
conda env create -f configs/vqe-gpu.yml
conda activate vqe-gpu

# Check that JAX sees the GPU
python -c "import jax; print(f'GPUs available: {jax.devices()}')"
```

**Pinned versions** (`vqe-gpu.yml` and `vqe-mpi.yml`):
- Python 3.12
- PennyLane 0.43.1
- PennyLane Catalyst 0.13.0
- JAX 0.6.2 (CUDA 11.8 build in `vqe-gpu.yml`)
- Optax 0.2.6
- OpenMPI and mpi4py

## Running the Code

### Local Execution

**Serial version:**
```bash
# Quick test (5 bond lengths, 50 iterations)
python src/main_study/test_main.py

# Full run
python src/main_study/main.py
```

**Optax + Catalyst JIT version:**
```bash
python src/main_study/vqe_serial_optax.py
```

**GPU version:**
```bash
python src/main_study/vqe_gpu.py
```

**MPI version** (needs OpenMPI):
```bash
mpirun -np 4 python src/main_study/vqe_mpi.py
```

### Reproducibility

Every optimization in `main.py` starts from zero parameters (`np.zeros`) on an analytic Lightning backend. The H2 scan uses no random numbers.

`main.py` prints a warning when `MAX_STEPS` is below `CONVERGENCE_MIN_STEPS` (50, set in `vqe_params.py`).

### HPC Cluster Execution

Submit the jobs with the PBS scheduler:

```bash
qsub pbs_scripts/run_serial.sh                     # serial baseline
qsub pbs_scripts/run_gpu.sh                        # GPU node job
qsub pbs_scripts/run_mpi_template.sh               # MPI version
qsub pbs_scripts/run_scaling_study.sh              # CPU vs GPU scaling study
qsub pbs_scripts/run_comprehensive_gpu_benchmark.sh  # multi-GPU benchmark
qstat -u $USER                                     # job status
```

The jobs write plots and data to `results/`.

## Performance Results

All runs used the ERAU Vega HPC cluster: AMD EPYC 9654 (192 cores) and 4× NVIDIA H100 GPUs (80 GB each).

### H2 scan, 100 bond lengths

| Implementation | Optimizer | Stopping rule | Start per bond | Iterations | Runtime | Time per iteration |
|---|---|---|---|---|---|---|
| `main.py` (CPU) | PennyLane Adam, lr 0.01 | fixed 300 | zeros | 30,000 | 593.95 s | 0.0198 s |
| `vqe_serial_optax.py` (CPU, Catalyst JIT) | Optax Adam, lr 0.05 | \|ΔE\| < 1e-8 or 300 | previous bond | 7,033 | 143.80 s | 0.0204 s |
| `vqe_gpu.py` (`lightning.gpu`) | Optax Adam, lr 0.05 | \|ΔE\| < 1e-8 or 300 | zeros | 11,345 | 164.91 s | 0.0145 s |

Time per iteration is the total runtime divided by the iteration count. It includes the Hamiltonian construction for each bond length, so it does not measure the compiled circuit alone. `vqe_serial_optax.py` has a 4.13× lower runtime than `main.py` and a 4.3× lower iteration count.

### MPI runs

| Ranks | 2 | 4 | 8 | 16 | 32 |
|---|---|---|---|---|---|
| Runtime (rank 0) | 8.45 s | 6.07 s | 5.48 s | 5.06 s | 5.04 s |

`vqe_mpi.py` records the elapsed time on rank 0 only, before the gather. The serial jobs ran with `ppn=1`. Split the serial per-bond times the same way as `vqe_mpi.py`: rank 0 gets 73.5 s of work at 2 ranks, against 8.45 s measured. The MPI jobs and the serial jobs therefore had different resources. The ratio of their runtimes is not a parallel speedup.

### CPU vs GPU scaling study (4 to 26 qubits)

Transverse-field Ising Hamiltonian (J = 1.0, h = 0.5), hardware-efficient ansatz with 2 layers.

| Qubits | CPU time | GPU time | CPU/GPU time ratio |
|---|---|---|---|
| 4 | 8.33 s | 0.79 s | 10.5 |
| 20 | 46.77 s | 1.08 s | 43.2 |
| 26 | 1425.06 s | 17.71 s | 80.5 |

The two paths use different software:

- CPU: `lightning.qubit`, the JAX interface, `jax.jit`, Optax Adam, and JAX in its default 32-bit mode.
- GPU: `lightning.gpu`, the autograd interface, adjoint differentiation, and PennyLane Adam.

The ratio includes these differences. It does not compare the hardware alone.

### Multi-GPU runs (4× H100)

All three runs use a sum-of-Pauli-Z Hamiltonian, `lightning.gpu` with adjoint differentiation, and gradient descent. The two multi-problem runs use one MPI rank per GPU.

- Largest state vector on one H100: 29 qubits. The 30-qubit run failed with an out-of-memory error.
- 8 independent 22-qubit problems (15 iterations each), 2 per GPU: the longest per-GPU time is 8.04 s, and the sum of the per-GPU times is 31.99 s. There is no single-GPU run of these 8 problems.
- 12 independent 20-qubit problems, 3 per GPU: the longest per-GPU time is 12.27 s.

## Code Structure

```
QuantumVQE/
├── src/
│   ├── main_study/          # H2 VQE versions
│   │   ├── main.py              # Serial baseline
│   │   ├── test_main.py         # Quick test
│   │   ├── vqe_serial_optax.py  # Optax + Catalyst JIT
│   │   ├── vqe_gpu.py           # lightning.gpu
│   │   ├── vqe_mpi.py           # MPI
│   │   ├── vqe_qjit.py          # Catalyst JIT
│   │   └── vqe_params.py        # Shared parameters
│   └── scaling_study/       # CPU vs GPU and multi-GPU benchmarks
│       ├── run_scaling_study.py
│       ├── comprehensive_gpu_benchmark.py
│       └── hamiltonians.py
├── pbs_scripts/             # PBS job scripts
├── scripts/                 # Plot scripts for the preprint figures
├── results/                 # Measured data and plots
│   ├── main_study/          # H2 scan
│   ├── scaling_study/       # CPU vs GPU scaling study
│   └── multi_gpu/           # Multi-GPU benchmark
├── configs/                 # Conda environment files
├── deliverables/            # arXiv preprint source
└── logs/                    # HPC job logs
```

## License

Academic project for MA453 - High Performance Computing, Fall 2025

Ashton Steed and Rylan Malarchick, Embry-Riddle Aeronautical University
