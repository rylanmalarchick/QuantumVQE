# Quantum VQE for H2 Molecule

Variational Quantum Eigensolver (VQE) implementation for computing the ground state energy of molecular hydrogen (H2) using PennyLane, optimized for HPC clusters with multi-GPU support.

## Overview

This project implements a quantum chemistry simulation using the Variational Quantum Eigensolver algorithm to compute the potential energy surface of the H2 molecule. We demonstrate comprehensive HPC parallelization achieving **117× total speedup** (593.95s → 5.04s) through JIT compilation, GPU acceleration, and MPI parallelization on the ERAU Vega HPC cluster featuring 4× NVIDIA H100 GPUs.

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
- Bond lengths scanned: 100 (from 0.1 to 3.0 Angstroms)
- VQE iterations per bond: 300
- Total circuit evaluations: 30,000

## Setup

### Environment Configurations

This project provides multiple environment configurations in `environment_configs/`:

1. **`environment.yml`** - Basic serial implementation (CPU only)
2. **`vqe-mpi.yml`** - CPU-based MPI parallelization 
3. **`vqe-gpu.yml`** - GPU-accelerated with CUDA support for HPC clusters
4. **`vqe-lightning-gpu.yml`** - Lightning GPU backend

### Local Setup (CPU)

For local testing and development:

```bash
# Create environment for serial/basic testing
conda env create -f environment_configs/environment.yml
conda activate quantumvqe

# OR for MPI testing (requires OpenMPI)
conda env create -f environment_configs/vqe-mpi.yml
conda activate vqe-openmpi
```

### HPC Cluster Setup (GPU)

For running on GPU-enabled HPC clusters:

```bash
# Create GPU-enabled environment
conda env create -f environment_configs/vqe-gpu.yml
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
python src/test_main.py

# Full run
python src/main.py
```

**JIT-compiled version with Optax** (recommended):
```bash
python src/vqe_serial_optax.py
```

**GPU-accelerated version**:
```bash
python src/vqe_gpu.py
```

**MPI parallel version** (requires OpenMPI):
```bash
# Run with 4 MPI processes
mpirun -np 4 python src/vqe_mpi.py
```

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

### Summary: 117× Total Speedup

| Implementation | Runtime | Speedup |
|----------------|---------|---------|
| Serial (PennyLane Adam) | 593.95s | 1.0× |
| Serial Optax+JIT | 143.80s | 4.13× |
| GPU (lightning.gpu) | 164.91s | 3.60× |
| MPI-32 (Optax+JIT) | 5.04s | **117.85×** |

### Four-Factor Speedup Decomposition

1. **Optimizer + JIT**: 4.13× (Optax + Catalyst JIT compilation)
2. **GPU Acceleration**: 3.60× at 4 qubits → 80.5× at 26 qubits
3. **MPI Parallelization**: 28.53× (embarrassingly parallel)
4. **Multi-GPU Scaling**: 3.98× with 99.4% efficiency across 4 H100s

### GPU Scaling Study (4-26 Qubits)

| Qubits | CPU Time | GPU Time | Speedup |
|--------|----------|----------|---------|
| 4 | 8.33s | 0.79s | 10.5× |
| 20 | 46.77s | 1.08s | 43.2× |
| 26 | 1425.06s | 17.71s | **80.5×** |

### Multi-GPU Performance

- **Maximum qubits**: 29 per H100 (8GB state vector, ~32GB with adjoint overhead)
- **Parallel efficiency**: 99.4% across 4 GPUs
- **Throughput**: ~1 problem/second at 20 qubits with 4 GPUs

## Code Structure

```
QuantumVQE/
├── src/
│   ├── main.py              # Serial VQE implementation (baseline)
│   ├── test_main.py         # Quick test version
│   ├── vqe_serial_optax.py  # JIT-compiled VQE with Optax
│   ├── vqe_gpu.py           # GPU-accelerated VQE
│   ├── vqe_mpi.py           # MPI parallel VQE
│   ├── vqe_qjit.py          # Catalyst JIT VQE
│   ├── vqe_params.py        # Shared configuration parameters
│   └── scaling_study/       # GPU scaling benchmarks
│       ├── run_scaling_study.py
│       ├── comprehensive_gpu_benchmark.py
│       └── hamiltonians.py
├── pbs_scripts/             # PBS job submission scripts
├── scripts/                 # Analysis and plotting scripts
├── configs/                 # Configuration files
├── results/                 # Output plots and data
│   ├── scaling_study/       # CPU vs GPU scaling results
│   └── multi_gpu/           # Multi-GPU benchmark results
├── deliverables/            # Project report and documentation
├── environment_configs/     # Conda environment files
│   ├── environment.yml
│   ├── vqe-mpi.yml
│   ├── vqe-gpu.yml
│   └── vqe-lightning-gpu.yml
└── logs/                    # HPC job logs
```

## HPC Optimization Strategies

This project demonstrates multiple parallelization approaches with measured results:

### 1. JIT Compilation + Optax Optimizer (4.13× speedup)
- Uses PennyLane Catalyst + JAX for just-in-time compilation
- Optax optimizer replaces PennyLane's built-in Adam
- Compiled gradient computation via catalyst.grad

### 2. GPU Acceleration (3.6× to 80.5× speedup)
- CUDA-enabled JAX on NVIDIA H100 GPUs
- `lightning.gpu` backend for quantum circuit simulation
- Speedup increases dramatically with qubit count

### 3. MPI Parallelism (28.53× speedup)
- Distributes bond length calculations across CPU cores
- Embarrassingly parallel with scatter-gather pattern
- Near-linear scaling up to 32 processes

### 4. Multi-GPU Scaling (3.98× speedup)
- Distributes workload across 4× H100 GPUs
- 99.4% parallel efficiency
- Single H100 limit: 29 qubits

## Key Findings

- **GPU wins at all scales**: 10× speedup at 4 qubits, 80× at 26 qubits
- **Algorithm matters**: Optimizer+JIT alone gives 4× improvement
- **Near-perfect multi-GPU scaling**: 99.4% efficiency with embarrassingly parallel workloads
- **Memory limits**: 29 qubits max per H100 due to adjoint differentiation overhead

## License

Academic project for MA453 - High Performance Computing, Fall 2025

Ashton Steed and Rylan Malarchick, Embry-Riddle Aeronautical University
