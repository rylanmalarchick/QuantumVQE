# Quantum VQE for H2 Molecule

Variational Quantum Eigensolver (VQE) implementation for computing the ground state energy of molecular hydrogen (H2) using PennyLane.

## Overview

This project implements a quantum chemistry simulation using the Variational Quantum Eigensolver algorithm to compute the potential energy surface of the H2 molecule across different bond lengths.

### The Model

The VQE algorithm finds the ground state energy by optimizing a parameterized quantum circuit (ansatz) to minimize the expectation value of the molecular Hamiltonian:

$$E(\theta) = \langle \psi(\theta) | H | \psi(\theta) \rangle$$

where $|\psi(\theta)\rangle$ is the quantum state prepared by our ansatz and $H$ is the molecular Hamiltonian.

**System Details:**
- **Molecule**: H2 (hydrogen dimer)
- **Qubits**: 4
- **Ansatz**: DoubleExcitation gate with Hartree-Fock initialization
- **Optimizer**: Adam (stepsize = 0.01)
- **Device**: PennyLane Lightning (CPU simulator)
- **Basis Set**: STO-3G
- **Method**: DHF (built-in Hartree-Fock)

**Computational Workload:**
- Bond lengths scanned: 40 (from 0.1 to 3.0 Angstroms)
- VQE iterations per bond: 200
- Total circuit evaluations: 8,000

## Setup

### Environment Configurations

This project provides multiple environment configurations for different use cases:

1. **`environment.yml`** - Basic serial implementation (CPU only)
2. **`vqe-mpi.yml`** - CPU-based MPI parallelization 
3. **`vqe-gpu.yml`** - GPU-accelerated with CUDA support for HPC clusters

### Local Setup (CPU)

For local testing and development:

```bash
# Create environment for serial/basic testing
conda env create -f environment.yml
conda activate quantumvqe

# OR for MPI testing (requires OpenMPI)
conda env create -f vqe-mpi.yml
conda activate vqe-openmpi
```

### HPC Cluster Setup (GPU)

For running on GPU-enabled HPC clusters (e.g., vegaln1.erau.edu):

```bash
# SSH into cluster
ssh malarchr@vegaln1.erau.edu

# Clone repository or transfer code
# git clone <repo-url> or scp -r local/path user@host:remote/path

# Create GPU-enabled environment
conda env create -f vqe-gpu.yml
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
# Quick test (5 bond lengths, 50 iterations, ~2 seconds)
python src/test_main.py

# Full run (40 bond lengths, 200 iterations, ~1 minute)
python src/main.py
```

**JIT-compiled version** (requires Catalyst):
```bash
python src/vqe_qjit.py
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
qsub pbs_scripts/run_serial.pbs

# Submit GPU-accelerated job
qsub pbs_scripts/run_gpu.pbs

# Submit MPI parallel job (multiple nodes)
qsub pbs_scripts/run_mpi.pbs

# Check job status
qstat -u $USER

# View output logs
cat output_*.log
```

**Results:** Output plots are saved in `results/` directory:
- `vqe_results.png` - Serial potential energy surface
- `vqe_results_optax.png` - JIT-compiled version
- `vqe_results_mpi.png` - MPI parallel version

## Performance Baseline

Serial implementation performance metrics:

| Metric | Value |
|--------|-------|
| Total Runtime | 50.64 seconds (0.84 minutes) |
| Time per Bond Length | 1.27 seconds |
| Time per VQE Iteration | 0.0063 seconds (6.3 ms) |
| Circuit Evaluations/sec | 157.98 |
| Total Circuit Evaluations | 8,000 |

## Code Structure

```
QuantumVQE/
├── src/
│   ├── main.py           # Serial VQE implementation (baseline)
│   ├── test_main.py      # Quick test version
│   ├── vqe_qjit.py       # JIT-compiled VQE with Catalyst
│   ├── vqe_mpi.py        # MPI parallel VQE
│   └── vqe_params.py     # Shared configuration parameters
├── pbs_scripts/          # PBS job submission scripts
├── configs/              # Configuration files
├── results/              # Output plots and data
├── deliverables/         # Project report and documentation
├── environment.yml       # Basic CPU environment
├── vqe-mpi.yml          # MPI-enabled CPU environment
└── vqe-gpu.yml          # GPU-accelerated environment (HPC)
```

## Algorithm Overview

The VQE algorithm follows this structure:

1. **Initialize**: Start with Hartree-Fock state
2. **For each bond length**:
   - Generate molecular Hamiltonian for H2 at that distance
   - Initialize variational parameters
   - **Optimize** (200 iterations):
     - Prepare quantum state with ansatz
     - Measure energy expectation value
     - Update parameters using Adam optimizer
   - Store converged energy
3. **Plot** potential energy surface

The outer loop over bond lengths is embarrassingly parallel - each bond length calculation is completely independent.

## HPC Optimization Strategies

This project demonstrates multiple parallelization approaches:

### 1. JIT Compilation (Phase 1)
- Uses PennyLane Catalyst + JAX for just-in-time compilation
- Optimizes quantum circuit execution
- Expected: 2-5x speedup over serial baseline

### 2. GPU Acceleration (Phase 1)
- CUDA-enabled JAX on HPC GPU nodes
- Leverages GPU tensor operations
- Particularly effective for circuit simulations

### 3. MPI Parallelism (Phase 2)
- Distributes bond length calculations across multiple nodes
- Uses mpi4py for process communication
- Expected: Near-linear scaling up to 40 processes

### 4. Shared-Memory Parallelism (Phase 3)
- Python multiprocessing for single-node parallelization
- Distributes work across CPU cores
- Expected: 4-8x speedup on multi-core systems

## Performance Comparison

Target metrics to demonstrate HPC benefits:
- **Baseline (Serial)**: 50.64s for 40 bond lengths
- **JIT + GPU**: Target 2-5x speedup
- **MPI (8 processes)**: Target 6-8x speedup
- **MPI (16 processes)**: Target 10-15x speedup

These results will be documented in the project report with speedup plots and efficiency analysis.

## License

Academic project for MA453 - Fall 2025
