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

The environment uses Micromamba (lightweight conda) with the following dependencies:
- Python 3.12
- PennyLane 0.43.1
- JAX 0.6.2
- NumPy, SciPy, Matplotlib
- PennyLane Catalyst (for future optimization)

**Install and activate environment:**
```bash
# Environment already installed in ./env/
# Run code using:
./bin/micromamba run -p ./env python3 src/main.py
```

## Running the Code

**Test run** (5 bond lengths, 50 iterations, ~2 seconds):
```bash
./bin/micromamba run -p ./env python3 src/test_main.py
```

**Full run** (40 bond lengths, 200 iterations, ~1 minute):
```bash
./bin/micromamba run -p ./env python3 src/main.py
```

Results are saved as PNG images in `results/`:
- `results/vqe_results.png` - Full potential energy surface
- `results/test_vqe_results.png` - Test run results

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
│   ├── main.py         # Full serial VQE implementation
│   └── test_main.py    # Quick test version
├── results/            # Output plots
├── configs/            # Configuration files
├── pbs_scripts/        # PBS job submission templates
├── environment.yml     # Conda environment specification
└── bin/micromamba      # Package manager
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

## Future Work: HPC Optimization

The current serial implementation will be optimized using:

1. **JAX JIT compilation** - Just-in-time compilation of cost function
2. **Multiprocessing** - Parallelize bond length calculations across CPU cores
3. **Ray distributed computing** - Scale across HPC cluster nodes
4. **PBS array jobs** - Distribute work across cluster using job scheduler

Expected performance improvements:
- JAX JIT: 2-5x speedup
- Multiprocessing: 4-8x on multi-core systems
- Cluster distribution: Linear scaling up to 40 nodes

## License

Academic project for MA453 - Fall 2025
