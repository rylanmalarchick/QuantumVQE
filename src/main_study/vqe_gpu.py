"""
GPU-Accelerated VQE Implementation (lightning.gpu + Optax)

This script uses PennyLane's lightning.gpu device for true GPU acceleration.
Used to benchmark actual GPU speedup over CPU implementations.

Key differences from other implementations:
- vs main.py: Uses Optax optimizer + lightning.gpu device (not PennyLane Adam)
- vs vqe_serial_optax.py: Uses lightning.gpu instead of lightning.qubit
- vs vqe_qjit.py: Uses lightning.gpu, NO Catalyst JIT (different acceleration path)
- vs vqe_mpi.py: Single GPU, no MPI parallelization

Requirements:
- PennyLane-Lightning-GPU package (pennylane-lightning-gpu)
- CUDA-compatible GPU
- Environment: vqe-lightning-gpu (see vqe-lightning-gpu.yml)

Note: This script does NOT use Catalyst JIT compilation.
GPU acceleration comes from the lightning.gpu device backend.
"""

import time
import pennylane as qml
from pennylane import qchem
from pennylane import numpy as np
import optax
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for cluster
import matplotlib.pyplot as plt
from vqe_params import *

# --- Device Setup ---
# Use lightning.gpu for GPU-accelerated quantum simulation
# Falls back to lightning.qubit if GPU not available
try:
    dev = qml.device("lightning.gpu", wires=QUBITS)
    DEVICE_NAME = "lightning.gpu"
    print(f"SUCCESS: Using device: lightning.gpu (GPU-accelerated)")
except Exception as e:
    print(f"Warning: lightning.gpu not available ({e})")
    print("Falling back to lightning.qubit (CPU)")
    dev = qml.device("lightning.qubit", wires=QUBITS)
    DEVICE_NAME = "lightning.qubit"

# --- Molecular Setup ---
# Get static molecular data (doesn't change with bond length)
hf_state = qchem.hf_state(ELECTRONS, QUBITS)
singles, doubles = qchem.excitations(ELECTRONS, QUBITS)
n_params = len(singles) + len(doubles)

# Define the ansatz function
def ansatz(params):
    """The quantum circuit template (ansatz)."""
    qml.BasisState(hf_state, wires=range(QUBITS))
    qml.DoubleExcitation(params[0], wires=range(QUBITS))

# Global Hamiltonian (will be set per bond length)
H = None

@qml.qnode(dev)
def cost_fn(params):
    """Cost function: expectation value of Hamiltonian."""
    ansatz(params)
    return qml.expval(H)


def run_vqe_optax(params_init, max_steps=MAX_STEPS, tol=1e-8):
    """
    Run VQE optimization using Optax Adam optimizer.
    
    This is a standard optimization loop (no JIT compilation).
    GPU acceleration comes from the lightning.gpu device.
    """
    optimizer = optax.adam(learning_rate=STEP_SIZE)
    params = np.array(params_init, requires_grad=True)
    opt_state = optimizer.init(params)
    
    prev_energy = float('inf')
    
    for step in range(max_steps):
        # Compute gradient using PennyLane autodiff
        grad_fn = qml.grad(cost_fn)
        grads = grad_fn(params)
        
        # Optax update
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        params = np.array(params, requires_grad=True)  # Maintain grad tracking
        
        # Check convergence
        current_energy = cost_fn(params)
        if abs(current_energy - prev_energy) < tol:
            return params, current_energy, step + 1
        prev_energy = current_energy
    
    return params, cost_fn(params), max_steps


def main():
    global H
    
    print("="*60)
    print("GPU-ACCELERATED VQE (lightning.gpu + Optax)")
    print("="*60)
    print(f"Device: {DEVICE_NAME}")
    print("This benchmark measures true GPU acceleration")
    print("Note: No Catalyst JIT - GPU accel from lightning.gpu device")
    print("="*60)
    
    bond_lengths = np.linspace(START_DIST, END_DIST, NUM_POINTS)
    energies = np.zeros_like(bond_lengths)
    
    print(f"Bond lengths to scan: {len(bond_lengths)}")
    print(f"Max VQE iterations per bond length: {MAX_STEPS}")
    print("="*60)
    
    start_time = time.time()
    total_steps = 0
    
    for i, bl in enumerate(bond_lengths):
        bl_start = time.time()
        
        # Generate molecular Hamiltonian for this bond length
        coordinates = np.array([[0.0, 0.0, -bl / 2], [0.0, 0.0, bl / 2]])
        hydrogen = qchem.Molecule(SYMBOLS, coordinates, unit=UNIT)
        H, _ = qchem.molecular_hamiltonian(hydrogen, method=METHOD)
        
        # Initialize parameters (fresh start for each bond length)
        params_init = np.zeros(n_params)
        
        # Run VQE optimization
        params, energy, steps = run_vqe_optax(params_init)
        total_steps += steps
        
        energies[i] = energy
        
        print(f"Step {i+1:03d}/{len(bond_lengths)}: Bond={bl:.3f} A | "
              f"Energy={energy:.6f} Ha | Time={time.time()-bl_start:.3f}s | "
              f"Iterations={steps}")
    
    total_time = time.time() - start_time
    
    # Performance Summary
    print("\n" + "="*60)
    print(f"PERFORMANCE METRICS - GPU ({DEVICE_NAME} + Optax)")
    print("="*60)
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Avg time per bond length: {total_time/len(bond_lengths):.2f} seconds")
    print(f"Total VQE iterations: {total_steps}")
    print(f"Avg time per VQE iteration: {total_time/total_steps:.4f} seconds")
    print("="*60)
    
    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(bond_lengths, energies, 'o-', linewidth=2, markersize=6, 
             label=f"VQE ({DEVICE_NAME})")
    plt.xlabel("Bond Length (Angstrom)", fontsize=12)
    plt.ylabel("Ground State Energy (Hartree)", fontsize=12)
    plt.title(f"H2 Potential Energy Surface (VQE + {DEVICE_NAME})", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('vqe_results_gpu.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: vqe_results_gpu.png")


if __name__ == "__main__":
    main()
