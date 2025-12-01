"""
Random Hamiltonian Generator for Scaling Study

Generates random Pauli Hamiltonians of arbitrary qubit count for benchmarking.
This avoids the complexity of molecular integrals while providing realistic
VQE workloads.
"""

import numpy as np
import pennylane as qml
from typing import Tuple, Optional
import random


def create_random_hamiltonian(
    n_qubits: int,
    n_terms: Optional[int] = None,
    seed: Optional[int] = None,
    locality: int = 2
) -> Tuple[qml.Hamiltonian, np.ndarray, list]:
    """
    Create a random Pauli Hamiltonian for benchmarking.
    
    Args:
        n_qubits: Number of qubits
        n_terms: Number of Pauli terms (default: n_qubits * 5)
        seed: Random seed for reproducibility
        locality: Maximum number of non-identity Paulis per term (default: 2)
                  Higher locality = more complex Hamiltonian
    
    Returns:
        Tuple of (Hamiltonian, coefficients array, list of Pauli strings)
    """
    if seed is not None:
        np.random.seed(seed)
        random.seed(seed)
    
    if n_terms is None:
        # Scale terms with qubits - more terms = harder problem
        n_terms = n_qubits * 5
    
    coeffs = []
    ops = []
    pauli_strings = []
    
    # Always include identity term (energy offset)
    coeffs.append(np.random.randn() * 0.5)
    ops.append(qml.Identity(0))
    pauli_strings.append("I" * n_qubits)
    
    # Generate random Pauli terms
    pauli_choices = ['I', 'X', 'Y', 'Z']
    
    for _ in range(n_terms - 1):
        # Random coefficient
        coeff = np.random.randn() * 0.5
        
        # Build Pauli string with controlled locality
        pauli_str = ['I'] * n_qubits
        
        # Choose random qubits to have non-identity Paulis
        n_active = random.randint(1, min(locality, n_qubits))
        active_qubits = random.sample(range(n_qubits), n_active)
        
        for q in active_qubits:
            pauli_str[q] = random.choice(['X', 'Y', 'Z'])
        
        pauli_str_joined = ''.join(pauli_str)
        
        # Skip if we already have this term
        if pauli_str_joined in pauli_strings:
            continue
        
        # Build the operator
        op_list = []
        for q, p in enumerate(pauli_str):
            if p == 'X':
                op_list.append(qml.PauliX(q))
            elif p == 'Y':
                op_list.append(qml.PauliY(q))
            elif p == 'Z':
                op_list.append(qml.PauliZ(q))
        
        if len(op_list) == 0:
            # All identity - skip
            continue
        elif len(op_list) == 1:
            op = op_list[0]
        else:
            # Tensor product of Paulis
            op = op_list[0]
            for o in op_list[1:]:
                op = op @ o
        
        coeffs.append(coeff)
        ops.append(op)
        pauli_strings.append(pauli_str_joined)
    
    coeffs = np.array(coeffs)
    H = qml.Hamiltonian(coeffs, ops)
    
    return H, coeffs, pauli_strings


def create_transverse_field_ising(n_qubits: int, J: float = 1.0, h: float = 0.5) -> qml.Hamiltonian:
    """
    Create a 1D Transverse Field Ising Model Hamiltonian.
    
    H = -J * sum_i(Z_i Z_{i+1}) - h * sum_i(X_i)
    
    This is a well-studied quantum many-body system.
    
    Args:
        n_qubits: Number of qubits (spins)
        J: Coupling strength
        h: Transverse field strength
    
    Returns:
        PennyLane Hamiltonian
    """
    coeffs = []
    ops = []
    
    # ZZ interactions (nearest neighbor)
    for i in range(n_qubits - 1):
        coeffs.append(-J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    
    # Transverse field (X terms)
    for i in range(n_qubits):
        coeffs.append(-h)
        ops.append(qml.PauliX(i))
    
    return qml.Hamiltonian(coeffs, ops)


def create_heisenberg_model(n_qubits: int, J: float = 1.0) -> qml.Hamiltonian:
    """
    Create a 1D Heisenberg XXX Model Hamiltonian.
    
    H = J * sum_i(X_i X_{i+1} + Y_i Y_{i+1} + Z_i Z_{i+1})
    
    Args:
        n_qubits: Number of qubits (spins)
        J: Coupling strength
    
    Returns:
        PennyLane Hamiltonian
    """
    coeffs = []
    ops = []
    
    for i in range(n_qubits - 1):
        # XX
        coeffs.append(J)
        ops.append(qml.PauliX(i) @ qml.PauliX(i + 1))
        # YY
        coeffs.append(J)
        ops.append(qml.PauliY(i) @ qml.PauliY(i + 1))
        # ZZ
        coeffs.append(J)
        ops.append(qml.PauliZ(i) @ qml.PauliZ(i + 1))
    
    return qml.Hamiltonian(coeffs, ops)


def estimate_memory_mb(n_qubits: int, include_gradients: bool = True) -> float:
    """
    Estimate memory usage for a given qubit count.
    
    Args:
        n_qubits: Number of qubits
        include_gradients: Include memory for gradient computation
    
    Returns:
        Estimated memory in MB
    """
    # State vector: 2^n complex128 (16 bytes each)
    state_vector_bytes = (2 ** n_qubits) * 16
    
    # For gradients, we need ~2-3x the state vector
    multiplier = 3.0 if include_gradients else 1.0
    
    total_bytes = state_vector_bytes * multiplier
    return total_bytes / (1024 ** 2)


def get_hamiltonian_info(H: qml.Hamiltonian) -> dict:
    """Get information about a Hamiltonian."""
    return {
        'n_terms': len(H.coeffs),
        'coeffs_mean': float(np.mean(np.abs(H.coeffs))),
        'coeffs_std': float(np.std(H.coeffs)),
    }


if __name__ == "__main__":
    # Test Hamiltonian generation
    print("Testing Hamiltonian generation...")
    
    for n_qubits in [4, 8, 12]:
        H, coeffs, paulis = create_random_hamiltonian(n_qubits, seed=42)
        mem = estimate_memory_mb(n_qubits)
        print(f"\n{n_qubits} qubits:")
        print(f"  Terms: {len(coeffs)}")
        print(f"  Est. memory: {mem:.2f} MB")
        print(f"  Sample terms: {paulis[:3]}")
    
    print("\n\nTesting Ising model...")
    H_ising = create_transverse_field_ising(8)
    print(f"8-qubit Ising: {len(H_ising.coeffs)} terms")
    
    print("\nTesting Heisenberg model...")
    H_heis = create_heisenberg_model(8)
    print(f"8-qubit Heisenberg: {len(H_heis.coeffs)} terms")
