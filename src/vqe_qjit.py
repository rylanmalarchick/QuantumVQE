import time
import pennylane as qml
from pennylane import qchem
from jax import numpy as jnp
import jax
import catalyst
import optax  # Import Optax
import matplotlib.pyplot as plt
from vqe_params import *

# --- Configuration ---
# Force jax to use cpu only
jax.config.update("jax_platform_name", "cpu")
jax.config.update("jax_enable_x64", True)



dev = qml.device("lightning.qubit", wires=QUBITS)



def get_h2_hamiltonian(bond_length):
    coordinates = jnp.array([[0.0, 0.0, -bond_length / 2], 
                             [0.0, 0.0, bond_length / 2]])
    molecule = qchem.Molecule(SYMBOLS, coordinates, unit=UNIT)
    H_obj, _ = qchem.molecular_hamiltonian(molecule, method=METHOD)
    # Return full object for mapping
    return H_obj

def get_static_molecular_data():
    # Use 1.4 A (near equilibrium) to ensure we get a comprehensive operator list
    H_obj = get_h2_hamiltonian(bond_length=DUMMY_BOND_LENGTH)
    _, static_ops = H_obj.terms()
    
    hf_state = qchem.hf_state(ELECTRONS, QUBITS)
    singles, doubles = qchem.excitations(ELECTRONS, QUBITS)
    doubles_wires = doubles[0]
    
    return static_ops, hf_state, doubles_wires, len(singles) + len(doubles)

def build_vqe_runner(static_ops, hf_state, doubles_wires, tol=1e-8):
    
    @qml.qnode(dev)
    def circuit(params, coeffs):
        H = qml.Hamiltonian(coeffs, static_ops)
        qml.BasisState(hf_state, wires=range(QUBITS))
        qml.DoubleExcitation(params[0], wires=doubles_wires)
        return qml.expval(H)

    @catalyst.qjit
    def optimize_step(params, coeffs):
        optimizer = optax.adam(learning_rate=STEP_SIZE)
        opt_state = optimizer.init(params)

        def cost_fn(p): return circuit(p, coeffs)
        grad_fn = catalyst.grad(cost_fn)

        # Initial Energy Calculation
        init_energy = cost_fn(params)
        
        # State: (params, opt_state, current_energy, prev_energy, step_count)
        init_carry = (params, opt_state, init_energy, init_energy + 100.0, 0)

        def cond_fn(carry):
            _, _, curr_E, prev_E, step = carry
            not_max_steps = step < MAX_STEPS
            not_converged = jnp.abs(curr_E - prev_E) > tol
            return jnp.logical_and(not_max_steps, not_converged)

        def body_fn(carry):
            curr_params, curr_opt_state, curr_E, _, step = carry
            
            grads = grad_fn(curr_params)
            updates, new_opt_state = optimizer.update(grads, curr_opt_state, curr_params)
            new_params = optax.apply_updates(curr_params, updates)
            
            new_energy = cost_fn(new_params)
            
            # Increment step count here
            return (new_params, new_opt_state, new_energy, curr_E, step + 1)

        final_carry = catalyst.while_loop(cond_fn)(body_fn)(init_carry)

        # Unpack the step count (index 4)
        final_params, _, final_energy, _, steps_taken = final_carry
        
        # Return it!
        return final_params, final_energy, steps_taken

    return optimize_step

def main():
    print("--- Starting H2 VQE Scan (Catalyst JIT + Optax) ---")
    
    static_ops, hf_state, doubles_wires, n_params = get_static_molecular_data()
    
    print("Building JIT-compiled VQE function...")
    vqe_runner = build_vqe_runner(static_ops, hf_state, doubles_wires)
    
    bond_lengths = jnp.linspace(START_DIST, END_DIST, NUM_POINTS)
    energies = []
    
    params = jnp.zeros(n_params)

    start_time = time.time()
    total_steps = 0
    print(f"Scanning {len(bond_lengths)} bond lengths...")
    
    for i, bl in enumerate(bond_lengths):
        bl_start = time.time()
        
        # 1. Get Hamiltonian Object
        H_obj = get_h2_hamiltonian(bl)
        
        aligned_coeffs, _ = H_obj.terms()
        
        # 3. Run VQE (Optax inside JIT)
        # Params is not wiped, so each leng starts with previous best as guess
        params, energy, steps = vqe_runner(params, aligned_coeffs)
        total_steps += steps
        
        energies.append(energy)
        
        print(f"Step {i+1:02d}: Bond Length={bl:.2f} A | Energy={energy:.6f} Ha | Time={time.time()-bl_start:.3f}s | Steps={steps}")

    total_time = time.time() - start_time
    
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE METRICS - JIT+OPTAX CODE")
    print("="*60)   
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Avg time per bond length: {total_time/len(bond_lengths):.2f} seconds")
    print(f"Avg time per VQE iteration: {total_time/total_steps:.4f} seconds")
    print(f"Circuit evaluations per second: {steps/total_time:.2f}")
    print("="*60)
    plt.figure(figsize=(10, 6))
    plt.plot(bond_lengths, energies, 'o-', linewidth=2, markersize=6, label="VQE (Optax)")
    
    
    
    plt.xlabel("Bond Length (Angstrom)", fontsize=12)
    plt.ylabel("Ground State Energy (Hartree)", fontsize=12)
    plt.title("H2 Potential Energy Surface (VQE + Optax)", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.savefig('vqe_results_optax.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: vqe_results_optax.png")

if __name__ == "__main__":
    main()