import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import time


#Define QML device  
dev = qml.device("lightning.qubit", wires=4)
#Initialize Molecule parameters and Hamiltonian in global scope
H = None
qubits = 4
symbols = ["H", "H"]
electrons = 2 

# Define Ansatz, the quantum circuit representing our molecule
# hf state is required for the ansatz, doesnt change with distance
hf_state = qchem.hf_state(electrons, qubits)
# Get total number of parameters needed
singles, doubles = qchem.excitations(electrons, qubits)
n_params = len(singles) + len(doubles)

# Define the ansatz function
def ansatz(params):
    """The quantum circuit template (ansatz)."""
    qml.BasisState(hf_state, wires=range(qubits))
    
    #Double Excitation for H2 is good, more general is allsinglesdoubles. 
    #Using Doubleexcitation, params has only one parameter
    
    #qml.AllSinglesDoubles(params, wires=range(qubits), hf_state=hf_state,  doubles=doubles)
    qml.DoubleExcitation(params[0], wires=range(qubits))
    
#Define Cost Function. 
# This applies the ansatz 
@qml.qnode(dev)
def cost_fn(params):
    ansatz(params)
    return qml.expval(H)

# Get the Hartree-Fock state (our starting point)
hf_state = qchem.hf_state(electrons, 4)

    

def main():
    #Ensure that Global Hamiltonian is used
    global H
    
    # SMALL TEST: Only 5 bond lengths, 50 iterations
    bond_lengths = np.linspace(0.5, 2.5, num=5) # in angstroms
    energies = np.zeros_like(bond_lengths)
    
    print("="*60)
    print("SMALL TEST RUN - Serial VQE Code")
    print("="*60)
    print(f"Bond lengths to test: {len(bond_lengths)}")
    print(f"VQE iterations per bond length: 50")
    print(f"Total circuit evaluations: {len(bond_lengths) * 50}")
    print("="*60)
    
    start_time = time.time()
    
    # Generate the molecule and Hamiltonian using qchem
    
    
    # This will be done for every trial bond length
    for i, bl in enumerate(bond_lengths):
        bl_start = time.time()
        print(f"\n[{i+1}/{len(bond_lengths)}] Bond length: {bl:.2f} Angstrom")
        
        #Generate molecule, hamiltonian, whatever else. This will be looped
        coordinates = np.array([[0.0, 0.0, -bl / 2], [0.0,0.0, bl/2]])
        #Define the molecule as a qchem object
        # Basis chosen might be significant, default is sto-3g
        hydrogen = qchem.Molecule(symbols, coordinates, unit="angstrom")
        #use qchem to generate the hamiltonian from this molecule
        # Dhf method is built in Hartee-Fock solver
        H, _ = qchem.molecular_hamiltonian(hydrogen, method="dhf")
        
        # Optimize paramters with VQE
        # Start from fresh parameters
        params = np.zeros(n_params, requires_grad=True)
        opt = qml.AdamOptimizer(stepsize=0.01)
        
    # Inner loop: The VQE optimization
        for n in range(50): # 50 optimization steps for test
            params, _ = opt.step_and_cost(cost_fn, params)
            
        # 4. Store the converged energy
        final_energy = cost_fn(params)
        energies[i] = final_energy
        
        bl_elapsed = time.time() - bl_start
        print(f"    Final energy: {final_energy:.6f} Ha")
        print(f"    Time: {bl_elapsed:.2f} seconds")
        
    total_time = time.time() - start_time
    
    # --- 6. Plotting ---
    print("\n" + "="*60)
    print("Scan complete. Generating plot...")
    plt.figure(figsize=(10, 6))
    plt.plot(bond_lengths, energies, 'o-', linewidth=2, markersize=8)
    plt.xlabel("Bond Length (Angstrom)", fontsize=12)
    plt.ylabel("Ground State Energy (Hartree)", fontsize=12)
    plt.title("H2 Potential Energy Surface (VQE) - Test Run", fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.savefig('test_vqe_results.png', dpi=150, bbox_inches='tight')
    print("Plot saved as: test_vqe_results.png")
    
    print("\n" + "="*60)
    print("PERFORMANCE METRICS - SMALL TEST")
    print("="*60)
    print(f"Total runtime: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print(f"Avg time per bond length: {total_time/len(bond_lengths):.2f} seconds")
    print(f"Avg time per VQE iteration: {total_time/(len(bond_lengths)*50):.4f} seconds")
    print(f"Circuit evaluations per second: {(len(bond_lengths)*50)/total_time:.2f}")
    print("="*60)
    
    # Estimate full run
    full_circuits = 40 * 200
    estimated_full_time = (total_time / (len(bond_lengths)*50)) * full_circuits
    print(f"\nEstimated time for full run (40 bonds, 200 iters): {estimated_full_time/60:.2f} minutes")
    print("="*60)
  

if __name__ == "__main__":
    main()
