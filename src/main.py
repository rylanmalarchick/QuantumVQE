import pennylane as qml
from pennylane import numpy as np
from pennylane import qchem
import matplotlib.pyplot as plt


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
    # Define the trial bond length
    bond_lengths = np.linspace(0.1, 3, num=40) # in angstroms
    energies = np.zeros_like(bond_lengths)
    
    
    # Generate the molecule and Hamiltonian using qchem
    
    
    # This will be done for every trial bond length
    for i, bl in enumerate(bond_lengths):
        print(f"Calculating energy for bond length: {bl:.2f} Angstrom")
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
        for n in range(200): # 200 optimization steps
            params, _ = opt.step_and_cost(cost_fn, params)
            
        # 4. Store the converged energy
        final_energy = cost_fn(params)
        energies[i] = final_energy
        
    # --- 6. Plotting ---
    print("Scan complete. Plotting results.")
    plt.plot(bond_lengths, energies, 'o-')
    plt.xlabel("Bond Length (Angstrom)")
    plt.ylabel("Ground State Energy (Hartree)")
    plt.title("H2 Potential Energy Surface (VQE)")
    plt.show()
    
  

if __name__ == "__main__":
    main()
    
    
    
    
    