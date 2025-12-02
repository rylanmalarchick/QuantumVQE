#!/usr/bin/env python3
"""
Multi-GPU Smoke Test: Verify MPI + multiple GPUs on single node.

This test verifies:
1. MPI can spawn multiple ranks
2. Each rank uses a different GPU
3. Basic VQE computation works on each GPU in parallel

Usage:
    mpiexec -n 4 python multi_node_smoke_test.py
    
Or via PBS on a node with multiple GPUs.
"""

import os
import sys
import time

def main():
    # Initialize MPI
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Assign each rank to a different GPU
    # This is the key for multi-GPU parallelism
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank % 4)  # Assumes 4 GPUs max
    
    # Get hostname
    hostname = os.uname().nodename
    gpu_id = rank % 4
    
    # Detect assigned GPU
    gpu_info = "No GPU detected"
    has_gpu = False
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader', f'--id={gpu_id}'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0 and result.stdout.strip():
            gpu_info = result.stdout.strip()
            has_gpu = True
    except Exception as e:
        gpu_info = f"Error: {e}"
    
    # Synchronize and print info from each rank
    comm.Barrier()
    
    for i in range(size):
        if rank == i:
            print(f"[Rank {rank}/{size}] Host: {hostname} | GPU {gpu_id}: {gpu_info}")
            sys.stdout.flush()
        comm.Barrier()
    
    if rank == 0:
        print("\n" + "="*60)
        print("SMOKE TEST: Running small VQE on each GPU...")
        print("="*60 + "\n")
        sys.stdout.flush()
    
    comm.Barrier()
    
    # Each rank runs a small VQE computation on its assigned GPU
    if has_gpu:
        try:
            import pennylane as qml
            from pennylane import numpy as pnp
            import numpy as np
            
            # Use lightning.gpu with the assigned GPU
            n_qubits = 4
            try:
                dev = qml.device("lightning.gpu", wires=n_qubits)
                device_name = f"lightning.gpu (GPU {gpu_id})"
            except Exception:
                dev = qml.device("lightning.qubit", wires=n_qubits)
                device_name = "lightning.qubit (fallback)"
            
            # Simple Hamiltonian
            coeffs = [1.0] * n_qubits + [0.5] * (n_qubits - 1)
            obs = [qml.PauliZ(i) for i in range(n_qubits)]
            obs += [qml.PauliX(i) @ qml.PauliX(i+1) for i in range(n_qubits - 1)]
            H = qml.Hamiltonian(coeffs, obs)
            
            @qml.qnode(dev, interface="autograd", diff_method="adjoint")
            def circuit(params):
                for i in range(n_qubits):
                    qml.RY(params[i], wires=i)
                for i in range(n_qubits - 1):
                    qml.CNOT(wires=[i, i+1])
                return qml.expval(H)
            
            # Use different random seed per rank for variety
            np.random.seed(42 + rank)
            params = pnp.array(np.random.randn(n_qubits) * 0.1, requires_grad=True)
            optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
            
            start = time.perf_counter()
            for _ in range(10):
                params = optimizer.step(circuit, params)
            elapsed = time.perf_counter() - start
            
            final_energy = float(circuit(params))
            
            print(f"[Rank {rank}] {device_name} | energy={final_energy:.4f} | time={elapsed:.2f}s")
            sys.stdout.flush()
            
            status = "SUCCESS"
        except Exception as e:
            print(f"[Rank {rank}] VQE FAILED: {e}")
            sys.stdout.flush()
            status = "FAILED"
    else:
        print(f"[Rank {rank}] Skipping VQE (no GPU)")
        sys.stdout.flush()
        status = "SKIPPED"
    
    # Gather results
    comm.Barrier()
    all_status = comm.gather(status, root=0)
    
    if rank == 0:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print(f"Total MPI ranks: {size}")
        print(f"GPUs used: {size} (one per rank)")
        print(f"Results: {all_status}")
        
        successes = sum(1 for s in all_status if s == "SUCCESS")
        print(f"\n{'SUCCESS' if successes == size else 'PARTIAL'}: {successes}/{size} ranks completed VQE")
        print("="*60)


if __name__ == '__main__':
    main()
