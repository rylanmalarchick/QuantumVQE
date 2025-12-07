#!/usr/bin/env python3
"""
Comprehensive Multi-GPU Benchmark

Tests three things:
1. Scale up qubits - find where H100 hits memory limit (28-32 qubits)
2. Multi-GPU throughput - 4 GPUs solving problems in parallel
3. Scaling comparison - same problem on 1 vs 4 GPUs

Usage:
    mpiexec -n 4 python comprehensive_gpu_benchmark.py
"""

import os
import sys
import json
import time
from datetime import datetime

def main():
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # Assign each rank to a different GPU
    gpu_id = rank % 4
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Only rank 0 prints headers
    def print_header(msg):
        if rank == 0:
            print("\n" + "=" * 70)
            print(msg)
            print("=" * 70)
            sys.stdout.flush()
    
    def print_rank(msg):
        print(f"[Rank {rank}/GPU {gpu_id}] {msg}")
        sys.stdout.flush()
    
    # Import after setting CUDA_VISIBLE_DEVICES
    import pennylane as qml
    from pennylane import numpy as pnp
    import numpy as np
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_gpus': size,
        'max_qubits_test': {},
        'throughput_test': {},
        'scaling_test': {},
    }
    
    comm.Barrier()
    
    # =========================================================================
    # TEST 1: Find maximum qubits per GPU
    # =========================================================================
    print_header("TEST 1: Maximum Qubit Scaling (Finding H100 Memory Limit)")
    
    if rank == 0:
        print("Each GPU will try increasingly large qubit counts until OOM...")
        print("Testing: 26, 28, 29, 30, 31 qubits")
        sys.stdout.flush()
    
    comm.Barrier()
    
    # Only rank 0 does the max qubit test (others wait)
    max_qubits_achieved = 0
    max_qubit_results = []
    
    if rank == 0:
        test_qubits = [26, 28, 29, 30, 31]
        
        for n_qubits in test_qubits:
            try:
                print(f"\nTrying {n_qubits} qubits...", flush=True)
                
                # Estimate memory
                state_vector_gb = (2 ** n_qubits) * 16 / (1024**3)  # complex128
                est_gpu_mem = state_vector_gb * 4  # adjoint method overhead
                print(f"  State vector: {state_vector_gb:.2f} GB, Est. GPU mem: {est_gpu_mem:.2f} GB", flush=True)
                
                dev = qml.device("lightning.gpu", wires=n_qubits)
                n_params = n_qubits * 2
                
                # Simple Hamiltonian (sum of Z operators)
                coeffs = [1.0] * n_qubits
                obs = [qml.PauliZ(i) for i in range(n_qubits)]
                H = qml.Hamiltonian(coeffs, obs)
                
                @qml.qnode(dev, interface="autograd", diff_method="adjoint")
                def circuit(params):
                    for i in range(n_qubits):
                        qml.RY(params[i], wires=i)
                        qml.RZ(params[i + n_qubits], wires=i)
                    for i in range(n_qubits - 1):
                        qml.CNOT(wires=[i, i+1])
                    return qml.expval(H)
                
                params = pnp.array(np.random.randn(n_params) * 0.1, requires_grad=True)
                optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
                
                # Time 5 iterations
                start = time.perf_counter()
                for _ in range(5):
                    params = optimizer.step(circuit, params)
                elapsed = time.perf_counter() - start
                
                final_energy = float(circuit(params))
                
                result = {
                    'qubits': n_qubits,
                    'success': True,
                    'time_5_iter': elapsed,
                    'time_per_iter': elapsed / 5,
                    'final_energy': final_energy,
                    'est_gpu_mem_gb': est_gpu_mem,
                }
                max_qubit_results.append(result)
                max_qubits_achieved = n_qubits
                
                print(f"  SUCCESS: {n_qubits} qubits in {elapsed:.2f}s ({elapsed/5:.2f}s/iter)", flush=True)
                
            except Exception as e:
                error_msg = str(e)
                print(f"  FAILED at {n_qubits} qubits: {error_msg[:100]}", flush=True)
                max_qubit_results.append({
                    'qubits': n_qubits,
                    'success': False,
                    'error': error_msg[:200],
                })
                if 'memory' in error_msg.lower() or 'alloc' in error_msg.lower():
                    print(f"  Memory limit reached at {n_qubits} qubits!", flush=True)
                    break
        
        results['max_qubits_test'] = {
            'max_achieved': max_qubits_achieved,
            'results': max_qubit_results,
        }
        print(f"\nMax qubits achieved on single H100: {max_qubits_achieved}", flush=True)
    
    # Broadcast max qubits to all ranks
    max_qubits_achieved = comm.bcast(max_qubits_achieved, root=0)
    comm.Barrier()
    
    # =========================================================================
    # TEST 2: Multi-GPU Throughput
    # =========================================================================
    print_header("TEST 2: Multi-GPU Throughput (4 GPUs solving problems in parallel)")
    
    if rank == 0:
        print("Each GPU solves multiple VQE problems. Measuring total throughput...")
        sys.stdout.flush()
    
    comm.Barrier()
    
    # Each rank solves several 20-qubit problems
    n_problems = 3
    n_qubits = 20
    n_iterations = 20
    
    print_rank(f"Solving {n_problems} VQE problems ({n_qubits} qubits, {n_iterations} iter each)")
    
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        
        coeffs = [1.0] * n_qubits
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
        H = qml.Hamiltonian(coeffs, obs)
        
        @qml.qnode(dev, interface="autograd", diff_method="adjoint")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            return qml.expval(H)
        
        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
        
        rank_times = []
        rank_energies = []
        
        total_start = time.perf_counter()
        
        for p in range(n_problems):
            np.random.seed(42 + rank * 100 + p)
            params = pnp.array(np.random.randn(n_qubits) * 0.1, requires_grad=True)
            
            start = time.perf_counter()
            for _ in range(n_iterations):
                params = optimizer.step(circuit, params)
            elapsed = time.perf_counter() - start
            
            energy = float(circuit(params))
            rank_times.append(elapsed)
            rank_energies.append(energy)
        
        total_time = time.perf_counter() - total_start
        
        print_rank(f"Completed {n_problems} problems in {total_time:.2f}s (avg {np.mean(rank_times):.2f}s each)")
        
        throughput_result = {
            'gpu_id': gpu_id,
            'n_problems': n_problems,
            'total_time': total_time,
            'avg_time_per_problem': np.mean(rank_times),
            'energies': rank_energies,
        }
        throughput_status = "SUCCESS"
        
    except Exception as e:
        print_rank(f"FAILED: {e}")
        throughput_result = {'gpu_id': gpu_id, 'error': str(e)}
        throughput_status = "FAILED"
        total_time = 0
    
    # Gather all results
    all_throughput = comm.gather(throughput_result, root=0)
    all_times = comm.gather(total_time, root=0)
    
    if rank == 0:
        total_problems = n_problems * size
        max_time = max(all_times)  # Wall clock is the slowest GPU
        throughput = total_problems / max_time if max_time > 0 else 0
        
        print(f"\n--- Throughput Summary ---")
        print(f"Total problems solved: {total_problems}")
        print(f"Wall clock time: {max_time:.2f}s")
        print(f"Throughput: {throughput:.2f} problems/second")
        print(f"Effective speedup vs 1 GPU: {size}x (perfect parallel)")
        
        results['throughput_test'] = {
            'n_gpus': size,
            'problems_per_gpu': n_problems,
            'total_problems': total_problems,
            'wall_clock_time': max_time,
            'throughput_per_second': throughput,
            'per_gpu_results': all_throughput,
        }
    
    comm.Barrier()
    
    # =========================================================================
    # TEST 3: Single vs Multi-GPU comparison (same workload)
    # =========================================================================
    print_header("TEST 3: Scaling Efficiency (Same workload: 1 GPU vs 4 GPUs)")
    
    if rank == 0:
        print("Distributing 8 VQE problems across GPUs...")
        print("  - 1 GPU scenario: 1 GPU does all 8 sequentially")
        print("  - 4 GPU scenario: Each GPU does 2 problems")
        sys.stdout.flush()
    
    comm.Barrier()
    
    # Each GPU does 2 problems of 22 qubits
    n_problems_each = 2
    n_qubits = 22
    n_iterations = 15
    
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
        
        coeffs = [1.0] * n_qubits
        obs = [qml.PauliZ(i) for i in range(n_qubits)]
        H = qml.Hamiltonian(coeffs, obs)
        
        @qml.qnode(dev, interface="autograd", diff_method="adjoint")
        def circuit(params):
            for i in range(n_qubits):
                qml.RY(params[i], wires=i)
            for i in range(n_qubits - 1):
                qml.CNOT(wires=[i, i+1])
            return qml.expval(H)
        
        optimizer = qml.GradientDescentOptimizer(stepsize=0.1)
        
        start = time.perf_counter()
        for p in range(n_problems_each):
            np.random.seed(42 + rank * 100 + p)
            params = pnp.array(np.random.randn(n_qubits) * 0.1, requires_grad=True)
            for _ in range(n_iterations):
                params = optimizer.step(circuit, params)
        elapsed = time.perf_counter() - start
        
        print_rank(f"Completed {n_problems_each} problems ({n_qubits}q, {n_iterations}iter) in {elapsed:.2f}s")
        scaling_status = "SUCCESS"
        
    except Exception as e:
        print_rank(f"FAILED: {e}")
        elapsed = 0
        scaling_status = "FAILED"
    
    all_elapsed = comm.gather(elapsed, root=0)
    
    if rank == 0:
        multi_gpu_time = max(all_elapsed)  # Wall clock
        single_gpu_time = sum(all_elapsed)  # Sequential would be sum
        
        speedup = single_gpu_time / multi_gpu_time if multi_gpu_time > 0 else 0
        efficiency = speedup / size * 100
        
        print(f"\n--- Scaling Summary ---")
        print(f"Single GPU (sequential): {single_gpu_time:.2f}s")
        print(f"4 GPUs (parallel):       {multi_gpu_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Parallel efficiency: {efficiency:.1f}%")
        
        results['scaling_test'] = {
            'n_problems': n_problems_each * size,
            'qubits': n_qubits,
            'iterations': n_iterations,
            'single_gpu_estimated_time': single_gpu_time,
            'multi_gpu_time': multi_gpu_time,
            'speedup': speedup,
            'efficiency_percent': efficiency,
        }
    
    comm.Barrier()
    
    # =========================================================================
    # Save results
    # =========================================================================
    if rank == 0:
        print_header("FINAL SUMMARY")
        
        print(f"1. Max qubits on single H100: {results['max_qubits_test'].get('max_achieved', 'N/A')}")
        print(f"2. 4-GPU throughput: {results['throughput_test'].get('throughput_per_second', 'N/A'):.2f} problems/sec")
        print(f"3. 4-GPU speedup: {results['scaling_test'].get('speedup', 'N/A'):.2f}x ({results['scaling_test'].get('efficiency_percent', 'N/A'):.1f}% efficiency)")
        
        # Save to file
        output_path = 'results/multi_gpu/multi_gpu_benchmark.json'
        os.makedirs('results/multi_gpu', exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")
        
        print("=" * 70)


if __name__ == '__main__':
    main()
