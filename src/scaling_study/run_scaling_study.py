#!/usr/bin/env python3
"""
VQE Scaling Study: CPU vs GPU Performance Analysis

This script runs a comprehensive benchmark comparing CPU+JIT vs GPU performance
across increasing qubit counts to find the crossover point where GPU wins.

Hardware Target:
    - CPU: 2x AMD EPYC 9654 (192 cores), 1.5TB RAM
    - GPU: NVIDIA H100 PCIe (81GB)

Usage:
    python run_scaling_study.py [--max-qubits N] [--output DIR]

The script runs everything in one execution - no need for multiple jobs.
"""

import os
import sys
import json
import time
import argparse
import traceback
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np

# Detect if we're on the cluster with GPU
def detect_environment():
    """Detect available compute environments."""
    env = {
        'has_gpu': False,
        'gpu_name': None,
        'gpu_memory_gb': None,
        'cpu_count': os.cpu_count(),
        'hostname': os.uname().nodename,
    }
    
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
            capture_output=True, text=True, timeout=10
        )
        if result.returncode == 0:
            env['has_gpu'] = True
            lines = result.stdout.strip().split('\n')
            if lines:
                parts = lines[0].split(',')
                env['gpu_name'] = parts[0].strip()
                env['gpu_memory_gb'] = int(parts[1].strip().split()[0]) / 1024
    except Exception:
        pass
    
    return env


# ============================================================================
# Configuration
# ============================================================================

# Qubit counts to test
DEFAULT_QUBIT_CONFIGS = [
    # (qubits, vqe_iterations, description)
    (4,  100, "Trivial - GPU overhead dominates"),
    (8,  100, "Small - Still CPU territory"),
    (12, 80,  "Medium - Approaching crossover"),
    (14, 60,  "Medium - Crossover zone"),
    (16, 50,  "Medium-Large - GPU warming up"),
    (18, 40,  "Large - GPU should lead"),
    (20, 30,  "Large - GPU advantage clear"),
    (22, 25,  "Very Large - Significant GPU win"),
    (24, 20,  "Very Large - Major GPU advantage"),
    (26, 15,  "Huge - Pushing limits"),
    (28, 10,  "Huge - Near GPU memory limit"),
    # (30, 5,   "Maximum - May OOM"),  # Uncomment to push limits
]

# Optimizer settings
LEARNING_RATE = 0.1
CONVERGENCE_TOL = 1e-6

# Ansatz settings
ANSATZ_LAYERS = 2  # Hardware-efficient ansatz layers


# ============================================================================
# CPU Benchmark (Optax + JIT)
# ============================================================================

def run_cpu_benchmark(n_qubits: int, n_iterations: int, seed: int = 42) -> Dict:
    """
    Run VQE benchmark on CPU with JIT compilation.
    
    Uses: lightning.qubit + Optax + JAX JIT
    """
    import jax
    jax.config.update("jax_platform_name", "cpu")
    import jax.numpy as jnp
    from jax import grad, jit, value_and_grad
    import optax
    import pennylane as qml
    
    from hamiltonians import create_transverse_field_ising
    
    # Create Hamiltonian
    H = create_transverse_field_ising(n_qubits)
    
    # Create device
    dev = qml.device("lightning.qubit", wires=n_qubits)
    
    # Number of parameters for hardware-efficient ansatz
    n_params = n_qubits * 2 * ANSATZ_LAYERS
    
    # Define the circuit
    @qml.qnode(dev, interface="jax")
    def circuit(params):
        params = params.reshape(ANSATZ_LAYERS, n_qubits, 2)
        for layer in range(ANSATZ_LAYERS):
            for q in range(n_qubits):
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(H)
    
    # JIT compile
    jit_circuit = jit(circuit)
    jit_grad = jit(grad(circuit))
    
    # Initialize parameters
    np.random.seed(seed)
    params = jnp.array(np.random.randn(n_params) * 0.1)
    
    # Optimizer
    optimizer = optax.adam(LEARNING_RATE)
    opt_state = optimizer.init(params)
    
    # Warm-up run (JIT compilation)
    _ = jit_circuit(params)
    _ = jit_grad(params)
    
    # Timed optimization
    energies = []
    start_time = time.perf_counter()
    
    for i in range(n_iterations):
        grads = jit_grad(params)
        updates, opt_state = optimizer.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        
        if i % 10 == 0 or i == n_iterations - 1:
            energy = float(jit_circuit(params))
            energies.append(energy)
    
    elapsed = time.perf_counter() - start_time
    final_energy = float(jit_circuit(params))
    
    return {
        'time_seconds': elapsed,
        'time_per_iteration': elapsed / n_iterations,
        'final_energy': final_energy,
        'n_iterations': n_iterations,
        'energies': energies,
    }


# ============================================================================
# GPU Benchmark (lightning.gpu + Gradient Descent)
# ============================================================================

def run_gpu_benchmark(n_qubits: int, n_iterations: int, seed: int = 42) -> Dict:
    """
    Run VQE benchmark on GPU.
    
    Uses: lightning.gpu + PennyLane's GradientDescentOptimizer
    Note: Optax is incompatible with PennyLane autograd arrays, so we use
          PennyLane's native optimizer instead.
    """
    import pennylane as qml
    from pennylane import numpy as pnp  # PennyLane's numpy for autograd
    
    # Check if lightning.gpu is available
    try:
        dev = qml.device("lightning.gpu", wires=n_qubits)
    except Exception as e:
        return {
            'error': f"lightning.gpu not available: {e}",
            'time_seconds': None,
        }
    
    from hamiltonians import create_transverse_field_ising
    
    # Create Hamiltonian
    H = create_transverse_field_ising(n_qubits)
    
    # Number of parameters
    n_params = n_qubits * 2 * ANSATZ_LAYERS
    
    # Define the circuit (autograd interface for GPU)
    @qml.qnode(dev, interface="autograd", diff_method="adjoint")
    def circuit(params):
        params = params.reshape(ANSATZ_LAYERS, n_qubits, 2)
        for layer in range(ANSATZ_LAYERS):
            for q in range(n_qubits):
                qml.RY(params[layer, q, 0], wires=q)
                qml.RZ(params[layer, q, 1], wires=q)
            for q in range(n_qubits - 1):
                qml.CNOT(wires=[q, q + 1])
        return qml.expval(H)
    
    # Initialize parameters (use PennyLane numpy for autograd compatibility)
    np.random.seed(seed)
    params = pnp.array(np.random.randn(n_params) * 0.1, requires_grad=True)
    
    # Use PennyLane's native Adam optimizer (compatible with autograd)
    optimizer = qml.AdamOptimizer(stepsize=LEARNING_RATE)
    
    # Warm-up run (GPU kernel compilation)
    _ = circuit(params)
    _ = qml.grad(circuit)(params)
    
    # Timed optimization
    energies = []
    start_time = time.perf_counter()
    
    for i in range(n_iterations):
        params, energy = optimizer.step_and_cost(circuit, params)
        
        if i % 10 == 0 or i == n_iterations - 1:
            energies.append(float(energy))
    
    elapsed = time.perf_counter() - start_time
    final_energy = float(circuit(params))
    
    return {
        'time_seconds': elapsed,
        'time_per_iteration': elapsed / n_iterations,
        'final_energy': final_energy,
        'n_iterations': n_iterations,
        'energies': energies,
    }


# ============================================================================
# Memory Estimation
# ============================================================================

def estimate_memory_usage(n_qubits: int) -> Dict:
    """Estimate memory requirements for a given qubit count."""
    state_vector_bytes = (2 ** n_qubits) * 16  # complex128
    
    # Multipliers for gradient computation overhead
    cpu_multiplier = 3.0  # JAX keeps intermediate states
    gpu_multiplier = 4.0  # cuStateVec + adjoint method
    
    return {
        'state_vector_mb': state_vector_bytes / (1024**2),
        'cpu_estimate_mb': state_vector_bytes * cpu_multiplier / (1024**2),
        'gpu_estimate_mb': state_vector_bytes * gpu_multiplier / (1024**2),
        'cpu_estimate_gb': state_vector_bytes * cpu_multiplier / (1024**3),
        'gpu_estimate_gb': state_vector_bytes * gpu_multiplier / (1024**3),
    }


# ============================================================================
# Plotting
# ============================================================================

def generate_plots(results: List[Dict], output_dir: str):
    """Generate performance comparison plots."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    # Extract data
    qubits = [r['n_qubits'] for r in results]
    cpu_times = [r['cpu']['time_seconds'] if r['cpu']['time_seconds'] else np.nan for r in results]
    gpu_times = [r['gpu']['time_seconds'] if r['gpu'].get('time_seconds') else np.nan for r in results]
    
    # Calculate speedup (GPU speedup = CPU_time / GPU_time)
    speedups = []
    for ct, gt in zip(cpu_times, gpu_times):
        if not np.isnan(ct) and not np.isnan(gt) and gt > 0:
            speedups.append(ct / gt)
        else:
            speedups.append(np.nan)
    
    # Find crossover point
    crossover_qubit = None
    for i, s in enumerate(speedups):
        if not np.isnan(s) and s > 1.0:
            crossover_qubit = qubits[i]
            break
    
    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VQE Scaling Study: CPU+JIT vs GPU (H100)\nAMD EPYC 9654 + NVIDIA H100 PCIe 81GB', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Absolute times (log scale)
    ax1 = axes[0, 0]
    ax1.semilogy(qubits, cpu_times, 'o-', linewidth=2, markersize=8, 
                 color='#3498DB', label='CPU+JIT (lightning.qubit)')
    ax1.semilogy(qubits, gpu_times, 's-', linewidth=2, markersize=8,
                 color='#E74C3C', label='GPU (lightning.gpu)')
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax1.set_title('Execution Time vs Qubit Count', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    if crossover_qubit:
        ax1.axvline(x=crossover_qubit, color='green', linestyle='--', alpha=0.7,
                   label=f'Crossover: {crossover_qubit} qubits')
    
    # Plot 2: GPU Speedup
    ax2 = axes[0, 1]
    colors = ['#2ECC71' if s > 1 else '#E74C3C' for s in speedups]
    bars = ax2.bar(qubits, speedups, color=colors, edgecolor='black', linewidth=1)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Breakeven (1.0×)')
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('GPU Speedup (CPU_time / GPU_time)', fontsize=12)
    ax2.set_title('GPU Speedup vs CPU+JIT', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar, val in zip(bars, speedups):
        if not np.isnan(val):
            ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.1,
                    f'{val:.2f}×', ha='center', va='bottom', fontsize=9)
    
    # Plot 3: Time per iteration
    ax3 = axes[1, 0]
    cpu_per_iter = [r['cpu'].get('time_per_iteration') if r['cpu'].get('time_per_iteration') else np.nan for r in results]
    gpu_per_iter = [r['gpu'].get('time_per_iteration') if r['gpu'].get('time_per_iteration') else np.nan for r in results]
    
    ax3.semilogy(qubits, cpu_per_iter, 'o-', linewidth=2, markersize=8,
                 color='#3498DB', label='CPU+JIT')
    ax3.semilogy(qubits, gpu_per_iter, 's-', linewidth=2, markersize=8,
                 color='#E74C3C', label='GPU')
    ax3.set_xlabel('Number of Qubits', fontsize=12)
    ax3.set_ylabel('Time per VQE Iteration (seconds, log)', fontsize=12)
    ax3.set_title('Per-Iteration Performance', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Memory usage
    ax4 = axes[1, 1]
    mem_data = [estimate_memory_usage(q) for q in qubits]
    cpu_mem = [m['cpu_estimate_gb'] for m in mem_data]
    gpu_mem = [m['gpu_estimate_gb'] for m in mem_data]
    
    ax4.semilogy(qubits, cpu_mem, 'o-', linewidth=2, markersize=8,
                 color='#3498DB', label='CPU Est. Memory')
    ax4.semilogy(qubits, gpu_mem, 's-', linewidth=2, markersize=8,
                 color='#E74C3C', label='GPU Est. Memory')
    ax4.axhline(y=81, color='#E74C3C', linestyle=':', linewidth=2, 
               label='H100 VRAM (81GB)', alpha=0.7)
    ax4.axhline(y=1500, color='#3498DB', linestyle=':', linewidth=2,
               label='System RAM (1.5TB)', alpha=0.7)
    ax4.set_xlabel('Number of Qubits', fontsize=12)
    ax4.set_ylabel('Estimated Memory (GB, log scale)', fontsize=12)
    ax4.set_title('Memory Requirements', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=9, loc='upper left')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'scaling_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")
    
    # Create summary plot
    create_summary_plot(results, speedups, crossover_qubit, output_dir)
    
    return crossover_qubit


def create_summary_plot(results: List[Dict], speedups: List[float], 
                        crossover_qubit: Optional[int], output_dir: str):
    """Create a single summary plot highlighting key findings."""
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')
    
    qubits = [r['n_qubits'] for r in results]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color by winner
    colors = ['#2ECC71' if s > 1 else '#3498DB' for s in speedups]
    bars = ax.bar(qubits, speedups, color=colors, edgecolor='black', linewidth=1.5)
    
    # Add breakeven line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    
    # Add crossover annotation
    if crossover_qubit:
        ax.axvline(x=crossover_qubit, color='#9B59B6', linestyle='--', linewidth=2, alpha=0.7)
        ax.annotate(f'Crossover: {crossover_qubit} qubits\nGPU wins beyond this point',
                   xy=(crossover_qubit, 1.0),
                   xytext=(crossover_qubit + 2, 0.5),
                   fontsize=11,
                   arrowprops=dict(arrowstyle='->', color='#9B59B6', lw=2),
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Labels
    ax.set_xlabel('Number of Qubits', fontsize=14)
    ax.set_ylabel('GPU Speedup (× faster than CPU+JIT)', fontsize=14)
    ax.set_title('VQE Scaling Study: When Does GPU Beat CPU+JIT?\n'
                'AMD EPYC 9654 (192 cores) vs NVIDIA H100 (81GB)', 
                fontsize=14, fontweight='bold')
    
    # Add value labels on bars
    for bar, val, q in zip(bars, speedups, qubits):
        if not np.isnan(val):
            label = f'{val:.2f}×'
            winner = 'GPU' if val > 1 else 'CPU'
            y_pos = bar.get_height() + 0.05 if val > 0 else 0.05
            ax.text(bar.get_x() + bar.get_width()/2., y_pos,
                   f'{label}\n({winner})', ha='center', va='bottom', 
                   fontsize=9, fontweight='bold')
    
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498DB', edgecolor='black', label='CPU+JIT Wins'),
        Patch(facecolor='#2ECC71', edgecolor='black', label='GPU Wins'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=12)
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(bottom=0)
    
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'scaling_summary.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Summary plot saved: {plot_path}")


# ============================================================================
# Main Driver
# ============================================================================

def run_scaling_study(
    qubit_configs: List[Tuple[int, int, str]] = None,
    output_dir: str = "results/scaling_study",
    max_qubits: int = 30,
    skip_gpu: bool = False,
    skip_cpu: bool = False,
) -> List[Dict]:
    """
    Run the complete scaling study.
    
    Args:
        qubit_configs: List of (n_qubits, n_iterations, description) tuples
        output_dir: Directory to save results
        max_qubits: Maximum qubit count to test
        skip_gpu: Skip GPU benchmarks (for testing)
        skip_cpu: Skip CPU benchmarks (for testing)
    
    Returns:
        List of result dictionaries
    """
    if qubit_configs is None:
        qubit_configs = [c for c in DEFAULT_QUBIT_CONFIGS if c[0] <= max_qubits]
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Detect environment
    env = detect_environment()
    
    print("=" * 70)
    print("VQE SCALING STUDY: CPU+JIT vs GPU")
    print("=" * 70)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Hostname: {env['hostname']}")
    print(f"CPU cores: {env['cpu_count']}")
    if env['has_gpu']:
        print(f"GPU: {env['gpu_name']} ({env['gpu_memory_gb']:.1f} GB)")
    else:
        print("GPU: Not detected")
    print(f"Output directory: {output_dir}")
    print(f"Qubit range: {qubit_configs[0][0]} to {qubit_configs[-1][0]}")
    print("=" * 70)
    
    results = []
    
    for n_qubits, n_iterations, description in qubit_configs:
        print(f"\n{'='*70}")
        print(f"TESTING {n_qubits} QUBITS ({description})")
        print(f"{'='*70}")
        
        mem = estimate_memory_usage(n_qubits)
        print(f"State vector: 2^{n_qubits} = {2**n_qubits:,} amplitudes")
        print(f"Est. CPU memory: {mem['cpu_estimate_gb']:.2f} GB")
        print(f"Est. GPU memory: {mem['gpu_estimate_gb']:.2f} GB")
        print(f"VQE iterations: {n_iterations}")
        
        result = {
            'n_qubits': n_qubits,
            'n_iterations': n_iterations,
            'description': description,
            'memory_estimate': mem,
            'cpu': {},
            'gpu': {},
        }
        
        # CPU Benchmark
        if not skip_cpu:
            print(f"\n--- CPU+JIT Benchmark ---")
            try:
                cpu_result = run_cpu_benchmark(n_qubits, n_iterations)
                result['cpu'] = cpu_result
                print(f"CPU Time: {cpu_result['time_seconds']:.2f}s "
                      f"({cpu_result['time_per_iteration']*1000:.2f}ms/iter)")
                print(f"Final energy: {cpu_result['final_energy']:.6f}")
            except Exception as e:
                print(f"CPU benchmark failed: {e}")
                traceback.print_exc()
                result['cpu'] = {'error': str(e), 'time_seconds': None}
        else:
            result['cpu'] = {'skipped': True, 'time_seconds': None}
        
        # GPU Benchmark
        if not skip_gpu and env['has_gpu']:
            print(f"\n--- GPU Benchmark ---")
            try:
                gpu_result = run_gpu_benchmark(n_qubits, n_iterations)
                result['gpu'] = gpu_result
                if gpu_result.get('time_seconds'):
                    print(f"GPU Time: {gpu_result['time_seconds']:.2f}s "
                          f"({gpu_result['time_per_iteration']*1000:.2f}ms/iter)")
                    print(f"Final energy: {gpu_result['final_energy']:.6f}")
                else:
                    print(f"GPU benchmark failed: {gpu_result.get('error')}")
            except Exception as e:
                print(f"GPU benchmark failed: {e}")
                traceback.print_exc()
                result['gpu'] = {'error': str(e), 'time_seconds': None}
        elif skip_gpu:
            result['gpu'] = {'skipped': True, 'time_seconds': None}
        else:
            result['gpu'] = {'no_gpu': True, 'time_seconds': None}
        
        # Calculate speedup
        cpu_time = result['cpu'].get('time_seconds')
        gpu_time = result['gpu'].get('time_seconds')
        if cpu_time and gpu_time:
            speedup = cpu_time / gpu_time
            winner = "GPU" if speedup > 1 else "CPU+JIT"
            print(f"\n>>> SPEEDUP: {speedup:.2f}x - {winner} wins! <<<")
            result['speedup'] = speedup
            result['winner'] = winner
        else:
            result['speedup'] = None
            result['winner'] = None
        
        results.append(result)
        
        # Save intermediate results
        results_path = os.path.join(output_dir, 'scaling_results.json')
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nIntermediate results saved to: {results_path}")
        
        # Check if we should stop (OOM likely)
        if result['cpu'].get('error') and 'memory' in result['cpu'].get('error', '').lower():
            print("\n!!! CPU ran out of memory - stopping study !!!")
            break
        if result['gpu'].get('error') and 'memory' in result['gpu'].get('error', '').lower():
            print("\n!!! GPU ran out of memory - stopping study !!!")
            break
    
    # Generate plots
    print(f"\n{'='*70}")
    print("GENERATING PLOTS")
    print(f"{'='*70}")
    
    try:
        crossover = generate_plots(results, output_dir)
        if crossover:
            print(f"\n>>> GPU CROSSOVER POINT: {crossover} qubits <<<")
    except Exception as e:
        print(f"Plot generation failed: {e}")
        traceback.print_exc()
    
    # Print summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Qubits':<8} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<10} {'Winner':<10}")
    print("-" * 52)
    for r in results:
        cpu_t = r['cpu'].get('time_seconds', 'N/A')
        gpu_t = r['gpu'].get('time_seconds', 'N/A')
        speedup = r.get('speedup', 'N/A')
        winner = r.get('winner') or 'N/A'
        
        cpu_str = f"{cpu_t:.2f}" if isinstance(cpu_t, float) else str(cpu_t)
        gpu_str = f"{gpu_t:.2f}" if isinstance(gpu_t, float) else str(gpu_t)
        speedup_str = f"{speedup:.2f}x" if isinstance(speedup, float) else str(speedup)
        winner_str = str(winner)
        
        print(f"{r['n_qubits']:<8} {cpu_str:<12} {gpu_str:<12} {speedup_str:<10} {winner_str:<10}")
    print("=" * 70)
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Results saved to: {output_dir}")
    
    return results


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='VQE Scaling Study: CPU+JIT vs GPU Performance'
    )
    parser.add_argument(
        '--max-qubits', type=int, default=28,
        help='Maximum number of qubits to test (default: 28)'
    )
    parser.add_argument(
        '--output', type=str, default='results/scaling_study',
        help='Output directory for results (default: results/scaling_study)'
    )
    parser.add_argument(
        '--skip-gpu', action='store_true',
        help='Skip GPU benchmarks (for testing on CPU-only machines)'
    )
    parser.add_argument(
        '--skip-cpu', action='store_true',
        help='Skip CPU benchmarks (for testing GPU only)'
    )
    
    args = parser.parse_args()
    
    run_scaling_study(
        output_dir=args.output,
        max_qubits=args.max_qubits,
        skip_gpu=args.skip_gpu,
        skip_cpu=args.skip_cpu,
    )


if __name__ == '__main__':
    main()
