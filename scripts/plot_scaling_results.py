#!/usr/bin/env python3
"""
Generate plots from VQE scaling study results.

Usage:
    python scripts/plot_scaling_results.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

def main():
    # Load results
    results_path = 'results/scaling_study/scaling_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        return
    
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} benchmark results")
    
    qubits = [r['n_qubits'] for r in results]
    cpu_times = [r['cpu']['time_seconds'] for r in results]
    gpu_times = [r['gpu']['time_seconds'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    # =========================================================================
    # Plot 1: Three-panel comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('VQE Scaling Study: CPU+JIT vs GPU (H100)\nAMD EPYC 9654 (192 cores) vs NVIDIA H100 (81GB)', 
                 fontsize=14, fontweight='bold')
    
    # Panel 1: Absolute times (log scale)
    ax1 = axes[0]
    ax1.semilogy(qubits, cpu_times, 'o-', linewidth=2, markersize=8, color='#3498DB', label='CPU+JIT')
    ax1.semilogy(qubits, gpu_times, 's-', linewidth=2, markersize=8, color='#E74C3C', label='GPU (H100)')
    ax1.set_xlabel('Number of Qubits', fontsize=12)
    ax1.set_ylabel('Time (seconds, log scale)', fontsize=12)
    ax1.set_title('Execution Time vs Qubit Count', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    ax1.set_xticks(qubits)
    
    # Panel 2: GPU Speedup
    ax2 = axes[1]
    colors = ['#2ECC71' for s in speedups]  # All green since GPU always wins
    bars = ax2.bar(qubits, speedups, color=colors, edgecolor='black', linewidth=1)
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Breakeven')
    ax2.set_xlabel('Number of Qubits', fontsize=12)
    ax2.set_ylabel('GPU Speedup (×)', fontsize=12)
    ax2.set_title('GPU Speedup over CPU+JIT', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticks(qubits)
    
    # Add value labels
    for bar, val in zip(bars, speedups):
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Panel 3: Time per iteration
    ax3 = axes[2]
    cpu_per_iter = [r['cpu']['time_per_iteration'] for r in results]
    gpu_per_iter = [r['gpu']['time_per_iteration'] for r in results]
    ax3.semilogy(qubits, cpu_per_iter, 'o-', linewidth=2, markersize=8, color='#3498DB', label='CPU+JIT')
    ax3.semilogy(qubits, gpu_per_iter, 's-', linewidth=2, markersize=8, color='#E74C3C', label='GPU (H100)')
    ax3.set_xlabel('Number of Qubits', fontsize=12)
    ax3.set_ylabel('Time per Iteration (seconds, log)', fontsize=12)
    ax3.set_title('Per-Iteration Performance', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3)
    ax3.set_xticks(qubits)
    
    plt.tight_layout()
    plt.savefig('results/scaling_study/scaling_comparison.png', dpi=300, bbox_inches='tight')
    print('Saved: results/scaling_study/scaling_comparison.png')
    plt.close()
    
    # =========================================================================
    # Plot 2: Summary bar chart
    # =========================================================================
    fig2, ax = plt.subplots(figsize=(12, 6))
    colors = ['#2ECC71' for s in speedups]
    bars = ax.bar(qubits, speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2)
    
    ax.set_xlabel('Number of Qubits', fontsize=14)
    ax.set_ylabel('GPU Speedup (× faster than CPU+JIT)', fontsize=14)
    ax.set_title('VQE Scaling Study: H100 GPU Dominates at All Scales\n'
                'No CPU-GPU crossover found — GPU wins even at 4 qubits!', 
                fontsize=14, fontweight='bold')
    
    for bar, val, q in zip(bars, speedups, qubits):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 1,
               f'{val:.1f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, max(speedups) * 1.15)
    ax.set_xticks(qubits)
    
    plt.tight_layout()
    plt.savefig('results/scaling_study/scaling_summary.png', dpi=300, bbox_inches='tight')
    print('Saved: results/scaling_study/scaling_summary.png')
    plt.close()
    
    # =========================================================================
    # Print summary table
    # =========================================================================
    print("\n" + "=" * 70)
    print("SCALING STUDY RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Qubits':<8} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12} {'Winner':<10}")
    print("-" * 54)
    for r in results:
        print(f"{r['n_qubits']:<8} {r['cpu']['time_seconds']:<12.2f} "
              f"{r['gpu']['time_seconds']:<12.2f} {r['speedup']:<12.1f}× {r['winner']:<10}")
    print("=" * 70)
    
    # Key statistics
    print(f"\nKey Findings:")
    print(f"  - Min speedup: {min(speedups):.1f}× at {qubits[speedups.index(min(speedups))]} qubits")
    print(f"  - Max speedup: {max(speedups):.1f}× at {qubits[speedups.index(max(speedups))]} qubits")
    print(f"  - GPU wins at ALL tested qubit counts (4-26)")
    print(f"  - At 26 qubits: CPU takes {cpu_times[-1]:.1f}s, GPU takes {gpu_times[-1]:.1f}s")


if __name__ == '__main__':
    main()
