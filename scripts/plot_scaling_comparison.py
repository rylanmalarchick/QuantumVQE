#!/usr/bin/env python3
"""
Plot the CPU vs GPU scaling study from results/scaling_study/scaling_results.json:
runtime of both software stacks and their CPU/GPU time ratio against qubit count.

Usage:
    python scripts/plot_scaling_comparison.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Set global style for presentation-quality figures
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

def load_scaling_results():
    """Load the scaling study results."""
    results_path = 'results/scaling_study/scaling_results.json'
    if not os.path.exists(results_path):
        print(f"Error: {results_path} not found")
        return None
    
    with open(results_path) as f:
        results = json.load(f)
    
    print(f"Loaded {len(results)} benchmark results")
    return results


def plot_scaling_comparison(results):
    """
    Create a clean 2-panel figure for CPU vs GPU scaling.
    Panel 1: Runtime comparison (log scale)
    Panel 2: CPU/GPU time ratio
    """
    qubits = np.array([r['n_qubits'] for r in results])
    cpu_times = np.array([r['cpu']['time_seconds'] for r in results])
    gpu_times = np.array([r['gpu']['time_seconds'] for r in results])
    ratios = np.array([r['speedup'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CPU vs GPU Scaling Study (4-26 Qubits)\nlightning.qubit + JAX/jax.jit + Optax vs lightning.gpu + autograd/adjoint + PennyLane Adam', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # =========================================================================
    # Panel 1: Execution Time (log scale)
    # =========================================================================
    ax1 = axes[0]
    
    # Plot with larger markers and thicker lines
    ax1.semilogy(qubits, cpu_times, 'o-', linewidth=2.5, markersize=10, 
                 color='#3498DB', label='CPU stack (lightning.qubit, JAX)', zorder=3)
    ax1.semilogy(qubits, gpu_times, 's-', linewidth=2.5, markersize=10, 
                 color='#E74C3C', label='GPU stack (lightning.gpu, autograd)', zorder=3)
    
    ax1.fill_between(qubits, gpu_times, cpu_times, alpha=0.15, color='#2ECC71')
    
    
    ax1.set_xlabel('Number of Qubits', fontsize=13)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=13)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left', frameon=True, fancybox=True, shadow=True)
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(qubits)
    ax1.set_xticklabels(qubits, rotation=0)
    ax1.set_xlim(3, 27)
    
    # Add state vector size annotations on secondary x-axis
    ax1_top = ax1.twiny()
    ax1_top.set_xlim(ax1.get_xlim())
    ax1_top.set_xticks([4, 12, 20, 26])
    ax1_top.set_xticklabels(['256B', '64KB', '16MB', '1GB'], fontsize=9, color='gray')
    ax1_top.set_xlabel('State Vector Size', fontsize=10, color='gray')
    
    # =========================================================================
    # Panel 2: CPU/GPU time ratio
    # =========================================================================
    ax2 = axes[1]
    
    # Color bars by ratio magnitude
    colors = ['#27AE60' if s >= 20 else '#F39C12' if s >= 10 else '#E74C3C' for s in ratios]
    bars = ax2.bar(qubits, ratios, color=colors, edgecolor='black', linewidth=1.2, width=1.5)
    
    # Add breakeven line
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Ratio 1')
    
    ax2.set_xlabel('Number of Qubits', fontsize=13)
    ax2.set_ylabel('CPU time / GPU time', fontsize=13)
    ax2.set_title('CPU/GPU Time Ratio (two software stacks)', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticks(qubits)
    ax2.set_xticklabels(qubits, rotation=0)
    ax2.set_xlim(2, 28)
    ax2.set_ylim(0, max(ratios) * 1.2)
    
    # Add value labels - stagger them to avoid overlap
    for i, (bar, val) in enumerate(zip(bars, ratios)):
        # Alternate label positions for dense bars
        y_offset = 2 if i % 2 == 0 else 5
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + y_offset,
                f'{val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/scaling_study/scaling_comparison.png', dpi=300, bbox_inches='tight')
    print('Saved: results/scaling_study/scaling_comparison.png')
    plt.close()


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("SCALING STUDY RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Qubits':<8} {'State Vec':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'CPU/GPU':<12}")
    print("-" * 56)
    for r in results:
        sv_size = 2**r['n_qubits'] * 16
        if sv_size < 1024:
            sv_str = f"{sv_size} B"
        elif sv_size < 1024**2:
            sv_str = f"{sv_size/1024:.0f} KB"
        elif sv_size < 1024**3:
            sv_str = f"{sv_size/1024**2:.0f} MB"
        else:
            sv_str = f"{sv_size/1024**3:.1f} GB"
        
        print(f"{r['n_qubits']:<8} {sv_str:<12} {r['cpu']['time_seconds']:<12.2f} "
              f"{r['gpu']['time_seconds']:<12.2f} {r['speedup']:<12.1f}")
    print("=" * 80)


def main():
    results = load_scaling_results()
    if results is None:
        return
    
    plot_scaling_comparison(results)
    print_summary_table(results)


if __name__ == '__main__':
    main()
