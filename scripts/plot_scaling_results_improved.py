#!/usr/bin/env python3
"""
Improved plots for VQE scaling study results.
Fixes: axis labels, overlapping text, cleaner styling for presentations.

Usage:
    python scripts/plot_scaling_results_improved.py
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
    Panel 2: GPU speedup with trend line
    """
    qubits = np.array([r['n_qubits'] for r in results])
    cpu_times = np.array([r['cpu']['time_seconds'] for r in results])
    gpu_times = np.array([r['gpu']['time_seconds'] for r in results])
    speedups = np.array([r['speedup'] for r in results])
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('CPU vs GPU Scaling Study (4-26 Qubits)\nAMD EPYC 9654 vs NVIDIA H100', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # =========================================================================
    # Panel 1: Execution Time (log scale)
    # =========================================================================
    ax1 = axes[0]
    
    # Plot with larger markers and thicker lines
    ax1.semilogy(qubits, cpu_times, 'o-', linewidth=2.5, markersize=10, 
                 color='#3498DB', label='CPU (JIT-compiled)', zorder=3)
    ax1.semilogy(qubits, gpu_times, 's-', linewidth=2.5, markersize=10, 
                 color='#E74C3C', label='GPU (H100)', zorder=3)
    
    # Fill between to show the gap
    ax1.fill_between(qubits, gpu_times, cpu_times, alpha=0.15, color='#2ECC71')
    
    # Annotate the gap at 26 qubits
    ax1.annotate(f'{speedups[-1]:.0f}x gap', 
                 xy=(26, np.sqrt(cpu_times[-1] * gpu_times[-1])),
                 fontsize=11, fontweight='bold', color='#27AE60',
                 ha='left', va='center',
                 xytext=(26.3, np.sqrt(cpu_times[-1] * gpu_times[-1])))
    
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
    # Panel 2: GPU Speedup
    # =========================================================================
    ax2 = axes[1]
    
    # Color bars by speedup magnitude
    colors = ['#27AE60' if s >= 20 else '#F39C12' if s >= 10 else '#E74C3C' for s in speedups]
    bars = ax2.bar(qubits, speedups, color=colors, edgecolor='black', linewidth=1.2, width=1.5)
    
    # Add breakeven line
    ax2.axhline(y=1.0, color='black', linestyle='--', linewidth=2, label='Breakeven (1x)')
    
    # Add trend annotation
    ax2.annotate('', xy=(26, 75), xytext=(8, 10),
                 arrowprops=dict(arrowstyle='->', color='#2C3E50', lw=2))
    ax2.text(16, 50, 'GPU advantage\nincreases with\nqubit count', 
             fontsize=10, ha='center', va='center', style='italic',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax2.set_xlabel('Number of Qubits', fontsize=13)
    ax2.set_ylabel('GPU Speedup (x faster)', fontsize=13)
    ax2.set_title('GPU Speedup over CPU', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_xticks(qubits)
    ax2.set_xticklabels(qubits, rotation=0)
    ax2.set_xlim(2, 28)
    ax2.set_ylim(0, max(speedups) * 1.2)
    
    # Add value labels - stagger them to avoid overlap
    for i, (bar, val) in enumerate(zip(bars, speedups)):
        # Alternate label positions for dense bars
        y_offset = 2 if i % 2 == 0 else 5
        ax2.text(bar.get_x() + bar.get_width()/2., bar.get_height() + y_offset,
                f'{val:.0f}x', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/scaling_study/scaling_comparison.png', dpi=300, bbox_inches='tight')
    print('Saved: results/scaling_study/scaling_comparison.png')
    plt.close()


def plot_scaling_summary(results):
    """
    Create a single summary figure showing the key scaling message.
    """
    qubits = np.array([r['n_qubits'] for r in results])
    cpu_times = np.array([r['cpu']['time_seconds'] for r in results])
    gpu_times = np.array([r['gpu']['time_seconds'] for r in results])
    speedups = np.array([r['speedup'] for r in results])
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Create gradient colors based on speedup
    norm = plt.Normalize(min(speedups), max(speedups))
    cmap = plt.cm.RdYlGn  # Red-Yellow-Green colormap
    colors = [cmap(norm(s)) for s in speedups]
    
    bars = ax.bar(qubits, speedups, color=colors, edgecolor='black', linewidth=1.5, width=1.6)
    
    # Add colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Speedup Factor', fontsize=11)
    
    # Breakeven line
    ax.axhline(y=1.0, color='black', linestyle='--', linewidth=2, alpha=0.7)
    ax.text(5, 3, 'Breakeven (1x)', fontsize=10, color='gray')
    
    ax.set_xlabel('Number of Qubits', fontsize=14)
    ax.set_ylabel('GPU Speedup (x faster than CPU)', fontsize=14)
    ax.set_title('VQE GPU Scaling: H100 Outperforms CPU at All Qubit Counts\n'
                 'Speedup ranges from 3.5x (12 qubits) to 81.5x (24 qubits)', 
                 fontsize=14, fontweight='bold')
    
    # Value labels
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
               f'{val:.0f}x', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(0, max(speedups) * 1.2)
    ax.set_xticks(qubits)
    
    # Add key insights box
    textstr = '\n'.join([
        'Key Findings:',
        f'  Min speedup: {min(speedups):.1f}x ({qubits[np.argmin(speedups)]} qubits)',
        f'  Max speedup: {max(speedups):.1f}x ({qubits[np.argmax(speedups)]} qubits)',
        f'  At 26 qubits: {cpu_times[-1]:.0f}s CPU vs {gpu_times[-1]:.1f}s GPU'
    ])
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props, family='monospace')
    
    plt.tight_layout()
    plt.savefig('results/scaling_study/scaling_summary.png', dpi=300, bbox_inches='tight')
    print('Saved: results/scaling_study/scaling_summary.png')
    plt.close()


def plot_exponential_scaling(results):
    """
    Show the exponential nature of quantum simulation and why GPU matters.
    """
    qubits = np.array([r['n_qubits'] for r in results])
    cpu_times = np.array([r['cpu']['time_seconds'] for r in results])
    gpu_times = np.array([r['gpu']['time_seconds'] for r in results])
    
    # State vector sizes
    state_vector_bytes = 2**qubits * 16  # complex128 = 16 bytes
    state_vector_mb = state_vector_bytes / (1024**2)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Quantum Simulation Scaling: Exponential Complexity', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    # Panel 1: State vector size (exponential)
    ax1 = axes[0]
    ax1.semilogy(qubits, state_vector_mb, 'o-', linewidth=2.5, markersize=10, color='#9B59B6')
    ax1.fill_between(qubits, state_vector_mb, alpha=0.2, color='#9B59B6')
    
    # Add memory limit lines
    ax1.axhline(y=80*1024, color='#E74C3C', linestyle='--', linewidth=2, label='H100 Memory (80GB)')
    ax1.axhline(y=1.5*1024*1024, color='#3498DB', linestyle='--', linewidth=2, label='Vega Node RAM (1.5TB)')
    
    ax1.set_xlabel('Number of Qubits', fontsize=13)
    ax1.set_ylabel('State Vector Size (MB, log scale)', fontsize=13)
    ax1.set_title('State Vector Size Grows Exponentially', fontsize=14, fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, linestyle='--')
    ax1.set_xticks(qubits)
    ax1.set_xlim(3, 27)
    
    # Annotate key points
    ax1.annotate('256 B', xy=(4, state_vector_mb[0]), xytext=(5.5, state_vector_mb[0]*3),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)
    ax1.annotate('1 GB', xy=(26, state_vector_mb[-1]), xytext=(24, state_vector_mb[-1]*3),
                arrowprops=dict(arrowstyle='->', color='gray'), fontsize=9)
    
    # Panel 2: Time per iteration scaling
    ax2 = axes[1]
    cpu_per_iter = np.array([r['cpu']['time_per_iteration'] for r in results])
    gpu_per_iter = np.array([r['gpu']['time_per_iteration'] for r in results])
    
    ax2.semilogy(qubits, cpu_per_iter, 'o-', linewidth=2.5, markersize=10, 
                 color='#3498DB', label='CPU')
    ax2.semilogy(qubits, gpu_per_iter, 's-', linewidth=2.5, markersize=10, 
                 color='#E74C3C', label='GPU')
    
    ax2.fill_between(qubits, gpu_per_iter, cpu_per_iter, alpha=0.15, color='#2ECC71')
    
    ax2.set_xlabel('Number of Qubits', fontsize=13)
    ax2.set_ylabel('Time per VQE Iteration (seconds)', fontsize=13)
    ax2.set_title('Per-Iteration Performance', fontsize=14, fontweight='bold')
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3, linestyle='--')
    ax2.set_xticks(qubits)
    ax2.set_xlim(3, 27)
    
    # Add annotation about GPU staying flat
    ax2.annotate('GPU scales gracefully', xy=(20, 0.04), fontsize=10, 
                 color='#E74C3C', fontweight='bold')
    ax2.annotate('CPU time explodes', xy=(20, 2), fontsize=10, 
                 color='#3498DB', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/scaling_study/exponential_scaling.png', dpi=300, bbox_inches='tight')
    print('Saved: results/scaling_study/exponential_scaling.png')
    plt.close()


def print_summary_table(results):
    """Print a formatted summary table."""
    print("\n" + "=" * 80)
    print("SCALING STUDY RESULTS SUMMARY")
    print("=" * 80)
    print(f"{'Qubits':<8} {'State Vec':<12} {'CPU (s)':<12} {'GPU (s)':<12} {'Speedup':<12}")
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
              f"{r['gpu']['time_seconds']:<12.2f} {r['speedup']:<12.1f}x")
    print("=" * 80)


def main():
    results = load_scaling_results()
    if results is None:
        return
    
    # Generate all plots
    plot_scaling_comparison(results)
    plot_scaling_summary(results)
    plot_exponential_scaling(results)
    print_summary_table(results)
    
    print("\nAll improved scaling plots generated successfully!")


if __name__ == '__main__':
    main()
