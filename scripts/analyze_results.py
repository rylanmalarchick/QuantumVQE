#!/usr/bin/env python3
"""
Analyze VQE benchmark results and generate performance plots.

This script generates visualizations for the three-factor speedup analysis:
1. Optimizer + JIT Effect (PennyLane Adam -> Optax+JIT)
2. GPU Device Effect (CPU -> lightning.gpu)
3. MPI Parallelization Effect (Serial -> MPI-N)
"""

import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Benchmark data from actual HPC runs
BENCHMARK_DATA = {
    # Implementation: (time_seconds, description)
    'Serial PennyLane Adam': (593.95, 'Baseline (no JIT)'),
    'Serial Optax+JIT': (143.80, 'CPU + Optax + Catalyst JIT'),
    'GPU (lightning.gpu)': (164.91, 'GPU + Optax (no Catalyst)'),
    'CPU+JIT (qjit)': (171.79, 'CPU + Catalyst @qjit'),
    'MPI-2': (8.45, '2 processes + Optax+JIT'),
    'MPI-4': (6.07, '4 processes + Optax+JIT'),
    'MPI-8': (5.48, '8 processes + Optax+JIT'),
    'MPI-16': (5.06, '16 processes + Optax+JIT'),
    'MPI-32': (5.04, '32 processes + Optax+JIT'),
}

BASELINE_TIME = 593.95  # Serial PennyLane Adam
OPTAX_JIT_TIME = 143.80  # Serial Optax+JIT (proper MPI baseline)


def print_results():
    """Print benchmark results to console."""
    print("=" * 70)
    print("VQE BENCHMARK RESULTS - THREE-FACTOR ANALYSIS")
    print("=" * 70)
    print(f"{'Implementation':<25} {'Time (s)':<12} {'Speedup':<12} {'vs Optax+JIT':<12}")
    print("-" * 70)
    
    for name, (time, desc) in BENCHMARK_DATA.items():
        speedup_vs_baseline = BASELINE_TIME / time
        speedup_vs_optax = OPTAX_JIT_TIME / time if 'MPI' in name else '-'
        if isinstance(speedup_vs_optax, float):
            print(f"{name:<25} {time:<12.2f} {speedup_vs_baseline:<12.2f}x {speedup_vs_optax:<12.2f}x")
        else:
            print(f"{name:<25} {time:<12.2f} {speedup_vs_baseline:<12.2f}x {speedup_vs_optax:<12}")
    print("=" * 70)


def create_performance_plots():
    """Generate comprehensive performance analysis plots."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VQE Performance Analysis - Three-Factor Speedup\nH₂ Molecule (4 qubits, 100 bond lengths)', 
                 fontsize=14, fontweight='bold')
    
    # Color scheme
    colors = {
        'baseline': '#E74C3C',      # Red - baseline
        'optax_jit': '#3498DB',     # Blue - Optax+JIT
        'gpu': '#9B59B6',           # Purple - GPU
        'qjit': '#1ABC9C',          # Teal - qjit
        'mpi': '#F39C12',           # Orange - MPI
    }
    
    # =========================================================================
    # Plot 1: Three-Factor Speedup Breakdown (Bar Chart)
    # =========================================================================
    ax1 = axes[0, 0]
    
    factors = ['Optimizer+JIT\n(4.13×)', 'GPU Device\n(3.60×)', 'MPI-32\n(28.53×)', 'Combined\n(117.85×)']
    speedups = [4.13, 3.60, 28.53, 117.85]
    factor_colors = [colors['optax_jit'], colors['gpu'], colors['mpi'], '#2ECC71']
    
    bars = ax1.bar(factors, speedups, color=factor_colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Speedup (×)', fontsize=12)
    ax1.set_title('Three-Factor Speedup Breakdown', fontsize=12, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_yscale('log')
    ax1.set_ylim([1, 200])
    
    # Add value labels
    for bar, val in zip(bars, speedups):
        ax1.text(bar.get_x() + bar.get_width()/2., bar.get_height() * 1.1,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # =========================================================================
    # Plot 2: All Implementations Runtime Comparison
    # =========================================================================
    ax2 = axes[0, 1]
    
    impl_names = ['Serial\nPennyLane', 'Serial\nOptax+JIT', 'GPU', 'MPI-2', 'MPI-4', 'MPI-8', 'MPI-16', 'MPI-32']
    impl_times = [593.95, 143.80, 164.91, 8.45, 6.07, 5.48, 5.06, 5.04]
    impl_colors = [colors['baseline'], colors['optax_jit'], colors['gpu']] + [colors['mpi']] * 5
    
    bars = ax2.bar(impl_names, impl_times, color=impl_colors, edgecolor='black', linewidth=1.2)
    ax2.set_ylabel('Runtime (seconds)', fontsize=12)
    ax2.set_title('Execution Time Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_yscale('log')
    
    # Add value labels
    for bar, val in zip(bars, impl_times):
        y_pos = bar.get_height() * 1.15
        ax2.text(bar.get_x() + bar.get_width()/2., y_pos,
                f'{val:.1f}s', ha='center', va='bottom', fontsize=8)
    
    # =========================================================================
    # Plot 3: MPI Strong Scaling (Corrected Baseline)
    # =========================================================================
    ax3 = axes[1, 0]
    
    mpi_procs = [1, 2, 4, 8, 16, 32]
    mpi_times = [OPTAX_JIT_TIME, 8.45, 6.07, 5.48, 5.06, 5.04]
    mpi_speedup = [OPTAX_JIT_TIME / t for t in mpi_times]
    ideal_speedup = mpi_procs  # Linear scaling
    
    ax3.plot(mpi_procs, mpi_speedup, 'o-', linewidth=2.5, markersize=10, 
             color=colors['mpi'], label='Actual Speedup', zorder=3)
    ax3.plot(mpi_procs, ideal_speedup, '--', linewidth=2, color='gray', 
             label='Ideal (Linear)', alpha=0.7)
    
    # Fill area between actual and ideal
    ax3.fill_between(mpi_procs, mpi_speedup, ideal_speedup, alpha=0.2, color=colors['mpi'])
    
    ax3.set_xlabel('Number of MPI Processes', fontsize=12)
    ax3.set_ylabel('Speedup vs Serial Optax+JIT (×)', fontsize=12)
    ax3.set_title('MPI Strong Scaling (Corrected Baseline)', fontsize=12, fontweight='bold')
    ax3.legend(loc='upper left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_xticks(mpi_procs)
    ax3.set_xticklabels(mpi_procs)
    
    # Annotate key points
    ax3.annotate(f'Super-linear!\n{mpi_speedup[1]:.1f}×', 
                xy=(2, mpi_speedup[1]), xytext=(2.5, 25),
                fontsize=9, ha='left',
                arrowprops=dict(arrowstyle='->', color='black', lw=1))
    
    # =========================================================================
    # Plot 4: Speedup vs Serial PennyLane Adam (Original Baseline)
    # =========================================================================
    ax4 = axes[1, 1]
    
    # All implementations vs original baseline
    all_names = ['Optax+JIT', 'GPU', 'MPI-2', 'MPI-4', 'MPI-8', 'MPI-16', 'MPI-32']
    all_times = [143.80, 164.91, 8.45, 6.07, 5.48, 5.06, 5.04]
    all_speedups = [BASELINE_TIME / t for t in all_times]
    bar_colors = [colors['optax_jit'], colors['gpu']] + [colors['mpi']] * 5
    
    bars = ax4.bar(all_names, all_speedups, color=bar_colors, edgecolor='black', linewidth=1.2)
    ax4.axhline(y=1, color=colors['baseline'], linestyle='--', linewidth=2, 
                label='Serial PennyLane Adam (1×)')
    ax4.set_ylabel('Speedup vs Serial PennyLane Adam (×)', fontsize=12)
    ax4.set_title('Total Speedup (vs Original Baseline)', fontsize=12, fontweight='bold')
    ax4.legend(loc='upper left')
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_xticklabels(all_names, rotation=30, ha='right')
    
    # Add value labels
    for bar, val in zip(bars, all_speedups):
        ax4.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 2,
                f'{val:.1f}×', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: results/performance_analysis.png")


def create_scaling_efficiency_plot():
    """Generate separate MPI scaling efficiency plot."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    mpi_procs = [1, 2, 4, 8, 16, 32]
    mpi_times = [OPTAX_JIT_TIME, 8.45, 6.07, 5.48, 5.06, 5.04]
    mpi_speedup = [OPTAX_JIT_TIME / t for t in mpi_times]
    efficiency = [s / p * 100 for s, p in zip(mpi_speedup, mpi_procs)]
    
    ax.plot(mpi_procs, efficiency, 's-', linewidth=2.5, markersize=12, 
            color='#F39C12', label='Parallel Efficiency')
    ax.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Ideal (100%)')
    
    ax.set_xlabel('Number of MPI Processes', fontsize=14)
    ax.set_ylabel('Parallel Efficiency (%)', fontsize=14)
    ax.set_title('MPI Parallel Efficiency\n(vs Serial Optax+JIT Baseline)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(mpi_procs)
    ax.set_xticklabels(mpi_procs)
    
    # Annotate efficiency values
    for p, e in zip(mpi_procs, efficiency):
        label = f'{e:.0f}%' if e < 200 else f'{e:.0f}%\n(super-linear)'
        ax.annotate(label, xy=(p, e), xytext=(0, 10), 
                   textcoords='offset points', ha='center', fontsize=10)
    
    # Add note about super-linear speedup
    ax.text(0.98, 0.02, 
            'Note: Efficiency >100% due to JIT compilation\n'
            'overhead amortized over fewer bond lengths\n'
            'per process (embarrassingly parallel)',
            transform=ax.transAxes, fontsize=9, va='bottom', ha='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('results/mpi_efficiency.png', dpi=300, bbox_inches='tight')
    print("Plot saved: results/mpi_efficiency.png")


def print_three_factor_analysis():
    """Print detailed three-factor analysis."""
    print("\n" + "=" * 70)
    print("THREE-FACTOR SPEEDUP ANALYSIS")
    print("=" * 70)
    
    print("\nFACTOR 1: Optimizer + JIT Compilation")
    print("-" * 50)
    print(f"  Before:  {BASELINE_TIME:.2f}s (PennyLane AdamOptimizer)")
    print(f"  After:   {OPTAX_JIT_TIME:.2f}s (Optax + Catalyst JIT)")
    print(f"  Speedup: {BASELINE_TIME/OPTAX_JIT_TIME:.2f}×")
    
    print("\nFACTOR 2: GPU Device Acceleration")
    print("-" * 50)
    gpu_time = BENCHMARK_DATA['GPU (lightning.gpu)'][0]
    print(f"  Before:  {BASELINE_TIME:.2f}s (lightning.qubit, CPU)")
    print(f"  After:   {gpu_time:.2f}s (lightning.gpu, H100)")
    print(f"  Speedup: {BASELINE_TIME/gpu_time:.2f}×")
    print(f"  Note:    CPU+JIT ({OPTAX_JIT_TIME:.2f}s) > GPU ({gpu_time:.2f}s) for 4 qubits")
    
    print("\nFACTOR 3: MPI Parallelization")
    print("-" * 50)
    print(f"  Baseline: {OPTAX_JIT_TIME:.2f}s (Serial Optax+JIT)")
    print(f"  {'Procs':<8} {'Time':<10} {'Speedup':<12} {'Efficiency':<12}")
    print(f"  {'-'*42}")
    
    mpi_data = [
        (2, 8.45), (4, 6.07), (8, 5.48), (16, 5.06), (32, 5.04)
    ]
    for procs, time in mpi_data:
        speedup = OPTAX_JIT_TIME / time
        efficiency = speedup / procs * 100
        print(f"  {procs:<8} {time:<10.2f} {speedup:<12.2f}× {efficiency:<12.1f}%")
    
    print("\nCOMBINED EFFECT")
    print("-" * 50)
    mpi32_time = BENCHMARK_DATA['MPI-32'][0]
    total_speedup = BASELINE_TIME / mpi32_time
    print(f"  From:    {BASELINE_TIME:.2f}s (Serial PennyLane Adam)")
    print(f"  To:      {mpi32_time:.2f}s (MPI-32 + Optax+JIT)")
    print(f"  Speedup: {total_speedup:.2f}×")
    print("=" * 70)


def main():
    """Main entry point."""
    print_results()
    print_three_factor_analysis()
    create_performance_plots()
    create_scaling_efficiency_plot()
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    print("Generated plots:")
    print("  - results/performance_analysis.png")
    print("  - results/mpi_efficiency.png")
    print("=" * 70)


if __name__ == '__main__':
    main()
