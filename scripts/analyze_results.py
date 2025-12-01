#!/usr/bin/env python3
"""
Analyze VQE benchmark results and generate performance plots.
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

def extract_runtime(log_file):
    """Extract total runtime in seconds from a log file."""
    with open(log_file, 'r') as f:
        content = f.read()
    
    match = re.search(r'Total runtime: ([\d.]+) seconds', content)
    if match:
        return float(match.group(1))
    return None

def main():
    logs_dir = Path('logs')
    
    # Extract timing data
    results = {
        'Serial': extract_runtime(logs_dir / 'serial_output.log'),
        'GPU/JIT': extract_runtime(logs_dir / 'gpu_output.log'),
        'MPI-2': extract_runtime(logs_dir / 'mpi_2_output.log'),
        'MPI-4': extract_runtime(logs_dir / 'mpi_4_output.log'),
        'MPI-8': extract_runtime(logs_dir / 'mpi_8_output.log'),
        'MPI-16': extract_runtime(logs_dir / 'mpi_16_output.log'),
        'MPI-32': extract_runtime(logs_dir / 'mpi_32_output.log'),
    }
    
    print("="*60)
    print("VQE BENCHMARK RESULTS")
    print("="*60)
    for name, runtime in results.items():
        if runtime:
            speedup = results['Serial'] / runtime if runtime else 0
            print(f"{name:12s}: {runtime:8.2f}s  (Speedup: {speedup:6.2f}x)")
    print("="*60)
    
    # Prepare data for plotting
    serial_time = results['Serial']
    
    # MPI scaling data
    mpi_procs = [1, 2, 4, 8, 16, 32]
    mpi_times = [
        serial_time,  # 1 process = serial
        results['MPI-2'],
        results['MPI-4'],
        results['MPI-8'],
        results['MPI-16'],
        results['MPI-32']
    ]
    mpi_speedup = [serial_time / t if t else 0 for t in mpi_times]
    mpi_efficiency = [s / p * 100 for s, p in zip(mpi_speedup, mpi_procs)]
    
    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('VQE Performance Analysis on HPC Cluster', fontsize=16, fontweight='bold')
    
    # Plot 1: Runtime comparison
    ax1 = axes[0, 0]
    implementations = ['Serial', 'GPU/JIT', 'MPI-2', 'MPI-4', 'MPI-8', 'MPI-16', 'MPI-32']
    times = [results[impl] for impl in implementations]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#BC4B51', '#8B1E3F']
    bars = ax1.bar(implementations, times, color=colors, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Runtime (seconds)', fontsize=12)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_xticklabels(implementations, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 2: Speedup comparison
    ax2 = axes[0, 1]
    speedups = [serial_time / t if t else 0 for t in times]
    bars = ax2.bar(implementations, speedups, color=colors, edgecolor='black', linewidth=1.5)
    ax2.axhline(y=1, color='red', linestyle='--', linewidth=2, label='Serial Baseline')
    ax2.set_ylabel('Speedup (×)', fontsize=12)
    ax2.set_title('Speedup vs Serial Baseline', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    ax2.set_xticklabels(implementations, rotation=45, ha='right')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}×',
                ha='center', va='bottom', fontsize=9)
    
    # Plot 3: MPI Strong Scaling
    ax3 = axes[1, 0]
    ax3.plot(mpi_procs, mpi_speedup, 'o-', linewidth=2, markersize=8, 
             color='#2E86AB', label='Actual Speedup')
    ax3.plot(mpi_procs, mpi_procs, '--', linewidth=2, color='red', 
             label='Ideal (Linear) Speedup')
    ax3.set_xlabel('Number of MPI Processes', fontsize=12)
    ax3.set_ylabel('Speedup (×)', fontsize=12)
    ax3.set_title('MPI Strong Scaling Performance', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    ax3.set_yscale('log', base=2)
    
    # Plot 4: Parallel Efficiency
    ax4 = axes[1, 1]
    ax4.plot(mpi_procs, mpi_efficiency, 's-', linewidth=2, markersize=8, 
             color='#F18F01', label='MPI Efficiency')
    ax4.axhline(y=100, color='red', linestyle='--', linewidth=2, label='Ideal (100%)')
    ax4.set_xlabel('Number of MPI Processes', fontsize=12)
    ax4.set_ylabel('Parallel Efficiency (%)', fontsize=12)
    ax4.set_title('MPI Parallel Efficiency', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 120])
    
    plt.tight_layout()
    plt.savefig('results/performance_analysis.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved: results/performance_analysis.png")
    
    # Generate summary table
    print("\n" + "="*60)
    print("MPI STRONG SCALING ANALYSIS")
    print("="*60)
    print(f"{'Procs':<8} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}")
    print("-"*60)
    for i, (p, t, s, e) in enumerate(zip(mpi_procs, mpi_times, mpi_speedup, mpi_efficiency)):
        print(f"{p:<8} {t:<12.2f} {s:<12.2f} {e:<15.1f}")
    print("="*60)
    
    # Save data to file
    with open('results/benchmark_results.txt', 'w') as f:
        f.write("VQE BENCHMARK RESULTS\n")
        f.write("="*60 + "\n\n")
        f.write("Runtime Summary:\n")
        f.write("-"*60 + "\n")
        for name, runtime in results.items():
            if runtime:
                speedup = serial_time / runtime
                f.write(f"{name:12s}: {runtime:8.2f}s  (Speedup: {speedup:6.2f}x)\n")
        f.write("\n" + "="*60 + "\n\n")
        f.write("MPI Strong Scaling:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Procs':<8} {'Time (s)':<12} {'Speedup':<12} {'Efficiency (%)':<15}\n")
        f.write("-"*60 + "\n")
        for p, t, s, e in zip(mpi_procs, mpi_times, mpi_speedup, mpi_efficiency):
            f.write(f"{p:<8} {t:<12.2f} {s:<12.2f} {e:<15.1f}\n")
        f.write("="*60 + "\n")
    
    print("\nResults saved: results/benchmark_results.txt")

if __name__ == '__main__':
    main()
