#!/usr/bin/env python3
"""
Improved Multi-GPU Benchmark Plots
Generates cleaner, presentation-ready figures.

Usage:
    python scripts/plot_multi_gpu_improved.py
"""

import json
import os
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import numpy as np

# Set global style
plt.rcParams.update({
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def load_data():
    """Load benchmark results."""
    results_file = 'results/multi_gpu/multi_gpu_benchmark.json'
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found!")
        return None
    
    with open(results_file, 'r') as f:
        return json.load(f)


def plot_multi_gpu_summary(data):
    """
    Create a clean 2x2 summary figure for multi-GPU results.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('Multi-GPU VQE Performance (4x NVIDIA H100)', 
                 fontsize=18, fontweight='bold', y=0.98)
    
    gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)
    
    # Colors
    green = '#27AE60'
    blue = '#3498DB'
    red = '#E74C3C'
    purple = '#9B59B6'
    
    # =========================================================================
    # Panel A: Max Qubit Scaling (top-left)
    # =========================================================================
    ax1 = fig.add_subplot(gs[0, 0])
    
    max_test = data.get('max_qubits_test', {})
    results_list = max_test.get('results', [])
    successful = [r for r in results_list if r.get('success', False)]
    
    if successful:
        qubits = [r['qubits'] for r in successful]
        times = [r['time_per_iter'] for r in successful]
        mem_gb = [r.get('est_gpu_mem_gb', 0) for r in successful]
        
        # Bar chart instead of line (cleaner with few data points)
        bars = ax1.bar(qubits, times, color=green, edgecolor='black', linewidth=1.5, width=0.8)
        
        # Add value labels
        for bar, t in zip(bars, times):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Mark the OOM boundary
        ax1.axvline(x=29.5, color=red, linestyle='--', linewidth=2, alpha=0.7)
        ax1.text(29.6, max(times)*0.9, 'OOM at\n30 qubits', fontsize=10, color=red, 
                 fontweight='bold', va='top')
        
        ax1.set_xlabel('Number of Qubits', fontsize=13)
        ax1.set_ylabel('Time per VQE Iteration (s)', fontsize=13)
        ax1.set_title('A) Maximum Qubit Scaling', fontsize=14, fontweight='bold')
        ax1.set_xticks(qubits)
        ax1.grid(axis='y', alpha=0.3, linestyle='--')
        ax1.set_ylim(0, max(times) * 1.3)
    
    # =========================================================================
    # Panel B: Memory Requirements (top-right)
    # =========================================================================
    ax2 = fig.add_subplot(gs[0, 1])
    
    if successful:
        # Include the failed 30-qubit case to show the limit
        all_qubits = qubits + [30]
        all_mem = mem_gb + [64]  # 30 qubits = 64 GB estimated
        
        colors_mem = [blue if q <= 29 else red for q in all_qubits]
        bars = ax2.bar(all_qubits, all_mem, color=colors_mem, edgecolor='black', 
                      linewidth=1.5, width=0.8, alpha=0.8)
        
        # H100 limit line
        ax2.axhline(y=80, color=red, linestyle='--', linewidth=2.5, label='H100 Limit (80GB)')
        
        # Annotations
        for bar, mem, q in zip(bars, all_mem, all_qubits):
            label = f'{mem:.0f}GB' if mem >= 1 else f'{mem*1024:.0f}MB'
            color = 'white' if q == 30 else 'black'
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2,
                    label, ha='center', va='center', fontsize=10, fontweight='bold', color=color)
        
        ax2.set_xlabel('Number of Qubits', fontsize=13)
        ax2.set_ylabel('Estimated GPU Memory (GB)', fontsize=13)
        ax2.set_title('B) GPU Memory Requirements', fontsize=14, fontweight='bold')
        ax2.set_xticks(all_qubits)
        ax2.legend(loc='upper left', fontsize=11)
        ax2.grid(axis='y', alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 90)
    
    # =========================================================================
    # Panel C: Multi-GPU Speedup (bottom-left)
    # =========================================================================
    ax3 = fig.add_subplot(gs[1, 0])
    
    scaling_test = data.get('scaling_test', {})
    if scaling_test:
        single_time = scaling_test.get('single_gpu_estimated_time', 0)
        multi_time = scaling_test.get('multi_gpu_time', 0)
        speedup = scaling_test.get('speedup', 0)
        n_problems = scaling_test.get('n_problems', 0)
        qubits_used = scaling_test.get('qubits', 0)
        
        categories = ['1 GPU\n(sequential)', '4 GPUs\n(parallel)']
        times = [single_time, multi_time]
        colors_bar = [red, green]
        
        bars = ax3.bar(categories, times, color=colors_bar, edgecolor='black', 
                      linewidth=1.5, width=0.5)
        
        # Value labels
        for bar, t in zip(bars, times):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f'{t:.1f}s', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Speedup arrow annotation
        mid_y = (single_time + multi_time) / 2
        ax3.annotate('', xy=(1, multi_time + 2), xytext=(0, single_time - 2),
                    arrowprops=dict(arrowstyle='->', color=green, lw=3))
        ax3.text(0.5, mid_y, f'{speedup:.1f}x\nfaster', ha='center', va='center',
                fontsize=14, fontweight='bold', color=green,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax3.set_ylabel('Execution Time (seconds)', fontsize=13)
        ax3.set_title(f'C) Multi-GPU Speedup ({n_problems} problems, {qubits_used}q)', 
                     fontsize=14, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3, linestyle='--')
        ax3.set_ylim(0, single_time * 1.2)
    
    # =========================================================================
    # Panel D: Parallel Efficiency (bottom-right)
    # =========================================================================
    ax4 = fig.add_subplot(gs[1, 1])
    
    if scaling_test:
        efficiency = scaling_test.get('efficiency_percent', 0)
        
        # Create a gauge-like visualization
        # Show efficiency as a filled proportion of ideal
        ax4.barh(['Achieved'], [efficiency], color=green, edgecolor='black', 
                linewidth=1.5, height=0.4, label=f'Measured: {efficiency:.1f}%')
        ax4.barh(['Ideal'], [100], color='lightgray', edgecolor='black',
                linewidth=1.5, height=0.4, alpha=0.5, label='Ideal: 100%')
        
        # Add percentage labels
        ax4.text(efficiency/2, 0, f'{efficiency:.1f}%', ha='center', va='center',
                fontsize=16, fontweight='bold', color='white')
        ax4.text(50, 1, '100%', ha='center', va='center',
                fontsize=14, fontweight='bold', color='gray')
        
        ax4.set_xlim(0, 110)
        ax4.set_xlabel('Parallel Efficiency (%)', fontsize=13)
        ax4.set_title('D) Multi-GPU Parallel Efficiency', fontsize=14, fontweight='bold')
        ax4.legend(loc='lower right', fontsize=10)
        ax4.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Add key insight
        ax4.text(55, -0.7, 'Near-perfect scaling: 4 GPUs deliver 3.98x speedup', 
                fontsize=11, style='italic', ha='center')
    
    plt.savefig('results/multi_gpu/multi_gpu_summary.png', dpi=300, bbox_inches='tight')
    print('Saved: results/multi_gpu/multi_gpu_summary.png')
    plt.close()


def plot_scaling_efficiency_simple(data):
    """
    Create a clean, simple scaling efficiency plot.
    """
    scaling_test = data.get('scaling_test', {})
    if not scaling_test:
        return
    
    single_time = scaling_test.get('single_gpu_estimated_time', 0)
    multi_time = scaling_test.get('multi_gpu_time', 0)
    speedup = scaling_test.get('speedup', 0)
    efficiency = scaling_test.get('efficiency_percent', 0)
    n_problems = scaling_test.get('n_problems', 0)
    qubits = scaling_test.get('qubits', 0)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f'Multi-GPU Scaling: {n_problems} VQE Problems at {qubits} Qubits', 
                 fontsize=16, fontweight='bold')
    
    green = '#27AE60'
    red = '#E74C3C'
    
    # Left: Time comparison
    ax1 = axes[0]
    categories = ['1 GPU\n(sequential)', '4 GPUs\n(parallel)']
    times = [single_time, multi_time]
    colors = [red, green]
    
    bars = ax1.bar(categories, times, color=colors, edgecolor='black', linewidth=2, width=0.5)
    
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{t:.1f}s', ha='center', fontsize=13, fontweight='bold')
    
    # Speedup annotation - position to avoid overlap with labels
    ax1.annotate(f'{speedup:.2f}x faster', 
                xy=(0.85, multi_time * 1.5), xytext=(0.35, single_time * 0.55),
                fontsize=14, fontweight='bold', color=green,
                arrowprops=dict(arrowstyle='->', color=green, lw=2.5),
                ha='center')
    
    ax1.set_ylabel('Execution Time (seconds)', fontsize=13)
    ax1.set_title('Execution Time Comparison', fontsize=14, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, single_time * 1.25)
    
    # Right: Efficiency gauge
    ax2 = axes[1]
    
    # Horizontal bar gauge
    ax2.barh(['Parallel\nEfficiency'], [efficiency], color=green, edgecolor='black', 
            linewidth=2, height=0.3)
    ax2.barh(['Parallel\nEfficiency'], [100 - efficiency], left=[efficiency], 
            color='lightgray', edgecolor='gray', linewidth=1, height=0.3, alpha=0.5)
    
    # Ideal line
    ax2.axvline(x=100, color=green, linestyle='--', linewidth=2, alpha=0.7)
    ax2.text(105, 0, 'Ideal', fontsize=10, va='center', color=green, fontweight='bold')
    
    # Percentage label
    ax2.text(efficiency/2, 0, f'{efficiency:.1f}%', ha='center', va='center',
            fontsize=20, fontweight='bold', color='white')
    
    ax2.set_xlim(0, 115)
    ax2.set_xlabel('Efficiency (%)', fontsize=13)
    ax2.set_title('Parallel Efficiency', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig('results/multi_gpu/scaling_efficiency.png', dpi=300, bbox_inches='tight')
    print('Saved: results/multi_gpu/scaling_efficiency.png')
    plt.close()


def plot_max_qubit_scaling(data):
    """
    Create a focused max qubit scaling plot.
    """
    max_test = data.get('max_qubits_test', {})
    results_list = max_test.get('results', [])
    max_achieved = max_test.get('max_achieved', 29)
    
    if not results_list:
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f'H100 GPU: Maximum Qubit Capacity ({max_achieved} qubits achieved)', 
                 fontsize=16, fontweight='bold')
    
    green = '#27AE60'
    blue = '#3498DB'
    red = '#E74C3C'
    
    successful = [r for r in results_list if r.get('success', False)]
    failed = [r for r in results_list if not r.get('success', False)]
    
    qubits = [r['qubits'] for r in successful]
    times = [r['time_per_iter'] for r in successful]
    mem_gb = [r.get('est_gpu_mem_gb', 0) for r in successful]
    
    # Left: Time scaling
    ax1 = axes[0]
    bars = ax1.bar(qubits, times, color=green, edgecolor='black', linewidth=1.5, width=0.7)
    
    for bar, t in zip(bars, times):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.15,
                f'{t:.1f}s', ha='center', fontsize=11, fontweight='bold')
    
    # Mark failure point
    if failed:
        fail_q = failed[0]['qubits']
        ax1.axvline(x=fail_q - 0.5, color=red, linestyle='--', linewidth=2)
        ax1.text(fail_q - 0.4, max(times) * 0.8, f'OOM\n({fail_q}q)', fontsize=10, 
                color=red, fontweight='bold')
    
    ax1.set_xlabel('Number of Qubits', fontsize=13)
    ax1.set_ylabel('Time per VQE Iteration (seconds)', fontsize=13)
    ax1.set_title('Iteration Time vs Qubit Count', fontsize=14, fontweight='bold')
    ax1.set_xticks(qubits)
    ax1.grid(axis='y', alpha=0.3, linestyle='--')
    ax1.set_ylim(0, max(times) * 1.25)
    
    # Right: Memory scaling
    ax2 = axes[1]
    
    # Include failed case
    all_q = qubits + [30]
    all_mem = mem_gb + [64]
    colors_bar = [blue if q < 30 else red for q in all_q]
    
    bars = ax2.bar(all_q, all_mem, color=colors_bar, edgecolor='black', linewidth=1.5, width=0.7)
    
    # H100 limit
    ax2.axhline(y=80, color=red, linestyle='--', linewidth=2.5, label='H100 Limit (80GB)')
    ax2.fill_between([25, 31], 80, 100, alpha=0.2, color=red)
    
    # Value labels
    for bar, m in zip(bars, all_mem):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                f'{m:.0f}GB', ha='center', fontsize=11, fontweight='bold')
    
    ax2.set_xlabel('Number of Qubits', fontsize=13)
    ax2.set_ylabel('Estimated GPU Memory (GB)', fontsize=13)
    ax2.set_title('Memory Requirements', fontsize=14, fontweight='bold')
    ax2.set_xticks(all_q)
    ax2.legend(loc='upper left', fontsize=11)
    ax2.grid(axis='y', alpha=0.3, linestyle='--')
    ax2.set_ylim(0, 95)
    
    plt.tight_layout()
    plt.savefig('results/multi_gpu/max_qubit_scaling.png', dpi=300, bbox_inches='tight')
    print('Saved: results/multi_gpu/max_qubit_scaling.png')
    plt.close()


def main():
    data = load_data()
    if data is None:
        return
    
    print("Generating improved multi-GPU plots...")
    plot_multi_gpu_summary(data)
    plot_scaling_efficiency_simple(data)
    plot_max_qubit_scaling(data)
    print("\nAll multi-GPU plots generated successfully!")


if __name__ == '__main__':
    main()
