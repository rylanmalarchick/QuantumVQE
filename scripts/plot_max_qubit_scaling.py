#!/usr/bin/env python3
"""
Plot the single-H100 maximum-size test from results/multi_gpu/multi_gpu_benchmark.json:
measured time per iteration and estimated GPU memory against qubit count.

Usage:
    python scripts/plot_max_qubit_scaling.py
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
    
    plot_max_qubit_scaling(data)


if __name__ == '__main__':
    main()
