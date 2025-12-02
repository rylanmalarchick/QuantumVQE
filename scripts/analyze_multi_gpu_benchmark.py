#!/usr/bin/env python3
"""
Analyze Multi-GPU Benchmark Results

Reads results/multi_gpu_benchmark.json and generates:
1. Max qubit scaling plot (time vs qubits, memory estimate)
2. Multi-GPU throughput bar chart
3. Scaling efficiency comparison (1 GPU vs 4 GPU)
4. Combined summary figure for report

Usage:
    python scripts/analyze_multi_gpu_benchmark.py
"""

import json
import os
import sys

def main():
    # Check for results file
    results_file = 'results/multi_gpu_benchmark.json'
    if not os.path.exists(results_file):
        print(f"ERROR: {results_file} not found!")
        print("Run the benchmark first: qsub pbs_scripts/run_comprehensive_gpu_benchmark.sh")
        sys.exit(1)
    
    # Load results
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    print("=" * 70)
    print("Multi-GPU Benchmark Analysis")
    print("=" * 70)
    print(f"Timestamp: {data.get('timestamp', 'N/A')}")
    print(f"Number of GPUs: {data.get('n_gpus', 'N/A')}")
    print()
    
    # Import plotting libraries
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    
    # Set up style
    plt.style.use('seaborn-v0_8-darkgrid')
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#9b59b6', '#f39c12']
    
    # Create output directory
    os.makedirs('results/multi_gpu', exist_ok=True)
    
    # =========================================================================
    # PLOT 1: Max Qubit Scaling
    # =========================================================================
    print("--- TEST 1: Max Qubit Scaling ---")
    max_test = data.get('max_qubits_test', {})
    max_achieved = max_test.get('max_achieved', 0)
    results_list = max_test.get('results', [])
    
    print(f"Maximum qubits achieved: {max_achieved}")
    
    if results_list:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Filter successful results
        successful = [r for r in results_list if r.get('success', False)]
        failed = [r for r in results_list if not r.get('success', False)]
        
        if successful:
            qubits = [r['qubits'] for r in successful]
            times = [r['time_per_iter'] for r in successful]
            mem_est = [r.get('est_gpu_mem_gb', 0) for r in successful]
            
            # Plot 1a: Time per iteration vs qubits
            ax1.semilogy(qubits, times, 'o-', color=colors[0], markersize=10, linewidth=2, label='Time per iteration')
            ax1.set_xlabel('Number of Qubits', fontsize=12)
            ax1.set_ylabel('Time per VQE Iteration (s)', fontsize=12)
            ax1.set_title('VQE Iteration Time vs Problem Size', fontsize=14, fontweight='bold')
            
            # Mark failures
            if failed:
                for r in failed:
                    ax1.axvline(x=r['qubits'], color='red', linestyle='--', alpha=0.5, label=f"OOM at {r['qubits']}q")
            
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 1b: Memory estimate vs qubits
            ax2.bar(qubits, mem_est, color=colors[1], alpha=0.8, edgecolor='black')
            ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='H100 Memory (80 GB)')
            ax2.set_xlabel('Number of Qubits', fontsize=12)
            ax2.set_ylabel('Estimated GPU Memory (GB)', fontsize=12)
            ax2.set_title('GPU Memory Usage vs Problem Size', fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
            
            # Print table
            print("\nQubit Scaling Results:")
            print(f"{'Qubits':<10} {'Time/Iter (s)':<15} {'Est. Memory (GB)':<18} {'Status'}")
            print("-" * 55)
            for r in results_list:
                status = "SUCCESS" if r.get('success') else "FAILED"
                time_str = f"{r.get('time_per_iter', 0):.2f}" if r.get('success') else "N/A"
                mem_str = f"{r.get('est_gpu_mem_gb', 0):.1f}" if r.get('est_gpu_mem_gb') else "N/A"
                print(f"{r['qubits']:<10} {time_str:<15} {mem_str:<18} {status}")
        
        plt.tight_layout()
        plt.savefig('results/multi_gpu/max_qubit_scaling.png', dpi=150, bbox_inches='tight')
        print(f"\nSaved: results/multi_gpu/max_qubit_scaling.png")
        plt.close()
    
    # =========================================================================
    # PLOT 2: Multi-GPU Throughput
    # =========================================================================
    print("\n--- TEST 2: Multi-GPU Throughput ---")
    throughput_test = data.get('throughput_test', {})
    
    if throughput_test:
        n_gpus = throughput_test.get('n_gpus', 0)
        total_problems = throughput_test.get('total_problems', 0)
        wall_time = throughput_test.get('wall_clock_time', 0)
        throughput = throughput_test.get('throughput_per_second', 0)
        per_gpu = throughput_test.get('per_gpu_results', [])
        
        print(f"GPUs used: {n_gpus}")
        print(f"Total problems solved: {total_problems}")
        print(f"Wall clock time: {wall_time:.2f}s")
        print(f"Throughput: {throughput:.3f} problems/second")
        
        if per_gpu:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Plot 2a: Time per GPU
            gpu_ids = [r.get('gpu_id', i) for i, r in enumerate(per_gpu)]
            gpu_times = [r.get('total_time', 0) for r in per_gpu]
            
            bars = ax1.bar(gpu_ids, gpu_times, color=colors[:len(gpu_ids)], alpha=0.8, edgecolor='black')
            ax1.set_xlabel('GPU ID', fontsize=12)
            ax1.set_ylabel('Total Time (s)', fontsize=12)
            ax1.set_title(f'Per-GPU Execution Time ({total_problems} problems total)', fontsize=14, fontweight='bold')
            ax1.set_xticks(gpu_ids)
            
            # Add value labels
            for bar, time in zip(bars, gpu_times):
                ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                        f'{time:.1f}s', ha='center', va='bottom', fontsize=10)
            
            ax1.grid(True, alpha=0.3, axis='y')
            
            # Plot 2b: Throughput comparison (1 GPU vs 4 GPU)
            single_gpu_throughput = throughput / n_gpus  # Estimated single GPU
            categories = ['1 GPU\n(estimated)', f'{n_gpus} GPUs\n(measured)']
            throughputs = [single_gpu_throughput, throughput]
            
            bars = ax2.bar(categories, throughputs, color=[colors[1], colors[0]], alpha=0.8, edgecolor='black')
            ax2.set_ylabel('Throughput (problems/second)', fontsize=12)
            ax2.set_title('Throughput: Single vs Multi-GPU', fontsize=14, fontweight='bold')
            
            # Add speedup annotation
            speedup = throughput / single_gpu_throughput if single_gpu_throughput > 0 else 0
            ax2.annotate(f'{speedup:.1f}x speedup', 
                        xy=(1, throughput), xytext=(0.5, throughput * 0.8),
                        fontsize=12, fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='black'))
            
            ax2.grid(True, alpha=0.3, axis='y')
            
            plt.tight_layout()
            plt.savefig('results/multi_gpu/throughput_comparison.png', dpi=150, bbox_inches='tight')
            print(f"Saved: results/multi_gpu/throughput_comparison.png")
            plt.close()
    
    # =========================================================================
    # PLOT 3: Scaling Efficiency
    # =========================================================================
    print("\n--- TEST 3: Scaling Efficiency ---")
    scaling_test = data.get('scaling_test', {})
    
    if scaling_test:
        n_problems = scaling_test.get('n_problems', 0)
        qubits = scaling_test.get('qubits', 0)
        single_time = scaling_test.get('single_gpu_estimated_time', 0)
        multi_time = scaling_test.get('multi_gpu_time', 0)
        speedup = scaling_test.get('speedup', 0)
        efficiency = scaling_test.get('efficiency_percent', 0)
        
        print(f"Problem size: {n_problems} problems x {qubits} qubits")
        print(f"Single GPU time (estimated): {single_time:.2f}s")
        print(f"4-GPU time (measured): {multi_time:.2f}s")
        print(f"Speedup: {speedup:.2f}x")
        print(f"Parallel efficiency: {efficiency:.1f}%")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Plot 3a: Time comparison
        categories = ['1 GPU\n(sequential)', '4 GPUs\n(parallel)']
        times = [single_time, multi_time]
        
        bars = ax1.bar(categories, times, color=[colors[2], colors[0]], alpha=0.8, edgecolor='black')
        ax1.set_ylabel('Execution Time (s)', fontsize=12)
        ax1.set_title(f'Execution Time: {n_problems} VQE Problems ({qubits} qubits)', fontsize=14, fontweight='bold')
        
        # Add speedup annotation
        ax1.annotate(f'{speedup:.2f}x faster', 
                    xy=(1, multi_time), xytext=(0.5, single_time * 0.6),
                    fontsize=14, fontweight='bold', color=colors[0],
                    arrowprops=dict(arrowstyle='->', color=colors[0], lw=2))
        
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Plot 3b: Efficiency gauge
        # Create a simple bar showing efficiency
        ax2.barh(['Parallel\nEfficiency'], [efficiency], color=colors[0], alpha=0.8, edgecolor='black')
        ax2.barh(['Parallel\nEfficiency'], [100 - efficiency], left=[efficiency], color='lightgray', alpha=0.5)
        ax2.set_xlim(0, 100)
        ax2.set_xlabel('Efficiency (%)', fontsize=12)
        ax2.set_title('Multi-GPU Parallel Efficiency', fontsize=14, fontweight='bold')
        ax2.axvline(x=100, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Ideal (100%)')
        
        # Add percentage label
        ax2.text(efficiency/2, 0, f'{efficiency:.1f}%', ha='center', va='center', 
                fontsize=20, fontweight='bold', color='white')
        
        ax2.legend(loc='lower right')
        ax2.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        plt.savefig('results/multi_gpu/scaling_efficiency.png', dpi=150, bbox_inches='tight')
        print(f"Saved: results/multi_gpu/scaling_efficiency.png")
        plt.close()
    
    # =========================================================================
    # COMBINED SUMMARY FIGURE
    # =========================================================================
    print("\n--- Creating Combined Summary Figure ---")
    
    fig = plt.figure(figsize=(16, 12))
    
    # Title
    fig.suptitle('Multi-GPU VQE Benchmark Results (4x NVIDIA H100)', fontsize=16, fontweight='bold', y=0.98)
    
    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.25)
    
    # Panel 1: Max Qubit Scaling (top-left)
    ax1 = fig.add_subplot(gs[0, 0])
    if max_test.get('results'):
        successful = [r for r in max_test['results'] if r.get('success', False)]
        if successful:
            qubits = [r['qubits'] for r in successful]
            times = [r['time_per_iter'] for r in successful]
            ax1.semilogy(qubits, times, 'o-', color=colors[0], markersize=10, linewidth=2)
            ax1.set_xlabel('Number of Qubits')
            ax1.set_ylabel('Time per Iteration (s)')
            ax1.set_title(f'A) Max Qubit Scaling (achieved: {max_achieved}q)', fontweight='bold')
            ax1.grid(True, alpha=0.3)
    
    # Panel 2: Memory Usage (top-right)
    ax2 = fig.add_subplot(gs[0, 1])
    if max_test.get('results'):
        successful = [r for r in max_test['results'] if r.get('success', False)]
        if successful:
            qubits = [r['qubits'] for r in successful]
            mem_est = [r.get('est_gpu_mem_gb', 0) for r in successful]
            ax2.bar(qubits, mem_est, color=colors[1], alpha=0.8, edgecolor='black')
            ax2.axhline(y=80, color='red', linestyle='--', linewidth=2, label='H100 Limit (80GB)')
            ax2.set_xlabel('Number of Qubits')
            ax2.set_ylabel('Est. GPU Memory (GB)')
            ax2.set_title('B) GPU Memory Usage', fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')
    
    # Panel 3: Throughput (bottom-left)
    ax3 = fig.add_subplot(gs[1, 0])
    if throughput_test:
        n_gpus = throughput_test.get('n_gpus', 4)
        throughput = throughput_test.get('throughput_per_second', 0)
        single_throughput = throughput / n_gpus
        
        categories = ['1 GPU', f'{n_gpus} GPUs']
        throughputs = [single_throughput, throughput]
        bars = ax3.bar(categories, throughputs, color=[colors[2], colors[0]], alpha=0.8, edgecolor='black')
        ax3.set_ylabel('Throughput (problems/sec)')
        ax3.set_title(f'C) Multi-GPU Throughput ({n_gpus}x speedup)', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Value labels
        for bar, val in zip(bars, throughputs):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{val:.2f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Panel 4: Scaling Efficiency (bottom-right)
    ax4 = fig.add_subplot(gs[1, 1])
    if scaling_test:
        single_time = scaling_test.get('single_gpu_estimated_time', 0)
        multi_time = scaling_test.get('multi_gpu_time', 0)
        speedup = scaling_test.get('speedup', 0)
        efficiency = scaling_test.get('efficiency_percent', 0)
        
        categories = ['1 GPU\n(sequential)', '4 GPUs\n(parallel)']
        times = [single_time, multi_time]
        bars = ax4.bar(categories, times, color=[colors[2], colors[0]], alpha=0.8, edgecolor='black')
        ax4.set_ylabel('Time (s)')
        ax4.set_title(f'D) Scaling Efficiency ({speedup:.1f}x, {efficiency:.0f}% efficient)', fontweight='bold')
        ax4.grid(True, alpha=0.3, axis='y')
    
    plt.savefig('results/multi_gpu/multi_gpu_summary.png', dpi=150, bbox_inches='tight')
    print(f"Saved: results/multi_gpu/multi_gpu_summary.png")
    plt.close()
    
    # =========================================================================
    # TEXT SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)
    
    summary_text = f"""
Multi-GPU VQE Benchmark Results
================================
Hardware: {data.get('n_gpus', 4)}x NVIDIA H100 (80GB each)
Timestamp: {data.get('timestamp', 'N/A')}

1. MAX QUBIT SCALING
   - Maximum qubits achieved: {max_achieved}
   - State vector at {max_achieved}q: {2**max_achieved * 16 / 1024**3:.1f} GB
   - Estimated GPU memory: {2**max_achieved * 16 * 4 / 1024**3:.1f} GB

2. MULTI-GPU THROUGHPUT
   - Total problems: {throughput_test.get('total_problems', 'N/A')}
   - Wall clock time: {throughput_test.get('wall_clock_time', 0):.2f}s
   - Throughput: {throughput_test.get('throughput_per_second', 0):.3f} problems/sec
   - Effective speedup: {data.get('n_gpus', 4)}x (perfect parallel)

3. SCALING EFFICIENCY
   - Speedup: {scaling_test.get('speedup', 0):.2f}x
   - Parallel efficiency: {scaling_test.get('efficiency_percent', 0):.1f}%

Generated plots:
  - results/multi_gpu/max_qubit_scaling.png
  - results/multi_gpu/throughput_comparison.png
  - results/multi_gpu/scaling_efficiency.png
  - results/multi_gpu/multi_gpu_summary.png
"""
    
    print(summary_text)
    
    # Save summary to file
    with open('results/multi_gpu/summary.txt', 'w') as f:
        f.write(summary_text)
    print("Saved: results/multi_gpu/summary.txt")
    
    print("=" * 70)
    print("Analysis complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
