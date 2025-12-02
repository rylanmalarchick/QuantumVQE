#!/bin/bash
#PBS -N comprehensive_gpu_bench
#PBS -l nodes=1:ppn=8:gpus=4
#PBS -l walltime=02:00:00
#PBS -q shortq
#PBS -o logs/comprehensive_gpu_bench_output.log
#PBS -e logs/comprehensive_gpu_bench_error.log

# Comprehensive Multi-GPU Benchmark
# Tests:
#   1. Max qubit scaling (find H100 memory limit: 26-31 qubits)
#   2. 4-GPU throughput (problems/second)
#   3. Scaling efficiency (1 GPU vs 4 GPU)

cd $PBS_O_WORKDIR

# Create logs and results directories
mkdir -p logs
mkdir -p results

# Activate conda environment
eval "$(/apps/spack/opt/spack/linux-rocky8-zen4/gcc-13.2.0/anaconda3-2023.09-0-3wl2qheo6tntdwtbjdmvouw24zd4rugj/bin/conda shell.bash hook)"
conda activate vqe-gpu

echo "========================================"
echo "Comprehensive Multi-GPU Benchmark"
echo "========================================"
echo "Job ID: $PBS_JOBID"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "Host: $(hostname)"
echo ""
echo "Python: $(which python)"
echo "mpiexec: $(which mpiexec 2>/dev/null || echo 'not found')"
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""
echo "========================================"
echo ""

# Run 4 MPI ranks, one per GPU
echo "Running benchmark with 4 GPUs..."
mpiexec -n 4 python src/scaling_study/comprehensive_gpu_benchmark.py

echo ""
echo "========================================"
echo "Benchmark complete"
echo "End time: $(date)"
echo "========================================"

# Show results file if created
if [ -f results/multi_gpu_benchmark.json ]; then
    echo ""
    echo "Results saved to: results/multi_gpu_benchmark.json"
    echo ""
    echo "Quick preview:"
    cat results/multi_gpu_benchmark.json
fi
