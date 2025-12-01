#!/bin/bash
#PBS -N vqe_scaling_study
#PBS -l select=1:ncpus=32:mem=256gb:ngpus=1
#PBS -l walltime=04:00:00
#PBS -q gpu
#PBS -o logs/scaling_study_output.log
#PBS -e logs/scaling_study_error.log

# ============================================================================
# VQE Scaling Study: CPU+JIT vs GPU Performance
# ============================================================================
# This script benchmarks VQE across multiple qubit counts to find the
# crossover point where GPU outperforms CPU+JIT.
#
# Hardware:
#   - CPU: AMD EPYC 9654 (192 cores, 1.5TB RAM)
#   - GPU: NVIDIA H100 PCIe (81GB)
#
# Expected runtime: 2-4 hours depending on max qubits
# ============================================================================

echo "============================================"
echo "VQE Scaling Study"
echo "============================================"
echo "Start time: $(date)"
echo "Job ID: $PBS_JOBID"
echo "Running on node: $(hostname)"
echo "Working directory: $PBS_O_WORKDIR"
echo "============================================"

# Change to working directory
cd $PBS_O_WORKDIR

# Create logs directory if needed
mkdir -p logs
mkdir -p results/scaling_study

# Load CUDA module
module load cuda/12.4.0-gcc-13.2.0-ym45qpm

# Print GPU info
echo ""
echo "GPU Info:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
echo ""
nvidia-smi
echo ""

# ============================================================================
# Environment Setup
# ============================================================================
# We need the vqe-lightning-gpu environment which has:
#   - pennylane
#   - pennylane-lightning-gpu
#   - optax
#   - jax
#   - matplotlib
# ============================================================================

source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/.conda/etc/profile.d/conda.sh 2>/dev/null

# Use the lightning-gpu environment (has lightning.gpu without Catalyst conflict)
conda activate vqe-lightning-gpu

echo "============================================"
echo "Python environment:"
which python
python --version
echo ""
echo "Key packages:"
python -c "import pennylane as qml; print(f'PennyLane: {qml.__version__}')"
python -c "import jax; print(f'JAX: {jax.__version__}')"
python -c "import optax; print(f'Optax: {optax.__version__}')"
echo "============================================"

# ============================================================================
# Run Scaling Study
# ============================================================================
# Default: Test 4 to 28 qubits
# Adjust --max-qubits if you want to push further (may OOM at 30+)
# ============================================================================

echo ""
echo "Starting scaling study..."
echo "============================================"

# Add src directory to Python path
export PYTHONPATH="${PBS_O_WORKDIR}/src/scaling_study:${PYTHONPATH}"

# Run the scaling study
python src/scaling_study/run_scaling_study.py \
    --max-qubits 28 \
    --output results/scaling_study

EXIT_CODE=$?

echo ""
echo "============================================"
echo "Scaling study complete!"
echo "Exit code: $EXIT_CODE"
echo "End time: $(date)"
echo "============================================"

# List output files
echo ""
echo "Output files:"
ls -la results/scaling_study/

exit $EXIT_CODE
