#!/bin/bash
#PBS -N vqe_serial
#PBS -l nodes=1:ppn=1
#PBS -l walltime=02:00:00
#PBS -o logs/serial_output.log
#PBS -e logs/serial_error.log
#PBS -m abe
#PBS -M rylan@example.com

# Serial VQE Baseline - Single CPU core

cd $PBS_O_WORKDIR

# Load micromamba and activate GPU environment
eval "$(./bin/micromamba shell hook --shell bash)"
micromamba activate vqe-gpu

echo "============================================"
echo "Serial VQE Baseline Run"
echo "============================================"
echo "Start time: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"
echo "Python: $(which python)"
echo "============================================"

# Run the serial code
python src/main.py

echo "============================================"
echo "End time: $(date)"
echo "============================================"
