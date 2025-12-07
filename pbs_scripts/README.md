# PBS Job Scripts

This directory contains PBS job scripts for running VQE benchmarks on the HPC cluster.

## Available Scripts

### 1. Serial Baseline
```bash
qsub pbs_scripts/run_serial.sh
```
- Runs `src/main_study/main.py` on a single CPU core
- Baseline performance metric
- Output: `logs/serial_output.log`

### 2. JIT-compiled GPU Version
```bash
qsub pbs_scripts/run_gpu.sh
```
- Runs `src/main_study/vqe_qjit.py` with GPU acceleration
- Uses Catalyst JIT compilation + CUDA
- Requires: 1 GPU node
- Output: `logs/gpu_output.log`

### 3. MPI Parallel Version
First, generate the MPI scripts for different process counts:
```bash
bash pbs_scripts/generate_mpi_scripts.sh
```

This creates scripts for 2, 4, 8, 16, and 32 processes. Then submit them:
```bash
qsub pbs_scripts/run_mpi_2.sh
qsub pbs_scripts/run_mpi_4.sh
qsub pbs_scripts/run_mpi_8.sh
qsub pbs_scripts/run_mpi_16.sh
qsub pbs_scripts/run_mpi_32.sh
```
- Output: `logs/mpi_N_output.log` (where N is the process count)

## Before Submitting Jobs

1. **Create logs directory:**
```bash
mkdir -p logs
```

2. **Update email in PBS scripts** (optional):
Edit the `#PBS -M` line in each script to use your email address.

3. **Test locally first:**
```bash
# Activate environment
conda activate vqe-gpu

# Test serial version
python src/main_study/main.py

# Test GPU version (on GPU node)
python src/main_study/vqe_qjit.py

# Test MPI version (on compute node)
mpiexec -n 4 python src/main_study/vqe_mpi.py
```

## Monitoring Jobs

Check job status:
```bash
qstat -u $USER
```

View job details:
```bash
qstat -f <job_id>
```

Cancel a job:
```bash
qdel <job_id>
```

View output in real-time:
```bash
tail -f logs/serial_output.log
```

## Expected Runtimes

- Serial: ~50-60 seconds (baseline)
- GPU/JIT: Expected ~10-20x speedup
- MPI (N processes): Expected ~N/2 to N speedup (depends on overhead)

## Troubleshooting

If jobs fail:
1. Check error logs in `logs/`
2. Verify environment is activated correctly
3. Test the code interactively on a compute node first
4. Check available resources: `pbsnodes -a`
