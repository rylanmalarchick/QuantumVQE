#!/bin/bash
#PBS -N quantum_job
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -o output.log
#PBS -e error.log

cd $PBS_O_WORKDIR
source activate vqe-hpc-env 
python submit_job.py



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%#

