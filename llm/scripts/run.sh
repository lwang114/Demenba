#!/bin/bash

if [ ! -d logs ]; then
    mkdir -p logs
fi

#sbatch run_fhs_text.slurm 2400 0 400
#sbatch run_fhs_text.slurm 2400 400 800
#sbatch run_fhs_text.slurm 2400 800 1200
#sbatch run_fhs_text.slurm 2400 1200 1600
#sbatch run_fhs_text.slurm 2400 1600 2000
#sbatch run_fhs_text.slurm 2400 2000 2400
sbatch run_fhs_text.slurm 200 0 200
