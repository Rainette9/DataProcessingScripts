#!/bin/bash
#SBATCH --job-name=MRD_script
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=10G
#SBATCH --account=s1329
#SBATCH --mail-user=rainette.engbers@slf.ch
#SBATCH --uenv=prgenv-gnu/24.11:v2
#SBATCH --partition=debug
#SBATCH --time=00:30:00
#SBATCH --output=MRD_script_%j.out
#SBATCH --error=MRD_script_%j.err


# Activate virtual environment if needed
source ~/py312_env/bin/activate

# Run the Python script
python ./notebooks/MRD_script.py
