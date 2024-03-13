#!/bin/sh
#SBATCH --partition=general
#SBATCH --job-name=dima_diffusion
#SBATCH --qos=long         
#SBATCH --time=168:00:00      
#SBATCH --ntasks=1         
#SBATCH --cpus-per-task=2   
#SBATCH --mem-per-cpu=512G
#SBATCH --mail-type=END     
#SBATCH --gres=gpu

module use /opt/insy/modulefiles
module load cuda/12.2 cudnn/12-8.9.1.23 miniconda/3.10

conda activate /tudelft.net/staff-umbrella/Mehul/DiMA/dima

export TMPDIR=./tmp/

srun torchrun --nproc_per_node=1 --master_port=31345  train_diffusion.py