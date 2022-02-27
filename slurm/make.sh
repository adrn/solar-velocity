#!/bin/bash
#SBATCH -J solaroid
#SBATCH -o logs/make.o%j
#SBATCH -e logs/make.e%j
#SBATCH -N 1
#SBATCH -t 04:00:00
#SBATCH -p cca

source ~/.bash_profile
cd /mnt/ceph/users/apricewhelan/projects/solar-velocity
init_conda
conda activate solaroid

date

make
make sgrA_star
make sgrA_star_combine
make basis_funcs

date
