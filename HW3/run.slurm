#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J parco2023_hw3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 3:00
#SBATCH -w cu04

export OMP_NUM_THREADS=1
# ./gaussian_blur
# ./gaussian_blur
# ./gaussian_blur
# ./gaussian_blur
# ./gaussian_blur

export OMP_NUM_THREADS=12
echo "TEST TIME * 5"
./gaussian_blur_try
./gaussian_blur_try
./gaussian_blur_try
./gaussian_blur_try
./gaussian_blur_try