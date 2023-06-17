#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 3:00
#SBATCH --nodelist cu05,cu04,cu03,cu02

echo "basic"
mpirun -np 1 ./attention "input1.in"

echo "final"
mpirun -n 16 ./attention_try "input2.in"