#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 1:00
#SBATCH -w cu08
export OMP_NUM_THREADS=12

echo serial
./serial "input2.txt"
./check

echo destination
./destination "input2.txt"
./check

echo pathway
./pathway "input2.txt"
./check

echo pathway_cache
./pathway_cache "input2.txt"
./check

echo collapse
./collapse "input2.txt"
./check

echo persistent without-O3
./persistent "input2.txt"
./check

echo persistent_T
./persistent_T "input2.txt"
./check

echo final:persistent
./final "input2.txt"
./check