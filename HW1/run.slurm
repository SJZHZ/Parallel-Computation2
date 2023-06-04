#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 1:00

echo serial
./serial "input2.txt"
echo destination
./destination "input2.txt"
echo cache
./cache "input2.txt"
# ./persistent "input2.txt"
echo pathway
./pathway "input2.txt"
echo pathway_cache
./pathway_cache "input2.txt"
