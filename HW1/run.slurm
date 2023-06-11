#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=12
#SBATCH -t 1:00
#SBATCH -w cu06
export OMP_NUM_THREADS=12
target="input1.txt"

echo serial
./serial $target
mv ./output.txt ./output_std.txt

# echo collapse
# ./collapse $target
# ./check

# echo destination
# ./destination $target
# ./check

# echo cache
# ./cache $target
# ./check

# echo pathway
# ./pathway "input2.txt"
# ./check

# echo pathway_cache
# ./pathway_cache "input2.txt"
# ./check

# echo persistent without Optimization
# ./persistent "input2.txt"
# ./check

# echo persistent
# ./persistent_O "input2.txt"
# ./check

# echo persistent_T without Optimization
# ./persistent_T_without "input2.txt"
# ./check

echo -e "\nFINAL":
echo persistent_T
./persistent_T $target
./check