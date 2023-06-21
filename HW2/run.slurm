#!/bin/bash
#SBATCH -o job_%j_%N.out
#SBATCH -p cpu
#SBATCH -J hw2
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH -t 3:00
#SBATCH --nodelist cu05,cu06,cu07,cu08


echo -----STANDARD-----
mpirun -np 1 ./attention "input2.in"

export OMP_NUM_THREADS=1
echo -----CHECK-----
echo INPUT1
mpirun -n 16 ./attention_try "input1.in"
echo INPUT2
mpirun -n 16 ./attention_try "input2.in"

echo -e "\n-----TEST TIME by input2 * 5-----"
echo 1
mpirun -n 16 ./attention_try "input2.in"
echo 2
mpirun -n 16 ./attention_try "input2.in"
echo 3
mpirun -n 16 ./attention_try "input2.in"
echo 4
mpirun -n 16 ./attention_try "input2.in"
echo 5
mpirun -n 16 ./attention_try "input2.in"