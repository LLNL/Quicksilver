#!/bin/bash

# Quicksilver CTS Benchmark
# weak scaling on a single node:

# 1 rank per core 
# 4096 mesh elements per rank (16^3)
# 10 particles per mesh element -> 40960 particles per rank

export OMP_NUM_THREADS=1

QS=../../src/qs

srun -N1 -n1  $QS -i CTS2.inp -X 16  -Y 16  -Z 16  -x 16  -y 16  -z 16  -I 1  -J 1  -K 1  -n 40960      > CTS2_01.out
srun -N1 -n2  $QS -i CTS2.inp -X 32  -Y 16  -Z 16  -x 32  -y 16  -z 16  -I 2  -J 1  -K 1  -n 81920      > CTS2_02.out
srun -N1 -n4  $QS -i CTS2.inp -X 32  -Y 32  -Z 16  -x 32  -y 32  -z 16  -I 2  -J 2  -K 1  -n 163840     > CTS2_04.out
srun -N1 -n8  $QS -i CTS2.inp -X 32  -Y 32  -Z 32  -x 32  -y 32  -z 32  -I 2  -J 2  -K 2  -n 327680     > CTS2_08.out
srun -N1 -n16 $QS -i CTS2.inp -X 64  -Y 32  -Z 32  -x 64  -y 32  -z 32  -I 4  -J 2  -K 2  -n 655360     > CTS2_16.out
srun -N1 -n32 $QS -i CTS2.inp -X 64  -Y 64  -Z 32  -x 64  -y 64  -z 32  -I 4  -J 4  -K 2  -n 1310720    > CTS2_32.out
srun -N1 -n36 $QS -i CTS2.inp -X 48  -Y 48  -Z 64  -x 48  -y 48  -z 64  -I 3  -J 3  -K 4  -n 1474560    > CTS2_36.out

