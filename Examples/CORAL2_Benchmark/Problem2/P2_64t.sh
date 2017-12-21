#!/bin/bash

#Problem 2:

# 1 rank per node 
# 64 threads per rank
# 1311 mesh elements per node (11^3)
# 40 particles per mesh element -> 53240 particles per node

export OMP_NUM_THREADS=64

QS=../../../src/qs

srun -N1     -n1     $QS -i Coral2_P2.inp -X 1  -Y 1  -Z 1  -x 11  -y 11  -z 11  -I 1  -J 1  -K 1  -n 53240      > p2n00001t64
srun -N2     -n2     $QS -i Coral2_P2.inp -X 2  -Y 1  -Z 1  -x 22  -y 11  -z 11  -I 2  -J 1  -K 1  -n 106480     > p2n00002t64
srun -N4     -n4     $QS -i Coral2_P2.inp -X 2  -Y 2  -Z 1  -x 22  -y 22  -z 11  -I 2  -J 2  -K 1  -n 212960     > p2n00004t64
srun -N8     -n8     $QS -i Coral2_P2.inp -X 2  -Y 2  -Z 2  -x 22  -y 22  -z 22  -I 2  -J 2  -K 2  -n 425920     > p2n00008t64
srun -N16    -n16    $QS -i Coral2_P2.inp -X 4  -Y 2  -Z 2  -x 44  -y 22  -z 22  -I 4  -J 2  -K 2  -n 851840     > p2n00016t64
srun -N32    -n32    $QS -i Coral2_P2.inp -X 4  -Y 4  -Z 2  -x 44  -y 44  -z 22  -I 4  -J 4  -K 2  -n 1703680    > p2n00032t64
srun -N64    -n64    $QS -i Coral2_P2.inp -X 4  -Y 4  -Z 4  -x 44  -y 44  -z 44  -I 4  -J 4  -K 4  -n 3407360    > p2n00064t64
srun -N128   -n128   $QS -i Coral2_P2.inp -X 8  -Y 4  -Z 4  -x 88  -y 44  -z 44  -I 8  -J 4  -K 4  -n 6814720    > p2n00128t64
srun -N256   -n256   $QS -i Coral2_P2.inp -X 8  -Y 8  -Z 4  -x 88  -y 88  -z 44  -I 8  -J 8  -K 4  -n 13629440   > p2n00256t64
srun -N512   -n512   $QS -i Coral2_P2.inp -X 8  -Y 8  -Z 8  -x 88  -y 88  -z 88  -I 8  -J 8  -K 8  -n 27258880   > p2n00512t64
srun -N1024  -n1024  $QS -i Coral2_P2.inp -X 16 -Y 8  -Z 8  -x 176 -y 88  -z 88  -I 16 -J 8  -K 8  -n 54517760   > p2n01024t64
srun -N2048  -n2048  $QS -i Coral2_P2.inp -X 16 -Y 16 -Z 8  -x 176 -y 176 -z 88  -I 16 -J 16 -K 8  -n 109035520  > p2n02048t64
srun -N4096  -n4096  $QS -i Coral2_P2.inp -X 16 -Y 16 -Z 16 -x 176 -y 176 -z 176 -I 16 -J 16 -K 16 -n 218071040  > p2n04096t64
srun -N8192  -n8192  $QS -i Coral2_P2.inp -X 32 -Y 16 -Z 16 -x 352 -y 176 -z 176 -I 32 -J 16 -K 16 -n 436142080  > p2n08192t64
srun -N16384 -n16384 $QS -i Coral2_P2.inp -X 32 -Y 32 -Z 16 -x 352 -y 352 -z 176 -I 32 -J 32 -K 16 -n 872284160  > p2n16384t64
srun -N24576 -n24576 $QS -i Coral2_P2.inp -X 48 -Y 32 -Z 16 -x 528 -y 532 -z 176 -I 48 -J 32 -K 16 -n 1308426240 > p2n24768t64
