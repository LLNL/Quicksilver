#!/bin/bash

#Problem 1:

# 4 ranks per node 
# 16 threads per rank
# 4096 mesh elements per node
# 40 particles per mesh element -> 163840 particles per node

export OMP_NUM_THREADS=16

QS=../../../src/qs

srun -N24576 -n98304 $QS -i Coral2_P1.inp -X 768 -Y 512 -Z 256 -x 768 -y 512 -z 256 -I 96 -J 32 -K 32 -n 4026531840 > p1n24576t16
srun -N16384 -n65536 $QS -i Coral2_P1.inp -X 512 -Y 512 -Z 256 -x 512 -y 512 -z 256 -I 64 -J 32 -K 32 -n 2684354560 > p1n16384t16
srun -N8192  -n32768 $QS -i Coral2_P1.inp -X 512 -Y 256 -Z 256 -x 512 -y 256 -z 256 -I 32 -J 32 -K 32 -n 1342117280 > p1n08192t16
srun -N4096  -n16384 $QS -i Coral2_P1.inp -X 256 -Y 256 -Z 256 -x 256 -y 256 -z 256 -I 32 -J 32 -K 16 -n 671088640  > p1n04092t16
srun -N2048  -n8192  $QS -i Coral2_P1.inp -X 256 -Y 256 -Z 128 -x 256 -y 256 -z 128 -I 32 -J 16 -K 16 -n 335544320  > p1n02048t16
srun -N1024  -n4096  $QS -i Coral2_P1.inp -X 256 -Y 128 -Z 128 -x 256 -y 128 -z 128 -I 16 -J 16 -K 16 -n 167772160  > p1n01024t16
srun -N512   -n2048  $QS -i Coral2_P1.inp -X 128 -Y 128 -Z 128 -x 128 -y 128 -z 128 -I 16 -J 16 -K 8  -n 83886080   > p1n00512t16
srun -N256   -n1024  $QS -i Coral2_P1.inp -X 128 -Y 128 -Z 64  -x 128 -y 128 -z 64  -I 16 -J 8  -K 8  -n 41943040   > p1n00256t16
srun -N128   -n512   $QS -i Coral2_P1.inp -X 128 -Y 64  -Z 64  -x 128 -y 64  -z 64  -I 8  -J 8  -K 8  -n 20971520   > p1n00128t16
srun -N64    -n256   $QS -i Coral2_P1.inp -X 64  -Y 64  -Z 64  -x 64  -y 64  -z 64  -I 8  -J 8  -K 4  -n 10485760   > p1n00064t16
srun -N32    -n128   $QS -i Coral2_P1.inp -X 64  -Y 64  -Z 32  -x 64  -y 64  -z 32  -I 8  -J 4  -K 4  -n 5242880    > p1n00032t16
srun -N16    -n64    $QS -i Coral2_P1.inp -X 64  -Y 32  -Z 32  -x 64  -y 32  -z 32  -I 4  -J 4  -K 4  -n 2621440    > p1n00016t16
srun -N8     -n32    $QS -i Coral2_P1.inp -X 32  -Y 32  -Z 32  -x 32  -y 32  -z 32  -I 4  -J 4  -K 2  -n 1310720    > p1n00008t16
srun -N4     -n16    $QS -i Coral2_P1.inp -X 32  -Y 32  -Z 16  -x 32  -y 32  -z 16  -I 4  -J 2  -K 2  -n 655360     > p1n00004t16
srun -N2     -n8     $QS -i Coral2_P1.inp -X 32  -Y 16  -Z 16  -x 32  -y 16  -z 16  -I 2  -J 2  -K 2  -n 327680     > p1n00002t16
srun -N1     -n4     $QS -i Coral2_P1.inp -X 16  -Y 16  -Z 16  -x 16  -y 16  -z 16  -I 2  -J 2  -K 1  -n 163840     > p1n00001t16
