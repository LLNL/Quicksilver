#!/bin/sh

export OMP_NUM_THREADS=64

set QS=../../src/qs

srun -N1    -n1    $QS -i Coral_2.inp -X 10  -Y 10  -Z 10  -x 10  -y 10  -z 10  -I 1  -J 1  -K 1  --nParticles 200000    > r0001t64
srun -N2    -n2    $QS -i Coral_2.inp -X 20  -Y 10  -Z 10  -x 20  -y 10  -z 10  -I 2  -J 1  -K 1  --nParticles 400000    > r0002t64
srun -N4    -n4    $QS -i Coral_2.inp -X 20  -Y 20  -Z 10  -x 20  -y 20  -z 10  -I 2  -J 2  -K 1  --nParticles 800000    > r0004t64
srun -N8    -n8    $QS -i Coral_2.inp -X 20  -Y 20  -Z 20  -x 20  -y 20  -z 20  -I 2  -J 2  -K 2  --nParticles 1600000   > r0008t64
srun -N16   -n16   $QS -i Coral_2.inp -X 40  -Y 20  -Z 20  -x 40  -y 20  -z 20  -I 4  -J 2  -K 2  --nParticles 3200000   > r0016t64
srun -N32   -n32   $QS -i Coral_2.inp -X 40  -Y 40  -Z 20  -x 40  -y 40  -z 20  -I 4  -J 4  -K 2  --nParticles 6400000   > r0032t64
srun -N64   -n64   $QS -i Coral_2.inp -X 40  -Y 40  -Z 40  -x 40  -y 40  -z 40  -I 4  -J 4  -K 4  --nParticles 12800000  > r0064t64
srun -N128  -n128  $QS -i Coral_2.inp -X 80  -Y 40  -Z 40  -x 80  -y 40  -z 40  -I 8  -J 4  -K 4  --nParticles 25600000  > r0128t64
srun -N256  -n256  $QS -i Coral_2.inp -X 80  -Y 80  -Z 40  -x 80  -y 80  -z 40  -I 8  -J 8  -K 4  --nParticles 51200000  > r0256t64
srun -N512  -n512  $QS -i Coral_2.inp -X 80  -Y 80  -Z 80  -x 80  -y 80  -z 80  -I 8  -J 8  -K 8  --nParticles 10240000  > r0512t64
srun -N1024 -n1024 $QS -i Coral_2.inp -X 160 -Y 80  -Z 80  -x 160 -y 80  -z 80  -I 16 -J 8  -K 8  --nParticles 204800000 > r1024t64
srun -N2048 -n2048 $QS -i Coral_2.inp -X 160 -Y 160 -Z 80  -x 160 -y 160 -z 80  -I 16 -J 16 -K 8  --nParticles 409600000 > r2048t64
srun -N4096 -n4096 $QS -i Coral_2.inp -X 160 -Y 160 -Z 160 -x 160 -y 160 -z 160 -I 16 -J 16 -K 16 --nParticles 819200000 > r4096t64
