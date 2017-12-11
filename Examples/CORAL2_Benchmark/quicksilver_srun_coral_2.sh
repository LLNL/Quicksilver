#!/bin/sh

export OMP_NUM_THREADS=64

QS=../../src/qs

#Problem 1

srun -N1    -n1    $QS -i Problem1/Coral_2.inp -X 10  -Y 10  -Z 10  -x 10  -y 10  -z 10  -I 1  -J 1  -K 1  --nParticles 100000    > p1r0001t64
srun -N2    -n2    $QS -i Problem1/Coral_2.inp -X 20  -Y 10  -Z 10  -x 20  -y 10  -z 10  -I 2  -J 1  -K 1  --nParticles 200000    > p1r0002t64
srun -N4    -n4    $QS -i Problem1/Coral_2.inp -X 20  -Y 20  -Z 10  -x 20  -y 20  -z 10  -I 2  -J 2  -K 1  --nParticles 400000    > p1r0004t64
srun -N8    -n8    $QS -i Problem1/Coral_2.inp -X 20  -Y 20  -Z 20  -x 20  -y 20  -z 20  -I 2  -J 2  -K 2  --nParticles 800000    > p1r0008t64
srun -N16   -n16   $QS -i Problem1/Coral_2.inp -X 40  -Y 20  -Z 20  -x 40  -y 20  -z 20  -I 4  -J 2  -K 2  --nParticles 1600000   > p1r0016t64
srun -N32   -n32   $QS -i Problem1/Coral_2.inp -X 40  -Y 40  -Z 20  -x 40  -y 40  -z 20  -I 4  -J 4  -K 2  --nParticles 3200000   > p1r0032t64
srun -N64   -n64   $QS -i Problem1/Coral_2.inp -X 40  -Y 40  -Z 40  -x 40  -y 40  -z 40  -I 4  -J 4  -K 4  --nParticles 6400000   > p1r0064t64
srun -N128  -n128  $QS -i Problem1/Coral_2.inp -X 80  -Y 40  -Z 40  -x 80  -y 40  -z 40  -I 8  -J 4  -K 4  --nParticles 12800000  > p1r0128t64
srun -N256  -n256  $QS -i Problem1/Coral_2.inp -X 80  -Y 80  -Z 40  -x 80  -y 80  -z 40  -I 8  -J 8  -K 4  --nParticles 25600000  > p1r0256t64
srun -N512  -n512  $QS -i Problem1/Coral_2.inp -X 80  -Y 80  -Z 80  -x 80  -y 80  -z 80  -I 8  -J 8  -K 8  --nParticles 51200000  > p1r0512t64
srun -N1024 -n1024 $QS -i Problem1/Coral_2.inp -X 160 -Y 80  -Z 80  -x 160 -y 80  -z 80  -I 16 -J 8  -K 8  --nParticles 102400000 > p1r1024t64
srun -N2048 -n2048 $QS -i Problem1/Coral_2.inp -X 160 -Y 160 -Z 80  -x 160 -y 160 -z 80  -I 16 -J 16 -K 8  --nParticles 204800000 > p1r2048t64
srun -N4096 -n4096 $QS -i Problem1/Coral_2.inp -X 160 -Y 160 -Z 160 -x 160 -y 160 -z 160 -I 16 -J 16 -K 16 --nParticles 409600000 > p1r4096t64

#Problem 2

srun -N1    -n1    $QS -i Problem2/Coral2_secondProblem.inp -X 1  -Y 1  -Z 1  -x 11  -y 11  -z 11  -I 1  -J 1  -K 1  --nParticles 20000    > p2r0001t64
srun -N2    -n2    $QS -i Problem2/Coral2_secondProblem.inp -X 2  -Y 1  -Z 1  -x 22  -y 10  -z 10  -I 2  -J 1  -K 1  --nParticles 40000    > p2r0002t64
srun -N4    -n4    $QS -i Problem2/Coral2_secondProblem.inp -X 2  -Y 2  -Z 1  -x 22  -y 22  -z 10  -I 2  -J 2  -K 1  --nParticles 80000    > p2r0004t64
srun -N8    -n8    $QS -i Problem2/Coral2_secondProblem.inp -X 2  -Y 2  -Z 2  -x 22  -y 22  -z 22  -I 2  -J 2  -K 2  --nParticles 160000   > p2r0008t64
srun -N16   -n16   $QS -i Problem2/Coral2_secondProblem.inp -X 4  -Y 2  -Z 2  -x 44  -y 22  -z 22  -I 4  -J 2  -K 2  --nParticles 320000   > p2r0016t64
srun -N32   -n32   $QS -i Problem2/Coral2_secondProblem.inp -X 4  -Y 4  -Z 2  -x 44  -y 44  -z 22  -I 4  -J 4  -K 2  --nParticles 640000   > p2r0032t64
srun -N64   -n64   $QS -i Problem2/Coral2_secondProblem.inp -X 4  -Y 4  -Z 4  -x 44  -y 44  -z 44  -I 4  -J 4  -K 4  --nParticles 1280000  > p2r0064t64
srun -N128  -n128  $QS -i Problem2/Coral2_secondProblem.inp -X 8  -Y 4  -Z 4  -x 88  -y 44  -z 44  -I 8  -J 4  -K 4  --nParticles 2560000  > p2r0128t64
srun -N256  -n256  $QS -i Problem2/Coral2_secondProblem.inp -X 8  -Y 8  -Z 4  -x 88  -y 88  -z 44  -I 8  -J 8  -K 4  --nParticles 5120000  > p2r0256t64
srun -N512  -n512  $QS -i Problem2/Coral2_secondProblem.inp -X 8  -Y 8  -Z 8  -x 88  -y 88  -z 88  -I 8  -J 8  -K 8  --nParticles 10240000 > p2r0512t64
srun -N1024 -n1024 $QS -i Problem2/Coral2_secondProblem.inp -X 16 -Y 8  -Z 8  -x 176 -y 88  -z 88  -I 16 -J 8  -K 8  --nParticles 20480000 > p2r1024t64
srun -N2048 -n2048 $QS -i Problem2/Coral2_secondProblem.inp -X 16 -Y 16 -Z 8  -x 176 -y 176 -z 88  -I 16 -J 16 -K 8  --nParticles 40960000 > p2r2048t64
srun -N4096 -n4096 $QS -i Problem2/Coral2_secondProblem.inp -X 16 -Y 16 -Z 16 -x 176 -y 176 -z 176 -I 16 -J 16 -K 16 --nParticles 81920000 > p2r4096t64
