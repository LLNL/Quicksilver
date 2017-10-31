#!/bin/bash
#MSUB -lnodes=4:knl,os=CLE_quad_cache
#MSUB -l walltime=2:00:00
#MSUB -A tos2-8

#
#
# To run interactively, grab a node like so:
#
# msub -I -lnodes=1:knl,os=CLE_quad_cache
#
# This relies on the bash shell for the 2>&2 | tee to work.
#
# To get average and max cycleTracking times:
# grep "cycleTracking                       10" *out | awk -F " " '{print $1 " " $4 " " $5}'
#

# ####################
# Thread Funneled Runs - No Hyper Threads
# ####################

# Set this to where you have the code built on lustre
cd /users/sdawson/Quicksilver-2017-Apr-19-12-45-27

export MPICH_MAX_THREAD_SAFETY=funneled
export OMP_PLACES=cores

# (Per Node) 64 MPI x  1 Threads - Thread Funneled
export OMP_NUM_THREADS=1
time aprun -r 4 -n 256 -d 1 -j 1 -cc depth ./qs \
    --lx=800 --ly=800 --lz=400 --nx=40 --ny=40 --nz=20 --xDom=8 --yDom=8 --zDom=4 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0256.d001.j01-ts.out

# (Per Node) 32 MPI x  2 Threads - Thread Funneled
export OMP_NUM_THREADS=2
time aprun -r 4 -n 128 -d 2 -j 1 -cc depth ./qs \
    --lx=800 --ly=400 --lz=400 --nx=40 --ny=40 --nz=20 --xDom=8 --yDom=4 --zDom=4 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0128.d002.j01-ts.out

# (Per Node) 16 MPI x  4 Threads - Thread Funneled
export OMP_NUM_THREADS=4
time aprun -r 4 -n 64 -d 4 -j 1 -cc depth ./qs \
    --lx=400 --ly=400 --lz=400 --nx=40 --ny=40 --nz=20 --xDom=4 --yDom=4 --zDom=4 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0064.d004.j01-ts.out

# (Per Node)  8 MPI x  8 Threads - Thread Funneled
export OMP_NUM_THREADS=8
time aprun -r 4 -n 32 -d 8 -j 1 -cc depth ./qs \
    --lx=400 --ly=400 --lz=200 --nx=40 --ny=40 --nz=20 --xDom=4 --yDom=4 --zDom=2 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0032.d008.j01-ts.out

# (Per Node)  4 MPI x 16 Threads - Thread Funneled
export OMP_NUM_THREADS=16
time aprun -r 4 -n 16 -d 16 -j 1 -cc depth ./qs \
    --lx=400 --ly=200 --lz=200 --nx=40 --ny=40 --nz=20 --xDom=4 --yDom=2 --zDom=2 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0016.d016.j01-ts.out

# (Per Node)  2 MPI x 32 Threads - Thread Funneled
export OMP_NUM_THREADS=32
time aprun -r 4 -n 8 -d 32 -j 1 -cc depth ./qs \
    --lx=200 --ly=200 --lz=200 --nx=40 --ny=40 --nz=20 --xDom=2 --yDom=2 --zDom=2 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0008.d032.j01-ts.out

# ####################
# Thread Funneled Runs - 2 Hyper Threads
#
# As we add hyper threads, we do not change the problem size, ideally this will decrease time
# spent in the threaded tracking though.
#
# Prior experience shows that while 4 hyper threads pays off on small node count, it is a wash
# at higher node count, so let's stop at 2 hyper threads.
# ####################

export MPICH_MAX_THREAD_SAFETY=funneled
export OMP_PLACES=threads

export OMP_NUM_THREADS=2
time aprun -r 4 -n 256 -d 2 -j 2 -cc depth ./qs \
    --lx=800 --ly=800 --lz=400 --nx=40 --ny=40 --nz=20 --xDom=8 --yDom=8 --zDom=4 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0256.d002.j02-ts.out

export OMP_NUM_THREADS=4
time aprun -r 4 -n 128 -d 4 -j 2 -cc depth ./qs \
    --lx=800 --ly=400 --lz=400 --nx=40 --ny=40 --nz=20 --xDom=8 --yDom=4 --zDom=4 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0128.d004.j02-ts.out

export OMP_NUM_THREADS=8
time aprun -r 4 -n 64 -d 8 -j 2 -cc depth ./qs \
    --lx=400 --ly=400 --lz=400 --nx=40 --ny=40 --nz=20 --xDom=4 --yDom=4 --zDom=4 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0064.d008.j02-ts.out

export OMP_NUM_THREADS=16
time aprun -r 4 -n 32 -d 16 -j 2 -cc depth ./qs \
    --lx=400 --ly=400 --lz=200 --nx=40 --ny=40 --nz=20 --xDom=4 --yDom=4 --zDom=2 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0032.d016.j02-ts.out

export OMP_NUM_THREADS=32
time aprun -r 4 -n 16 -d 32 -j 2 -cc depth ./qs \
    --lx=400 --ly=200 --lz=200 --nx=40 --ny=40 --nz=20 --xDom=4 --yDom=2 --zDom=2 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0016.d032.j02-ts.out

export OMP_NUM_THREADS=64
time aprun -r 4 -n 8 -d 64 -j 2 -cc depth ./qs \
    --lx=200 --ly=200 --lz=200 --nx=40 --ny=40 --nz=20 --xDom=2 --yDom=2 --zDom=2 --nParticles=8000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.trinity.Node0004.n0008.d064.j02-ts.out

#
# end of file
#
