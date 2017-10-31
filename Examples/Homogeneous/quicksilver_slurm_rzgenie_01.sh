#
# salloc 1 nodes exclusively, then run these tests.
# Or put them in batch script
#

export -n KMP_AFFINITY
export OMP_PROC_BIND=FALSE

# ####################
# Thread Funneled Runs
# ####################

# 32 MPI x 1 Thread - Thread Funneled
export OMP_NUM_THREADS=1
srun -n 32 --distribution=cyclic --mpibind ./qs \
    --lx=400 --ly=400 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=4 --yDom=4 --zDom=2 --nParticles=4000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzgenie.Node0001.n0032.t0001-ts.out

# 16 MPI x 2 Thread - Thread Funneled
export OMP_NUM_THREADS=2
srun -n 16 --distribution=cyclic --mpibind ./qs \
    --lx=400 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=4 --yDom=2 --zDom=2 --nParticles=4000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzgenie.Node0001.n0016.t0002-ts.out

#  8 MPI x 4 Thread - Thread Funneled
export OMP_NUM_THREADS=4
srun -n 8 --distribution=cyclic ./qs \
    --lx=200 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=2 --nParticles=4000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzgenie.Node0001.n0008.t0004-ts.out

#  4 MPI x 8 Thread - Thread Funneled
export OMP_NUM_THREADS=8
srun -n 4 --distribution=cyclic ./qs \
    --lx=200 --ly=200 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=4000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzgenie.Node0001.n0004.t0008-ts.out

#  4 MPI x 16 Thread - Thread Funneled
export OMP_NUM_THREADS=16
srun -n 2 --distribution=cyclic ./qs \
    --lx=200 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=1 --zDom=1 --nParticles=4000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzgenie.Node0001.n0002.t0016-ts.out

# ####################
# Thread Multiple Runs
# For testing, does not show improvement on Xeon
# ####################

# 32 MPI x 1 Thread - Thread Multiple
#export OMP_NUM_THREADS=1
#srun -n 32 --distribution=cyclic ./qs --mpiThreadMultiple  \
#    --lx=400 --ly=400 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=4 --yDom=4 --zDom=2 --nParticles=4000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzgenie.Node0001.n0032.t0001-tm.out

# 16 MPI x 2 Thread - Thread Multiple
#export OMP_NUM_THREADS=2
#srun -n 16 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=400 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=4 --yDom=2 --zDom=2 --nParticles=4000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzgenie.Node0001.n0016.t0002-tm.out

#  8 MPI x 4 Thread - Thread Multiple
#export OMP_NUM_THREADS=4
#srun -n 8 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=200 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=2 --nParticles=4000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzgenie.Node0001.n0008.t0004-tm.out

#  4 MPI x 8 Thread - Thread Multiple
#export OMP_NUM_THREADS=8
#srun -n 4 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=200 --ly=200 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=4000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzgenie.Node0001.n0004.t0008-tm.out

#  4 MPI x 16 Thread - Thread Multiple
#export OMP_NUM_THREADS=16
#srun -n 2 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=200 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=1 --zDom=1 --nParticles=4000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzgenie.Node0001.n0002.t0016-tm.out

#
# end of file
#
