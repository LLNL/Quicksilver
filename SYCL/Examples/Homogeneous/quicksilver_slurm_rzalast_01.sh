#
# salloc 1 nodes exclusively, then run these tests.  
# Or put them in batch script
#

export -n KMP_AFFINITY
export OMP_PROC_BIND=FALSE

# ####################
# Thread Funneled Runs
# ####################

# (Per Node) 16 MPI x 1 Threads - Thread Funneled
export OMP_NUM_THREADS=1;
srun -n16 --distribution=cyclic ./qs \
    --lx=400 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=4 --yDom=2 --zDom=2 --nParticles=2000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzalast.Node0001.n0016.t0001-ts.out

# (Per Node) 8 MPI x 2 Threads - Thread Funneled
export OMP_NUM_THREADS=2;
srun -n8 --distribution=cyclic ./qs \
    --lx=200 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=2 --nParticles=2000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzalast.Node0001.n0008.t0002-ts.out

# (Per Node) 4 MPI x 4 Threads - Thread Funneled
export OMP_NUM_THREADS=4;
srun -n4 --distribution=cyclic ./qs \
    --lx=200 --ly=200 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=2000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzalast.Node0001.n0004.t0004-ts.out

# (Per Node) 2 MPI x 8 Threads - Thread Funneled
export OMP_NUM_THREADS=8;
srun -n2 --distribution=cyclic ./qs \
    --lx=200 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=1 --zDom=1 --nParticles=2000000 \
    -i Input/homogeneousProblem_v5_ts.inp 2>&1 | tee qs.rzalast.Node0001.n0002.t0008-ts.out

# ####################
# Thread Multiple Runs 
# For testing, does not show improvement on Xeon
# ####################

# (Per Node) 16 MPI x 1 Threads - Thread Multiple
#export OMP_NUM_THREADS=1;
#srun -n16 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=400 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=4 --yDom=2 --zDom=2 --nParticles=2000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzalast.Node0001.n0016.t0001-tm.out

# (Per Node) 8 MPI x 2 Threads - Thread Multiple
#export OMP_NUM_THREADS=2;
#srun -n8 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=200 --ly=200 --lz=200 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=2 --nParticles=2000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzalast.Node0001.n0008.t0002-tm.out

# (Per Node) 4 MPI x 4 Threads - Thread Multiple
#export OMP_NUM_THREADS=4;
#srun -n4 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=200 --ly=200 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=2000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzalast.Node0001.n0004.t0004-tm.out

# (Per Node) 2 MPI x 8 Threads - Thread Multiple
#export OMP_NUM_THREADS=8;
#srun -n2 --distribution=cyclic ./qs --mpiThreadMultiple \
#    --lx=200 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=1 --zDom=1 --nParticles=2000000 \
#    -i Input/homogeneousProblem_v5_tm.inp 2>&1 | tee qs.rzalast.Node0001.n0002.t0008-tm.out

#
# end of file
#
