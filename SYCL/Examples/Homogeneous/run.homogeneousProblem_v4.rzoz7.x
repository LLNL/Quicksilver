#
# 2016-Oct-06 Note by S. Dawson
#
# Note on running thread multiple vs thread single.
#
# Its a bit clunky as one has to set up a separate test deck for thread single vs thread multiple, AS WELL AS
# specify the correct command line argument.  
#
# This has to do with the desire to fire up MPI before processing the command line arguments, yet still
# have the input deck reflect how the deck is run.
#
# Also, despite with the command line help says, one can not specify a flag to --mpiThreadMultiple.  If one
# says --mpiThreadMultiple=1 or --mpiThreadMultiple=0 the code complains, it is just --mpiThreadMultiple
# to turn it on and the default is to be in thrad single mode
#

export -n KMP_CPUINFO_FILE
export KMP_CPUINFO_FILE=/home/dawson/cpuinfo_sad;
export I_MPI_PIN_DOMAIN=64:compact
export KMP_AFFINITY="granularity=fine,scatter"
export KMP_HW_SUBSET=1T
export KMP_BLOCKTIME=0
export OMP_NUM_THREADS=16
#export OMP_PLACES=cores

export MPICH_MAX_THREAD_SAFETY=multiple
time mpirun -np 4 ./qs --mpiThreadMultiple --lx=100 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=20000000 -i homogeneousProblem_v4_tm.inp | tee rzoz18.N01.n04.t016.tm.out
export MPICH_MAX_THREAD_SAFETY=funneled
time mpirun -np 4 ./qs                     --lx=100 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=20000000 -i homogeneousProblem_v4_ts.inp | tee rzoz18.N01.n04.t016.ts.out

export KMP_HW_SUBSET=2T;
export OMP_NUM_THREADS=32;
export MPICH_MAX_THREAD_SAFETY=multiple
time mpirun -np 4 ./qs --mpiThreadMultiple --lx=100 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=20000000 -i homogeneousProblem_v4_tm.inp | tee rzoz18.N01.n04.t032.tm.out
export MPICH_MAX_THREAD_SAFETY=funneled
time mpirun -np 4 ./qs                     --lx=100 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=20000000 -i homogeneousProblem_v4_ts.inp | tee rzoz18.N01.n04.t032.ts.out

export KMP_HW_SUBSET=4T;
export OMP_NUM_THREADS=64;
export MPICH_MAX_THREAD_SAFETY=multiple
time mpirun -np 4 ./qs --mpiThreadMultiple --lx=100 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=20000000 -i homogeneousProblem_v4_tm.inp | tee rzoz18.N01.n04.t064.tm.out
export MPICH_MAX_THREAD_SAFETY=funneled
time mpirun -np 4 ./qs                     --lx=100 --ly=100 --lz=100 --nx=20 --ny=20 --nz=20 --xDom=2 --yDom=2 --zDom=1 --nParticles=20000000 -i homogeneousProblem_v4_ts.inp | tee rzoz18.N01.n04.t064.ts.out


