#if defined(HAVE_OPENMP)
    int num_physical_cores = mc_get_num_physical_procs();
    if ((mcco->processor_info->rank == 0)  && (mcco->_params.simulationParams.debugThreads >= 2))
       { printf("OpenMP Looping over %d cores\n",num_physical_cores); }
    #pragma omp parallel for schedule (static) num_threads(num_physical_cores)
#endif
