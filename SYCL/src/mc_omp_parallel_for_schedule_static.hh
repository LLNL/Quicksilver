#if defined(HAVE_OPENMP)
    if ( (mcco->processor_info->rank == 0)  && (mcco->_params.simulationParams.debugThreads >= 2))
       { printf("OpenMP Looping over %d threads\n",omp_get_max_threads()); }
    #pragma omp parallel for schedule (static)
#endif
