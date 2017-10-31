#if defined(HAVE_OPENMP)
    #pragma omp parallel for schedule (static) MC_OMP_PARALLEL_FOR_IF_CONDITION
#endif

