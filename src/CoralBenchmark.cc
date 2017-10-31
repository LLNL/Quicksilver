#include "CoralBenchmark.hh"
#include "MonteCarlo.hh"
#include "Parameters.hh"
#include "Tallies.hh"
#include "utilsMpi.hh"
#include "MC_Processor_Info.hh"


#ifdef CORAL_2_BENCHMARK
void scalarFluxTest( MonteCarlo* monteCarlo );
#endif

#ifdef CORAL_2_BENCHMARK
void coralBenchmarkCorrectness( MonteCarlo* monteCarlo, Parameters &params )
{

    fprintf(stdout,"\n");
    fprintf(stdout, "Testing Ratios for Absorbtion, Fission, and Scattering are maintained\n");
    //Test Balance Tallies for relative correctness
    //  Expected ratios of absorbs,fisisons, scatters are maintained
    //  withing some tolerance, based on input expectation
    Balance &balTally = monteCarlo->_tallies->_balanceCumulative;

    uint64_t absorb     = balTally._absorb;
    uint64_t fission    = balTally._fission;
    uint64_t scatter    = balTally._scatter;
    double absorbRatio, fissionRatio, scatterRatio;

    for (auto matIter = params.materialParams.begin();
              matIter != params.materialParams.end(); 
              matIter++)
    {
        const MaterialParameters& mp = matIter->second;
        fissionRatio = mp.fissionCrossSectionRatio;
        scatterRatio = mp.scatteringCrossSectionRatio;
        absorbRatio  = mp.absorptionCrossSectionRatio;
    }

    double Absorb2Scatter  = ( absorb /  absorbRatio  ) * (scatterRatio / scatter);
    double Absorb2Fission  = ( absorb /  absorbRatio  ) * (fissionRatio / fission);
    double Scatter2Absorb  = ( scatter / scatterRatio ) * (absorbRatio  / absorb );
    double Scatter2Fission = ( scatter / scatterRatio ) * (fissionRatio / fission);
    double Fission2Absorb  = ( fission / fissionRatio ) * (absorbRatio  / absorb );
    double Fission2Scatter = ( fission / fissionRatio ) * (scatterRatio / scatter);

    double percent_tolerance = 1.0;
    double tolerance = percent_tolerance / 100.0;
    double min_tolerance = 1-(1*tolerance);
    double max_tolerance = 1+(1*tolerance);

    bool print_results = false;

    if( print_results )
    {
        fprintf(stdout, "absorb:  %12" PRIu64 "\t%g\n", absorb, absorbRatio);
        fprintf(stdout, "scatter: %12" PRIu64 "\t%g\n", scatter, scatterRatio);
        fprintf(stdout, "fission: %12" PRIu64 "\t%g\n", fission, fissionRatio);
        fprintf(stdout, "Relative Absorb to Scatter:  %g < %g < %g\n", min_tolerance, ( absorb /  absorbRatio  ) * (scatterRatio / scatter), max_tolerance );
        fprintf(stdout, "Relative Absorb to Fission:  %g < %g < %g\n", min_tolerance, ( absorb /  absorbRatio  ) * (fissionRatio / fission), max_tolerance );
        fprintf(stdout, "Relative Scatter to Absorb:  %g < %g < %g\n", min_tolerance, ( scatter / scatterRatio ) * (absorbRatio  / absorb ), max_tolerance );
        fprintf(stdout, "Relative Scatter to Fission: %g < %g < %g\n", min_tolerance, ( scatter / scatterRatio ) * (fissionRatio / fission), max_tolerance );
        fprintf(stdout, "Relative Fission to Absorb:  %g < %g < %g\n", min_tolerance, ( fission / fissionRatio ) * (absorbRatio  / absorb ), max_tolerance );
        fprintf(stdout, "Relative Fission to Scatter: %g < %g < %g\n", min_tolerance, ( fission / fissionRatio ) * (scatterRatio / scatter), max_tolerance );
    }
    
    bool pass = true;

    if( min_tolerance > Absorb2Scatter && Absorb2Scatter > max_tolerance )
    {
        pass = false;
        fprintf(stdout,  "Absorb to Scatter Ratio is not less then tolerance: %g\n", tolerance );
    }
    if( min_tolerance > Absorb2Fission && Absorb2Scatter > max_tolerance )
    {
        pass = false;
        fprintf(stdout,  "Absorb to Fission Ratio is not less then tolerance: %g\n", tolerance);
    }
    if( min_tolerance > Scatter2Absorb && Scatter2Absorb > max_tolerance )
    {
        pass = false;
        fprintf(stdout,  "Scatter to Absorb Ratio is not less then tolerance: %g\n", tolerance);
    }
    if( min_tolerance > Scatter2Fission && Scatter2Fission > max_tolerance )
    {
        pass = false;
        fprintf(stdout,  "Scatter to Fission Ratio is not less then tolerance: %g\n", tolerance);
    }
    if( min_tolerance > Fission2Absorb && Fission2Absorb > max_tolerance )
    {
        pass = false;
        fprintf(stdout,  "Fission to Absorb Ratio is not less then tolerance: %g\n", tolerance);
    }
    if( min_tolerance > Fission2Scatter && Fission2Scatter > max_tolerance )
    {
        pass = false;
        fprintf(stdout,  "Fission to Scatter Ratio is not less then tolerance: %g\n", tolerance);
    }

    if( pass )
    {
        fprintf(stdout, "PASS:: Absorption / Fission / Scatter Ratios maintained with %g%% tolerance\n", tolerance*100.0);
    }
    else
    {
        fprintf(stdout, "FAIL:: Absorption / Fission / Scatter Ratios NOT maintained with %g%% tolerance\n", tolerance*100.0);
    }

    //Test for lost particles during the simulation
    //  This test should always succeed unless test for 
    //  done was broken, or we are running with 1 MPI rank
    //  and so never preform this test duing test_for_done

    fprintf(stdout,"\n");
    fprintf(stdout, "Test for lost / unaccounted for particles in this simulation\n");

    uint64_t gains = 0, losses = 0;
    
    gains   = balTally._start  + balTally._source + balTally._produce + balTally._split;
    losses  = balTally._absorb + balTally._census + balTally._escape  + balTally._rr + balTally._fission;

    if( gains == losses )
    {
        fprintf( stdout, "PASS:: No Particles Lost During Run\n" );
    }
    else
    {
        fprintf( stdout, "FAIL:: Particles Were Lost During Run, test for done should have failed\n" );
    }

    //Test that the scalar flux is homogenous across cells for the problem
    //  This test really required alot of particles or cycles or both
    //  This solution should converge to a homogenous solution

    scalarFluxTest( monteCarlo );

}
#endif


#ifdef CORAL_2_BENCHMARK
void scalarFluxTest( MonteCarlo* monteCarlo )
{
    fprintf(stdout,"\n");
    fprintf(stdout, "Test Scalar Flux for homogenous values across cells\n");
    double local_sum = 0.0;

    for (int domainIndex = 0; domainIndex < monteCarlo->_tallies->_scalarFluxDomain.size(); domainIndex++)
    {
        for (int replicationIndex = 0; replicationIndex < monteCarlo->_tallies->GetNumFluxReplications(); replicationIndex++)
        {
            int numCells = monteCarlo->_tallies->_scalarFluxDomain[domainIndex]._task[replicationIndex]._cell.size();

            for (int cellIndex = 0; cellIndex < numCells; cellIndex++)
            {
                int numGroups = monteCarlo->_tallies->_scalarFluxDomain[domainIndex]._task[replicationIndex]._cell[cellIndex].size();
                double groupSum = 0.0;
                for (int groupIndex = 0; groupIndex < numGroups; groupIndex++)
                {
                    groupSum += monteCarlo->_tallies->_scalarFluxDomain[domainIndex]._task[replicationIndex]._cell[cellIndex]._group[groupIndex];
                }
                local_sum += groupSum;
            }
        }
    }

    double max_diff = 0.0;
    
    for (int domainIndex = 0; domainIndex < monteCarlo->_tallies->_scalarFluxDomain.size(); domainIndex++)
    {
        for (int replicationIndex = 0; replicationIndex < monteCarlo->_tallies->GetNumFluxReplications(); replicationIndex++)
        {
            int numCells = monteCarlo->_tallies->_scalarFluxDomain[domainIndex]._task[replicationIndex]._cell.size();
            
            double average = local_sum / numCells;

            for (int cellIndex = 0; cellIndex < numCells; cellIndex++)
            {
                int numGroups = monteCarlo->_tallies->_scalarFluxDomain[domainIndex]._task[replicationIndex]._cell[cellIndex].size();
                double groupSum = 0.0;
                for (int groupIndex = 0; groupIndex < numGroups; groupIndex++)
                {
                    groupSum += monteCarlo->_tallies->_scalarFluxDomain[domainIndex]._task[replicationIndex]._cell[cellIndex]._group[groupIndex];
                }
                local_sum += groupSum;
                double percent_diff = (((groupSum > average) ? groupSum - average : average - groupSum ) / average)*100;
                max_diff = ( (max_diff > percent_diff) ? max_diff : percent_diff );
            }
        }
    }

    double percent_tolerance = 10.0;

    if( max_diff > percent_tolerance )
    {
        fprintf( stdout, "FAIL:: Scalar Flux not homogenous across cells within %g%% tolerance\n", percent_tolerance);
        fprintf( stdout, "\tTry running more particles or more cycles to see if Max Percent Difference goes down.\n");
        fprintf( stdout, "\tCurrent Max Percent Diff: %4.1f%%\n", max_diff);
    }
    else
    {
        fprintf( stdout, "PASS:: Scalar Flux is homogenous across cells with %g%% tolerance\n", percent_tolerance );
    }


}
#endif
