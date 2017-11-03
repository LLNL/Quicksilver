#include "CoralBenchmark.hh"
#include "MonteCarlo.hh"
#include "Parameters.hh"
#include "Tallies.hh"
#include "utilsMpi.hh"
#include "MC_Processor_Info.hh"
#include <cmath>

void BalanceRatioTest( MonteCarlo* monteCarlo, Parameters &params );
void MissingParticleTest( MonteCarlo* monteCarlo );
void FluenceTest( MonteCarlo* monteCarlo );

void coralBenchmarkCorrectness( MonteCarlo* monteCarlo, Parameters &params )
{
    if( params.simulationParams.coralBenchmark )
    {

        if( monteCarlo->processor_info->rank == 0 )
        {
            //Test Balance Tallies for relative correctness
            //  Expected ratios of absorbs,fisisons, scatters are maintained
            //  withing some tolerance, based on input expectation
            BalanceRatioTest( monteCarlo, params );
            
            //Test for lost particles during the simulation
            //  This test should always succeed unless test for 
            //  done was broken, or we are running with 1 MPI rank
            //  and so never preform this test duing test_for_done
            MissingParticleTest( monteCarlo );
        }

        //Test that the scalar flux is homogenous across cells for the problem
        //  This test really required alot of particles or cycles or both
        //  This solution should converge to a homogenous solution
        FluenceTest( monteCarlo );
    }
}

void BalanceRatioTest( MonteCarlo *monteCarlo, Parameters &params )
{
    fprintf(stdout,"\n");
    fprintf(stdout, "Testing Ratios for Absorbtion, Fission, and Scattering are maintained\n");

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

    double Absorb2Scatter  = std::abs( ( absorb /  absorbRatio  ) * (scatterRatio / scatter) - 1);
    double Absorb2Fission  = std::abs( ( absorb /  absorbRatio  ) * (fissionRatio / fission) - 1);
    double Scatter2Absorb  = std::abs( ( scatter / scatterRatio ) * (absorbRatio  / absorb ) - 1);
    double Scatter2Fission = std::abs( ( scatter / scatterRatio ) * (fissionRatio / fission) - 1);
    double Fission2Absorb  = std::abs( ( fission / fissionRatio ) * (absorbRatio  / absorb ) - 1);
    double Fission2Scatter = std::abs( ( fission / fissionRatio ) * (scatterRatio / scatter) - 1);

    double percent_tolerance = 1.0;
    double tolerance = percent_tolerance / 100.0;

    bool pass = true;

    if( Absorb2Scatter  > tolerance ) pass = false;
    if( Absorb2Fission  > tolerance ) pass = false;
    if( Scatter2Absorb  > tolerance ) pass = false;
    if( Scatter2Fission > tolerance ) pass = false;
    if( Fission2Absorb  > tolerance ) pass = false;
    if( Fission2Scatter > tolerance ) pass = false;

    if( pass )
    {
        fprintf(stdout, "PASS:: Absorption / Fission / Scatter Ratios maintained with %g%% tolerance\n", tolerance*100.0);
    }
    else
    {
        fprintf(stdout, "FAIL:: Absorption / Fission / Scatter Ratios NOT maintained with %g%% tolerance\n", tolerance*100.0);
        fprintf(stdout, "absorb:  %12" PRIu64 "\t%g\n", absorb, absorbRatio);
        fprintf(stdout, "scatter: %12" PRIu64 "\t%g\n", scatter, scatterRatio);
        fprintf(stdout, "fission: %12" PRIu64 "\t%g\n", fission, fissionRatio);
        fprintf(stdout, "Relative Absorb to Scatter:  %g < %g < %g\n", min_tolerance, Absorb2Scatter , max_tolerance );
        fprintf(stdout, "Relative Absorb to Fission:  %g < %g < %g\n", min_tolerance, Absorb2Fission , max_tolerance );
        fprintf(stdout, "Relative Scatter to Absorb:  %g < %g < %g\n", min_tolerance, Scatter2Absorb , max_tolerance );
        fprintf(stdout, "Relative Scatter to Fission: %g < %g < %g\n", min_tolerance, Scatter2Fission, max_tolerance );
        fprintf(stdout, "Relative Fission to Absorb:  %g < %g < %g\n", min_tolerance, Fission2Absorb , max_tolerance );
        fprintf(stdout, "Relative Fission to Scatter: %g < %g < %g\n", min_tolerance, Fission2Scatter, max_tolerance );
    }

}

void MissingParticleTest( MonteCarlo *monteCarlo )
{
    fprintf(stdout,"\n");
    fprintf(stdout, "Test for lost / unaccounted for particles in this simulation\n");

    Balance &balTally = monteCarlo->_tallies->_balanceCumulative;

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


}


void FluenceTest( MonteCarlo* monteCarlo )
{
    if( monteCarlo->processor_info->rank == 0 )
    {
        fprintf(stdout,"\n");
        fprintf(stdout, "Test Fluence for homogeneity across cells\n");
    }

    double max_diff = 0.0;

    int numDomains = monteCarlo->_tallies->_fluence._domain.size();
    for (int domainIndex = 0; domainIndex < numDomains; domainIndex++)
    {
        
        double local_sum = 0.0;
        int numCells = monteCarlo->_tallies->_fluence._domain[domainIndex]->size(); 

        for (int cellIndex = 0; cellIndex < numCells; cellIndex++)
        {
            local_sum += monteCarlo->_tallies->_fluence._domain[domainIndex]->getCell( cellIndex );
        }

        double average = local_sum / numCells;
        
        for (int cellIndex = 0; cellIndex < numCells; cellIndex++)
        {
            double cellValue = monteCarlo->_tallies->_fluence._domain[domainIndex]->getCell( cellIndex );
            double percent_diff = (((cellValue > average) ? cellValue - average : average - cellValue ) / (( cellValue + average)/2.0))*100;
            max_diff = ( (max_diff > percent_diff) ? max_diff : percent_diff );
        }
    }

    double percent_tolerance = 10.0;

    double max_diff_global = 0.0;

    mpiAllreduce(&max_diff, &max_diff_global, 1, MPI_DOUBLE, MPI_MAX, monteCarlo->processor_info->comm_mc_world);

    if( monteCarlo->processor_info->rank == 0 )
    {
        if( max_diff_global > percent_tolerance )
        {
            fprintf( stdout, "FAIL:: Fluence not homogenous across cells within %g%% tolerance\n", percent_tolerance);
            fprintf( stdout, "\tTry running more particles or more cycles to see if Max Percent Difference goes down.\n");
            fprintf( stdout, "\tCurrent Max Percent Diff: %4.1f%%\n", max_diff_global);
        }
        else
        {
            fprintf( stdout, "PASS:: Fluence is homogenous across cells with %g%% tolerance\n", percent_tolerance );
            fprintf( stdout, "\tMax Percent Diff: %4.1f%%\n", max_diff_global);
        }
    }

}
