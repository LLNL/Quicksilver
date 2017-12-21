This is Quicksilver Problem #1 for the CORAL2 Benchmark.

This problem is required.

The input files in this directory are configured to support a scaling
study and collection of the Figure of Merit on Vulcan (BG/Q) at LLNL.

The essential physics of the problem are defined in the input file
Coral2_P1.inp.  The parameters in this file should not be changed.
Parameters to set the size of the problem (number of particles, number
of mesh elements, size of domain, and MPI decomposition), can all be
specified on the command line (for example see P1_64t.sh).  Alternately,
you can copy Coral2_P1.inp to a new file and add the necessary
parameters (see Coral2_P1_1.inp).

Note that parameters in the input deck overrride corresponding command
line arguments.

For the scaling study here, we have chosen 4096 mesh elements per node.
This allows a uniform spatial decomopostion of mesh elements for both 1
rank per node and 64 ranks per node (and any power of two in between).
We also choose 40 particles per mesh element.  This is divisible by 10
(so we get an integer number of particles sourced in) and gives a
reasonable cyle time of 2-4 seconds.


MANIFEST:

00_README.TXT        This file
Coral2_P1.inp        Input without problem size specification.
                     Useful to build scaling study with commmand line
                     arguments. 
Coral2_P1_1.inp      Input file for a single MPI rank
Coral2_P1_4096.inp   Input file for 4096 MPI ranks
P1_04t.sh            Example scaling study for BG/Q with 4 threads per rank
P1_16t.sh            Example scaling study for BG/Q with 16 threads per rank
P1_64t.sh            Example scaling study for BG/Q with 64 threads per rank
