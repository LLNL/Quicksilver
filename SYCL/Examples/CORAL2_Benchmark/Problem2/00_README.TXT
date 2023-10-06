This is Quicksilver Problem #2 for the CORAL2 Benchmark.

This problem is *not* required.

Cross sections in this problem are tailored to give a broader energy
spectrum for the particles and a different reaction mix compared to
problem 1.   

The essential physics of the problem are defined in the input file
Coral2_P2.inp.  The parameters in this file should not be changed.
Parameters to set the size of the problem (number of particles, number
of mesh elements, size of domain, and MPI decomposition), can all be
specified on the command line (for example see P2_64t.sh).  Alternately,
you can copy Coral2_P2.inp to a new file and add the necessary
parameters (see Coral2_P2_1.inp).

Note that parameters in the input deck overrride corresponding command
line arguments.

For the scaling study here, we have chosen 11^3 = 1331 mesh elements per
node.  This makes it difficult to uniformly decomopostion of mesh
elements anything other than 1 rank per node.  We also choose 40
particles per mesh element.  This is divisible by 10 (so we get an
integer number of particles sourced in) and gives a reasonable cyle time
of 2-4 seconds.

MANIFEST:

00_README.TXT        This file
Coral2_P2.inp        Input without problem size specification.
                     Useful to build scaling study with commmand line
                     arguments. 
Coral2_P2_1.inp      Input file for a single MPI rank
Coral2_P2_4096.inp   Input file for 4096 MPI ranks
P1_64t.sh            Example scaling study for BG/Q with 64 threads per rank
