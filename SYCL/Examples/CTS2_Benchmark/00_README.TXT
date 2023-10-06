This the Quicksilver CTS2 Benchmark Problem.

This problem is based on the optional Problem #2 from the CORAL2
Benchmark, but it has been changed slightly.

The essential physics of the problem are defined in the input file
CTS2.inp.  The parameters in this file should not be changed.
Parameters to set the size of the problem (number of particles, number
of mesh elements, size of domain, and MPI decomposition), can all be
specified on the command line (for example see CTS2_scaling.sh).
Alternately, you can copy CTS2.inp to a new file and add the
necessary parameters (see CTS2_1.inp).

Note that parameters in the input deck overrride corresponding command
line arguments.

This problem should be run on a single node with weak scaling
for the number of cores on a node.  You are required to run this
problem with
 * 1 rank per core
 * 16^3 = 4096 mesh elements per rank
 * 40960 partilces per rank


MANIFEST:

00_README.TXT     This file
CTS2.inp          Input without problem size specification.
                  Useful to build scaling study with commmand line
                  arguments. 
CTS2_1.inp        Input file for a single MPI rank
CTS2_36.inp       Input file for 36 MPI ranks
CTS2_scaling.sh   Example scaling study for 36 cores per node
