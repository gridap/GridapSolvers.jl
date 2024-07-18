#!/bin/bash
#PBS -P np01
#PBS -q {{q}} 
#PBS -l walltime={{walltime}}
#PBS -l ncpus={{ncpus}}
#PBS -l mem={{mem}}
#PBS -N {{{name}}}
#PBS -l wd
#PBS -o {{{o}}}
#PBS -e {{{e}}} 

source {{{modules}}}

julia --project={{{projectdir}}} -O3 -e\
  '
  using Scalability
  using Gridap, GridapDistributed, PartitionedArrays, GridapSolvers, GridapPETSc
  using FileIO, BSON
  '

mpiexec -n {{ncpus}} julia --project={{{projectdir}}} -O3 -e\
  '
  using Scalability;
  stokes_main(;
    nr={{nr}},
    np={{np}},
    nc={{nc}},
    title="{{{title}}}",
  )
  '
