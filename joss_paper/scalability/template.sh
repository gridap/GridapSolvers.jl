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

mpiexec -n {{ncpus}} julia --project={{{projectdir}}} -O3 --check-bounds=no -e\
  '
  using Scalability;
  stokes_main(;
    nr={{nr}},
    np={{np}},
    nc={{nc}},
    np_per_level={{np_per_level}},
    title="{{{title}}}",
  )
  '
