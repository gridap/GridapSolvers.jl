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

julia --project={{{projectdir}}} -O3 -J{{{sysimage}}} -e\
  '
  @time "Preloading libraries (serial)" begin
    using Scalability;
    using Gridap, GridapDistributed, PartitionedArrays, GridapSolvers, GridapPETSc;
    using FileIO, BSON;
  end
  '

mpiexec -n {{ncpus}} julia --project={{{projectdir}}} -O3 -J{{{sysimage}}} -e\
  '
  @time "Loading libraries (MPI)" begin
    using Scalability;
  end

  petsc_options = """
    -ksp_type cg
    -ksp_rtol 1.0e-5
    -ksp_atol 1.0e-8
    -ksp_converged_reason
    -pc_type asm
    -pc_asm_overlap 1
    -pc_asm_type restrict
    -sub_ksp_type preonly
    -sub_pc_type lu
  """

  stokes_gmg_main(;
    nr={{nr}},
    np={{np}},
    nc={{nc}},
    np_per_level={{np_per_level}},
    title="{{{title}}}",
  )
  '
