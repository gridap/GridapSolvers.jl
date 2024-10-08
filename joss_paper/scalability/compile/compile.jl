using PackageCompiler

create_sysimage(
  [
   "Gridap","GridapDistributed","GridapPETSc","GridapSolvers","PartitionedArrays",
   "BSON","DrWatson","FileIO"
  ],
  sysimage_path=joinpath(@__DIR__,"..","Scalability.so"),
  precompile_execution_file=joinpath(@__DIR__,"warmup.jl")
)
