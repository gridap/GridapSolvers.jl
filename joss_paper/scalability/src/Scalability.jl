module Scalability

using FileIO

using Gridap, PartitionedArrays, GridapDistributed, GridapSolvers, GridapPETSc
using Gridap.Algebra, Gridap.MultiField

using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.BlockSolvers

include("utils.jl")
include("stokes.jl")

export stokes_main

end # module