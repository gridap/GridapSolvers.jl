module Scalability

using FileIO

using Gridap, PartitionedArrays, GridapDistributed, GridapSolvers, GridapPETSc
using Gridap.Algebra, Gridap.MultiField

using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools, GridapSolvers.BlockSolvers

include("utils.jl")
include("stokes.jl")
include("stokes_gmg.jl")

export stokes_main, stokes_gmg_main

end # module