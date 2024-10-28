module GridapPETScExt

using LinearAlgebra
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.CellData
using Gridap.Arrays, Gridap.FESpaces, Gridap.MultiField

using GridapDistributed
using PartitionedArrays
using GridapPETSc

using GridapSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.SolverInterfaces
using GridapSolvers.PatchBasedSmoothers

include("PETScUtils.jl")
include("PETScCaches.jl")
include("ElasticitySolvers.jl")
include("HipmairXuSolvers.jl")

end # module