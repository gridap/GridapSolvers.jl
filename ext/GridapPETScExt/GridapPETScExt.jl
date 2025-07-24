module GridapPETScExt

using MPI
using LinearAlgebra
using SparseArrays
using SparseMatricesCSR

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.CellData
using Gridap.Arrays, Gridap.FESpaces, Gridap.MultiField
using Gridap.Fields, Gridap.ReferenceFEs

using GridapDistributed
using PartitionedArrays
using GridapPETSc

using GridapSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.SolverInterfaces
using GridapSolvers.LinearSolvers
using GridapSolvers.PatchBasedSmoothers

using PartitionedArrays: getany

include("PETScUtils.jl")
include("PETScCaches.jl")
include("ElasticitySolvers.jl")
include("HPDDMLinearSolvers.jl")
include("HipmairXuSolvers.jl")

end # module