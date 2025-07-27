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
using PartitionedArrays: getany

using GridapPETSc
using GridapPETSc.PETSC
using GridapPETSc: PETScLinearSolverNS

using GridapSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.SolverInterfaces
using GridapSolvers.LinearSolvers
using GridapSolvers.PatchBasedSmoothers

include("PETScUtils.jl")
include("PETScCaches.jl")
include("ElasticitySolvers.jl")
include("HPDDMLinearSolvers.jl")
include("HipmairXuSolvers.jl")

end # module