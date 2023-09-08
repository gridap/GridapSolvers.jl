module MultilevelTools

using MPI
using LinearAlgebra
using FillArrays
using BlockArrays
using IterativeSolvers

using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Adaptivity
using PartitionedArrays

using GridapDistributed
using GridapDistributed: redistribute_cell_dofs, redistribute_cell_dofs!, get_redistribute_cell_dofs_cache
using GridapDistributed: redistribute_free_values, redistribute_free_values!, get_redistribute_free_values_cache
using GridapDistributed: redistribute_fe_function
using GridapDistributed: get_old_and_new_parts
using GridapDistributed: generate_subparts, local_views

export allocate_col_vector, allocate_row_vector
export change_parts, num_parts, i_am_in
export generate_level_parts, generate_subparts

export ModelHierarchy
export num_levels, get_level, get_level_parts
export get_model, get_model_before_redist, has_refinement, has_redistribution

export FESpaceHierarchy
export get_fe_space, get_fe_space_before_redist
export compute_hierarchy_matrices

export DistributedGridTransferOperator
export RestrictionOperator, ProlongationOperator
export setup_transfer_operators
export mul!

include("Algebra.jl")
include("SubpartitioningTools.jl")
include("GridapFixes.jl")
include("RefinementTools.jl")
include("ModelHierarchies.jl")
include("FESpaceHierarchies.jl")
include("DistributedGridTransferOperators.jl")

end