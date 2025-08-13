module MultilevelTools

using MPI
using LinearAlgebra
using FillArrays
using BlockArrays

using Gridap
using Gridap.Helpers, Gridap.Algebra, Gridap.Arrays, Gridap.Fields, Gridap.CellData
using Gridap.ReferenceFEs, Gridap.Geometry, Gridap.FESpaces, Gridap.Adaptivity, Gridap.MultiField

using Gridap.Adaptivity
using Gridap.Adaptivity: get_model

using PartitionedArrays, GridapDistributed

using Base: unsafe_getindex

using Gridap.FESpaces: BasisStyle, TestBasis, TrialBasis, SingleFieldFEBasis
using Gridap.MultiField: MultiFieldFEBasisComponent

using GridapDistributed: RedistributeGlue
using GridapDistributed: redistribute_cell_dofs, redistribute_cell_dofs!, get_redistribute_cell_dofs_cache
using GridapDistributed: redistribute_free_values, redistribute_free_values!, get_redistribute_free_values_cache
using GridapDistributed: redistribute_fe_function
using GridapDistributed: get_old_and_new_parts
using GridapDistributed: i_am_in, num_parts, change_parts, generate_subparts, local_views

export change_parts, num_parts, i_am_in
export generate_level_parts, generate_subparts

export HierarchicalArray
export num_levels, get_level_parts, with_level, matching_level_parts, unsafe_getindex

export ModelHierarchy, CartesianModelHierarchy
export num_levels, get_level, get_level_parts
export get_model, get_model_before_redist, has_refinement, has_redistribution

export FESpaceHierarchy
export get_fe_space, get_fe_space_before_redist
export compute_hierarchy_matrices

export TriangulationHierarchy
export get_triangulation_before_redist

export LocalProjectionMap

export DistributedGridTransferOperator
export RestrictionOperator, ProlongationOperator
export setup_transfer_operators, setup_prolongation_operators, setup_restriction_operators
export mul!

export MultiFieldTransferOperator

include("SubpartitioningTools.jl")
include("HierarchicalArrays.jl")
include("GridapFixes.jl")
include("RefinementTools.jl")

include("ModelHierarchies.jl")
include("TriangulationHierarchies.jl")
include("FESpaceHierarchies.jl")
include("LocalProjectionMaps.jl")
include("GridTransferOperators.jl")
include("MultiFieldTransferOperators.jl")

end