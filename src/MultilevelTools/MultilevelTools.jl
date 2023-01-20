module MultilevelTools

using MPI
using LinearAlgebra
using FillArrays
using IterativeSolvers
using Gridap
using Gridap.Helpers
using Gridap.Algebra
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.Adaptivity
using PartitionedArrays
using GridapDistributed

import LinearAlgebra: mul!
import GridapDistributed: local_views
import GridapP4est: i_am_in, i_am_main


export change_parts
export generate_level_parts
export generate_subparts

export redistribute_fe_function
export redistribute_free_values!

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

include("SubpartitioningTools.jl")
include("GridapDistributedExtensions.jl")
include("GridapFixes.jl")
include("RefinementTools.jl")
include("RedistributeTools.jl")
include("ModelHierarchies.jl")
include("FESpaceHierarchies.jl")
include("DistributedGridTransferOperators.jl")


end

