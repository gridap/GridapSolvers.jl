
struct BlockFEOperator{NB,SB,P} <: FEOperator
  global_op    :: FEOperator
  block_ops    :: Matrix{<:Union{<:FEOperator,Missing,Nothing}}
  is_nonlinear :: Matrix{Bool}
end

const BlockFESpaceTypes{NB,SB,P} = Union{<:MultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}},<:GridapDistributed.DistributedMultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}}}

function BlockFEOperator(
  res::Matrix{<:Union{<:Function,Missing,Nothing}},
  jac::Matrix{<:Union{<:Function,Missing,Nothing}},
  trial::BlockFESpaceTypes,
  test::BlockFESpaceTypes;
  kwargs...
)
  assem = SparseMatrixAssembler(test,trial)
  return BlockFEOperator(res,jac,trial,test,assem)
end

function BlockFEOperator(
  res::Matrix{<:Union{<:Function,Missing,Nothing}},
  jac::Matrix{<:Union{<:Function,Missing,Nothing}},
  trial::BlockFESpaceTypes{NB,SB,P},
  test::BlockFESpaceTypes{NB,SB,P},
  assem::MultiField.BlockSparseMatrixAssembler{NB,NV,SB,P};
  is_nonlinear::Matrix{Bool}=fill(true,(NB,NB))
) where {NB,NV,SB,P}
  @check size(res,1) == size(jac,1) == NB
  @check size(res,2) == size(jac,2) == NB

  global_res = residual_from_blocks(NB,SB,P,res)
  global_jac = jacobian_from_blocks(NB,SB,P,jac)
  global_op  = FEOperator(global_res,global_jac,trial,test,assem)

  trial_blocks = blocks(trial)
  test_blocks  = blocks(test)
  assem_blocks = blocks(assem)
  block_ops = map(CartesianIndices(res)) do I
    if !ismissing(res[I]) && !isnothing(res[I])
      FEOperator(res[I],jac[I],test_blocks[I[1]],trial_blocks[I[2]],assem_blocks[I])
    end
  end
  return BlockFEOperator{NB,SB,P}(global_op,block_ops,is_nonlinear)
end

# BlockArrays API

BlockArrays.blocks(op::BlockFEOperator) = op.block_ops

# FEOperator API

FESpaces.get_test(op::BlockFEOperator) = get_test(op.global_op)
FESpaces.get_trial(op::BlockFEOperator) = get_trial(op.global_op)
Algebra.allocate_residual(op::BlockFEOperator,u) = allocate_residual(op.global_op,u)
Algebra.residual(op::BlockFEOperator,u) = residual(op.global_op,u)
Algebra.allocate_jacobian(op::BlockFEOperator,u) = allocate_jacobian(op.global_op,u)
Algebra.jacobian(op::BlockFEOperator,u) = jacobian(op.global_op,u)
Algebra.residual!(b::AbstractVector,op::BlockFEOperator,u) = residual!(b,op.global_op,u)

function Algebra.jacobian!(A::AbstractBlockMatrix,op::BlockFEOperator{NB},u) where NB
  map(blocks(A),blocks(op),op.is_nonlinear) do A,op,nl
    if nl
      residual!(A,op,u)
    end
  end
  return A
end

# Private methods

function residual_from_blocks(NB,SB,P,block_residuals)
  function res(u,v)
    block_ranges = MultiField.get_block_ranges(NB,SB,P)
    block_u = map(r -> (length(r) == 1) ? u[r[1]] : Tuple(u[r]), block_ranges)
    block_v = map(r -> (length(r) == 1) ? v[r[1]] : Tuple(v[r]), block_ranges)
    block_contrs = map(CartesianIndices(block_residuals)) do I
      if !ismissing(block_residuals[I]) && !isnothing(block_residuals[I])
        block_residuals[I](block_u[I[2]],block_v[I[1]])
      end
    end
    return add_block_contribs(block_contrs)
  end
  return res
end

function jacobian_from_blocks(NB,SB,P,block_jacobians)
  function jac(u,du,dv)
    block_ranges = MultiField.get_block_ranges(NB,SB,P)
    block_u  = map(r -> (length(r) == 1) ?  u[r[1]] : Tuple(u[r]) , block_ranges)
    block_du = map(r -> (length(r) == 1) ? du[r[1]] : Tuple(du[r]), block_ranges)
    block_dv = map(r -> (length(r) == 1) ? dv[r[1]] : Tuple(dv[r]), block_ranges)
    block_contrs = map(CartesianIndices(block_jacobians)) do I
      if !ismissing(block_jacobians[I]) && !isnothing(block_jacobians[I])
        block_jacobians[I](block_u[I[2]],block_du[I[2]],block_dv[I[1]])
      end
    end
    return add_block_contribs(block_contrs)
  end
  return jac
end

function add_block_contribs(contrs)
  c = contrs[1]
  for ci in contrs[2:end]
    if !ismissing(ci) && !isnothing(ci)
      c = c + ci
    end
  end
  return c
end

function BlockArrays.blocks(a::MultiField.BlockSparseMatrixAssembler)
  return a.block_assemblers
end

function BlockArrays.blocks(f::MultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_spaces = map(block_ranges) do range
    (length(range) == 1) ? f[range[1]] : MultiFieldFESpace(f[range])
  end
  return block_spaces
end

function BlockArrays.blocks(f::GridapDistributed.DistributedMultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  block_gids   = blocks(get_free_dof_ids(f))

  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_spaces = map(block_ranges,block_gids) do range, gids
    if (length(range) == 1) 
      space = f[range[1]]
    else
      global_sf_spaces = f[range]
      local_sf_spaces  = GridapDistributed.to_parray_of_arrays(map(local_views,global_sf_spaces))
      local_mf_spaces  = map(MultiFieldFESpace,local_sf_spaces)
      vector_type = GridapDistributed._find_vector_type(local_mf_spaces,gids)
      space = MultiFieldFESpace(global_sf_spaces,local_mf_spaces,gids,vector_type)
    end
    space
  end
  return block_spaces
end
