
struct BlockFEOperator{NB,SB,P} <: FEOperator
  global_op :: FEOperator
  block_ids :: Vector{CartesianIndex{2}}
  block_ops :: Vector{FEOperator}
  nonlinear :: Vector{Bool}
end

function BlockFEOperator(
  res::Vector{<:Union{<:Function,Missing,Nothing}},
  jac::Matrix{<:Union{<:Function,Missing,Nothing}},
  args...; nonlinear = fill(true,size(res))
)
  keep(x) = !ismissing(x) && !isnothing(x)
  ids = findall(keep, res)
  @assert ids == findall(keep, jac)
  _res = [res[I] for I in ids]
  _jac = [jac[I] for I in ids]
  return BlockFEOperator(_res,_jac,ids,args...; nonlinear = nonlinear[ids])
end

function BlockFEOperator(
  contributions :: Vector{<:Tuple{Any,Function,Function}}, args...; kwargs...
)
  ids = [CartesianIndex(c[1]) for c in contributions]
  res = [c[2] for c in contributions]
  jac = [c[3] for c in contributions]
  return BlockFEOperator(res,jac,ids,args...;kwargs...)
end

function BlockFEOperator(
  res::Vector{<:Function},
  jac::Vector{<:Function},
  ids::Vector{CartesianIndex{2}},
  trial::BlockFESpaceTypes,
  test ::BlockFESpaceTypes;
  kwargs...
)
  assem = SparseMatrixAssembler(test,trial)
  return BlockFEOperator(res,jac,ids,trial,test,assem;kwargs...)
end

# TODO: I think nonlinear should not be a kwarg, since its the whole point of this operator
function BlockFEOperator(
  res::Vector{<:Function},
  jac::Vector{<:Function},
  ids::Vector{CartesianIndex{2}},
  trial::BlockFESpaceTypes{NB,SB,P},
  test::BlockFESpaceTypes{NB,SB,P},
  assem::MultiField.BlockSparseMatrixAssembler;
  nonlinear::Vector{Bool}=fill(true,length(res))
) where {NB,SB,P}
  ranges = MultiField.get_block_ranges(NB,SB,P)
  global_res = residual_from_blocks(ids,ranges,res)
  global_jac = jacobian_from_blocks(ids,ranges,jac)
  global_op  = FEOperator(global_res,global_jac,trial,test,assem)

  block_ops = map(FEOperator,res,jac,blocks(trial),blocks(test),blocks(assem))
  return BlockFEOperator{NB,SB,P}(global_op,block_ids,block_ops,nonlinear)
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

function Algebra.jacobian!(A::AbstractBlockMatrix,op::BlockFEOperator,u)
  A_blocks = blocks(A)
  for (i,I) in enumerate(op.block_ids)
    if op.nonlinear[i]
      jacobian!(A_blocks[I],op.block_ops[i],u)
    end
  end
  return A
end

# Private methods

function BlockArrays.blocks(a::MultiField.BlockSparseMatrixAssembler)
  return a.block_assemblers
end

function BlockArrays.blocks(f::MultiFieldFESpace{<:BlockMultiFieldStyle{NB,SB,P}}) where {NB,SB,P}
  block_ranges = MultiField.get_block_ranges(NB,SB,P)
  block_spaces = map(block_ranges) do range
    (length(range) == 1) ? f[range[1]] : MultiFieldFESpace(f.spaces[range])
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
      global_sf_spaces = f.field_fe_space[range]
      local_sf_spaces  = to_parray_of_arrays(map(local_views,global_sf_spaces))
      local_mf_spaces  = map(MultiFieldFESpace,local_sf_spaces)
      vector_type = GridapDistributed._find_vector_type(local_mf_spaces,gids)
      space = DistributedMultiFieldFESpace(global_sf_spaces,local_mf_spaces,gids,vector_type)
    end
    space
  end
  return block_spaces
end

function liform_from_blocks(ids, ranges, liforms)
  function biform(v)
    c = DomainContribution()
    for (I,lf) in zip(ids, liforms)
      vk = v[ranges[I]]
      c += lf(uk,vk)
    end
    return c
  end
  return biform
end

function biform_from_blocks(ids, ranges, biforms)
  function biform(u,v)
    c = DomainContribution()
    for (I,bf) in zip(ids, biforms)
      uk = u[ranges[I[1]]]
      vk = v[ranges[I[2]]]
      c += bf(uk,vk)
    end
    return c
  end
  return biform
end

function triform_from_blocks(ids, ranges, triforms)
  function triform(u,du,dv)
    c = DomainContribution()
    for (I,tf) in zip(ids, triforms)
      duk = du[ranges[I[1]]]
      dvk = dv[ranges[I[2]]]
      c += tf(u,duk,dvk)
    end
    return c
  end
  return triform
end
