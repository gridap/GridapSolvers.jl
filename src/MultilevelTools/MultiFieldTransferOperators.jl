
"""
"""
struct MultiFieldTransferOperator{T,A,B,C,D}
  Vh_in  :: A
  Vh_out :: B
  ops    :: C
  cache  :: D

  function MultiFieldTransferOperator(
    op_type::Symbol,Vh_in::A,Vh_out::B,ops::C,cache::D
  ) where {A,B,C,D}
    T = Val{op_type}
    new{T,A,B,C,D}(Vh_in,Vh_out,ops,cache)
  end
end

function MultiFieldTransferOperator(lev::Integer,sh::FESpaceHierarchy,operators;op_type=:prolongation)
  @check op_type in (:prolongation,:restriction)
  cparts = get_level_parts(sh,lev+1)
  Vh = get_fe_space(sh,lev)
  VH = i_am_in(cparts) ? get_fe_space(sh,lev+1) : nothing

  Vh_out, Vh_in = (op_type == :prolongation) ? (Vh,VH) : (VH,Vh)
  xh = isnothing(Vh_out) ? nothing : zero_free_values(Vh_out)
  yh = isnothing(Vh_in) ? nothing : zero_free_values(Vh_in)
  caches = xh, yh
  return MultiFieldTransferOperator(op_type,Vh_in,Vh_out,operators,caches)
end

function MultiFieldTransferOperator(sh::FESpaceHierarchy,operators;op_type=:prolongation)
  nlevs = num_levels(sh)
  @check all(map(a -> length(a) == nlevs-1, operators))

  mfops = Vector{MultiFieldTransferOperator}(undef,nlevs-1)
  for (lev,ops) in enumerate(zip(operators...))
    parts = get_level_parts(sh,lev)
    if i_am_in(parts)
      mfops[lev] = MultiFieldTransferOperator(lev,sh,ops;op_type)
    end
  end
  return mfops
end

function update_transfer_operator!(
  op::MultiFieldTransferOperator{Val{:prolongation}},x::Union{AbstractVector,Nothing}
)
  xh, _ = op.cache

  if !isnothing(x)
    copy!(xh,x)
    isa(xh,PVector) && wait(consistent!(xh))
  end

  for (i,op_i) in enumerate(op.ops)
    xh_i = isnothing(xh) ? nothing : MultiField.restrict_to_field(op.Vh_out,xh,i)
    update_transfer_operator!(op_i,xh_i)
  end
end

function update_transfer_operator!(
  op::MultiFieldTransferOperator{Val{:restriction}},y::Union{AbstractVector,Nothing}
)
  _, yh = op.cache

  if !isnothing(y)
    copy!(yh,y)
    isa(yh,PVector) && wait(consistent!(yh))
  end

  for (i,op_i) in enumerate(op.ops)
    yh_i = isnothing(yh) ? nothing : MultiField.restrict_to_field(op.Vh_in,yh,i)
    update_transfer_operator!(op_i,yh_i)
  end
end

function LinearAlgebra.mul!(
  x::Union{Nothing,<:AbstractVector},
  op::MultiFieldTransferOperator,
  y::Union{Nothing,<:AbstractVector}
)
  xh, yh = op.cache

  if !isnothing(yh)
    copy!(yh,y)
  end

  for (i,op_i) in enumerate(op.ops)
    xh_i = isnothing(xh) ? nothing : MultiField.restrict_to_field(op.Vh_out,xh,i)
    yh_i = isnothing(yh) ? nothing : MultiField.restrict_to_field(op.Vh_in,yh,i)
    LinearAlgebra.mul!(xh_i,op_i,yh_i)
  end

  if !isnothing(xh)
    copy!(x,xh)
    isa(x,PVector) && wait(consistent!(x))
  end

  return x
end
