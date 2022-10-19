

struct InterGridTransferOperator{T,R,A,B,C}
  sh     :: A
  ref_op :: B
  cache  :: C

  function InterGridTransferOperator(op_type::Symbol,redist::Bool,sh::FESpaceHierarchy,ref_op,cache)
    T = typeof(Val(op_type))
    R = typeof(Val(redist))
    A = typeof(sh)
    B = typeof(ref_op)
    C = typeof(cache)
    new{T,R,A,B,C}(sh,cache)
  end
end

### Constructors

RestrictionOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int) = InterGridTransferOperator(lev,sh,qdegree,:restriction)
ProlongationOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int) = InterGridTransferOperator(lev,sh,qdegree,:prolongation)

function InterGridTransferOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int,op_type::Symbol)
  mh = sh.mh
  @check lev != num_levels(mh)
  @check op_type ∈ [:restriction, :prolongation]
  
  UH = get_space(sh,lev+1)
  Uh = get_space_before_redist(sh,lev)
  Uh_red = get_space(sh,lev)

  # Refinement
  from, to = (op_type == :restriction) ? (Uh, UH) : (UH, Uh)
  ref_op = RefinementTransferOperator(from,to;qdegree=qdegree)

  # Redistribution
  redist = has_redistribution(mh,lev)
  if redist
    model_h     = get_model_before_redist(mh,lev)
    model_h_red = get_model(mh,lev)
    fv_h        = PVector(0.0,Uh.gids)
    fv_h_red    = PVector(0.0,Uh_red.gids)

    cache = fv_h, Uh, fv_h_red, Uh_red, model_h, model_h_red, glue
  else
    cache = nothing
  end

  return InterGridTransferOperator(:op_type,redist,sh,ref_op,cache)
end

function setup_transfer_operators(sh::FESpaceHierarchy, qdegree::Int)
  restrictions   = Vector{InterGridTransferOperator}(undef,num_levels(sh)-1)
  interpolations = Vector{InterGridTransferOperator}(undef,num_levels(sh)-1)
  for lev in 1:num_levels(sh)-1
    restrictions[lev]   = RestrictionOperator(lev,sh,qdegree)
    interpolations[lev] = InterpolationOperator(lev,sh,qdegree)
  end
  return restrictions, interpolations
end

### Applying the operators: 

## A) Without redistribution (same for interpolation/restriction)
function LinearAlgebra.mul!(y::PVector,A::InterGridTransferOperator{T,Val{false}},x::PVector) where T

  map_parts(y,A.ref_op,x) do y, ref_op, x
    mul!(y,ref_op,x)
  end

  return y
end

## B) Prolongation (coarse to fine), with redistribution
function LinearAlgebra.mul!(y::PVector,A::InterGridTransferOperator{Val{:prolongation},Val{true}},x::PVector)
  fv_h, Uh, fv_h_red, Uh_red, model_h, model_h_red, glue = A.cache

  # 1 - Solve c2f projection in coarse partition
  mul!(fv_h,ref_op,x)

  # 2 - Redistribute from coarse partition to fine partition
  redistribute_free_values!(fv_h_red,Uh_red,fv_h,Uh,model_h_red,glue;reverse=false)
  copy!(y,fv_h_red)

  return y
end

## C) Restriction (fine to coarse), with redistribution
function LinearAlgebra.mul!(y::PVector,A::InterGridTransferOperator{Val{:restriction},Val{true}},x::PVector)
  fv_h, Uh, fv_h_red, Uh_red, model_h, model_h_red, glue = A.cache

  # 1 - Redistribute from coarse partition to fine partition
  copy!(fv_h_red,x)
  redistribute_free_values!(fv_h,Uh,fv_h_red,Uh_red,model_h,glue;reverse=true)

  # 2 - Solve f2c projection in fine partition
  mul!(y,ref_op,fv_h)

  return y 
end
