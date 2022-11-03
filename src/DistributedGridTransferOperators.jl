

struct DistributedGridTransferOperator{T,R,A,B,C} <: Gridap.Refinement.GridTransferOperator
  sh     :: A
  ref_op :: B
  cache  :: C

  function DistributedGridTransferOperator(op_type::Symbol,redist::Bool,sh::FESpaceHierarchy,ref_op,cache)
    T = typeof(Val(op_type))
    R = typeof(Val(redist))
    A = typeof(sh)
    B = typeof(ref_op)
    C = typeof(cache)
    new{T,R,A,B,C}(sh,ref_op,cache)
  end
end

### Constructors

RestrictionOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int) = DistributedGridTransferOperator(lev,sh,qdegree,:restriction)
ProlongationOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int) = DistributedGridTransferOperator(lev,sh,qdegree,:prolongation)

function DistributedGridTransferOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int,op_type::Symbol)
  mh = sh.mh
  cparts = get_level_parts(mh,lev+1)
  @check lev < num_levels(mh)
  @check op_type ∈ [:restriction, :prolongation]

  Uh = get_fe_space_before_redist(sh,lev)
  Ωh = get_triangulation(Uh,get_model_before_redist(mh,lev))
  
  # Refinement
  if GridapP4est.i_am_in(cparts)
    UH = get_fe_space(sh,lev+1)
    ΩH = get_triangulation(UH,get_model(mh,lev+1))

    from,   to   = (op_type == :restriction) ? (Uh, UH) : (UH, Uh)
    Ω_from, Ω_to = (op_type == :restriction) ? (Ωh, ΩH) : (ΩH, Ωh)
    ref_op = ProjectionTransferOperator(from,Ω_from,to,Ω_to;qdegree=qdegree)
  else 
    ref_op = nothing
  end

  # Redistribution
  redist = has_redistribution(mh,lev)
  if redist
    Uh_red      = get_fe_space(sh,lev)
    model_h     = get_model_before_redist(mh,lev)
    model_h_red = get_model(mh,lev)
    fv_h        = PVector(0.0,Uh.gids)
    fv_h_red    = PVector(0.0,Uh_red.gids)
    glue = mh.levels[lev].red_glue

    cache = fv_h, Uh, fv_h_red, Uh_red, model_h, model_h_red, glue
  else
    cache = nothing
  end

  return DistributedGridTransferOperator(op_type,redist,sh,ref_op,cache)
end

function setup_transfer_operators(sh::FESpaceHierarchy, qdegree::Int)
  restrictions   = Vector{DistributedGridTransferOperator}(undef,num_levels(sh)-1)
  interpolations = Vector{DistributedGridTransferOperator}(undef,num_levels(sh)-1)
  for lev in 1:num_levels(sh)-1
    restrictions[lev]   = RestrictionOperator(lev,sh,qdegree)
    interpolations[lev] = InterpolationOperator(lev,sh,qdegree)
  end
  return restrictions, interpolations
end

### Applying the operators: 

## A) Without redistribution (same for interpolation/restriction)
function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{T,Val{false}},x::PVector) where T

  map_parts(y,A.ref_op,x) do y, ref_op, x
    mul!(y,ref_op,x)
  end

  return y
end

## B) Prolongation (coarse to fine), with redistribution
function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{Val{:prolongation},Val{true}},x::PVector)
  fv_h, Uh, fv_h_red, Uh_red, model_h, model_h_red, glue = A.cache

  # 1 - Solve c2f projection in coarse partition
  mul!(fv_h,ref_op,x)

  # 2 - Redistribute from coarse partition to fine partition
  redistribute_free_values!(fv_h_red,Uh_red,fv_h,Uh,model_h_red,glue;reverse=false)
  copy!(y,fv_h_red)

  return y
end

## C) Restriction (fine to coarse), with redistribution
function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{Val{:restriction},Val{true}},x::PVector)
  fv_h, Uh, fv_h_red, Uh_red, model_h, model_h_red, glue = A.cache

  # 1 - Redistribute from coarse partition to fine partition
  copy!(fv_h_red,x)
  redistribute_free_values!(fv_h,Uh,fv_h_red,Uh_red,model_h,glue;reverse=true)

  # 2 - Solve f2c projection in fine partition
  mul!(y,ref_op,fv_h)

  return y 
end
