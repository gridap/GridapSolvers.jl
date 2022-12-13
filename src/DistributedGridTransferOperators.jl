

struct DistributedGridTransferOperator{T,R,A,B}
  sh     :: A
  cache  :: B

  function DistributedGridTransferOperator(op_type::Symbol,redist::Bool,sh::FESpaceHierarchy,cache)
    T = typeof(Val(op_type))
    R = typeof(Val(redist))
    A = typeof(sh)
    B = typeof(cache)
    new{T,R,A,B}(sh,cache)
  end
end

### Constructors

RestrictionOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int) = DistributedGridTransferOperator(lev,sh,qdegree,:restriction)
ProlongationOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int) = DistributedGridTransferOperator(lev,sh,qdegree,:prolongation)

function DistributedGridTransferOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int,op_type::Symbol)
  mh = sh.mh
  @check lev < num_levels(mh)
  @check op_type ∈ [:restriction, :prolongation]

  # Refinement
  if (op_type == :restriction)
    cache_refine = _get_restriction_cache(lev,sh,qdegree)
  else
    cache_refine = _get_prolongation_cache(lev,sh,qdegree)
  end

  # Redistribution
  redist = has_redistribution(mh,lev)
  cache_redist = _get_redistribution_cache(lev,sh)

  cache = cache_refine, cache_redist
  return DistributedGridTransferOperator(op_type,redist,sh,cache)
end

function _get_prolongation_cache(lev::Int,sh::FESpaceHierarchy,qdegree::Int)
  mh = sh.mh
  cparts = get_level_parts(mh,lev+1)

  if GridapP4est.i_am_in(cparts)
    model_h = get_model_before_redist(mh,lev)
    Uh = get_fe_space_before_redist(sh,lev)
    fv_h = PVector(0.0,Uh.gids)
    dv_h = get_dirichlet_dof_values(Uh) # Should this be zeros? 

    UH = get_fe_space(sh,lev+1)
    fv_H = PVector(0.0,UH.gids)
    dv_H = get_dirichlet_dof_values(UH)

    cache_refine = model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H
  else
    model_h = get_model_before_redist(mh,lev)
    Uh      = get_fe_space_before_redist(sh,lev)
    cache_refine = model_h, Uh, nothing, nothing, nothing, nothing, nothing
  end

  return cache_refine
end

function _get_restriction_cache(lev::Int,sh::FESpaceHierarchy,qdegree::Int)
  mh = sh.mh
  cparts = get_level_parts(mh,lev+1)

  if GridapP4est.i_am_in(cparts)
    model_h = get_model_before_redist(mh,lev)
    Uh = get_fe_space_before_redist(sh,lev)
    Ωh = get_triangulation(Uh,get_model_before_redist(mh,lev))
    fv_h = PVector(0.0,Uh.gids)
    dv_h = get_dirichlet_dof_values(Uh) # Should this be zeros? 

    UH   = get_fe_space(sh,lev+1)
    VH   = get_test_space(UH)
    ΩH   = get_triangulation(UH,get_model(mh,lev+1))
    dΩH  = Measure(ΩH,qdegree)
    dΩhH = Measure(ΩH,Ωh,qdegree)

    aH(u,v)  = ∫(v⋅u)*dΩH
    lH(v,uh) = ∫(v⋅uh)*dΩhH
    AH = assemble_matrix(aH,UH,VH)
    xH = PVector(0.0,AH.rows)

    cache_refine = model_h, Uh, fv_h, dv_h, VH, AH, lH, xH
  else
    model_h = get_model_before_redist(mh,lev)
    Uh      = get_fe_space_before_redist(sh,lev)
    cache_refine = model_h, Uh, nothing, nothing, nothing, nothing, nothing, nothing
  end

  return cache_refine
end

function _get_redistribution_cache(lev::Int,sh::FESpaceHierarchy)
  mh = sh.mh
  redist = has_redistribution(mh,lev)
  if redist
    Uh_red      = get_fe_space(sh,lev)
    model_h_red = get_model(mh,lev)
    fv_h_red    = PVector(0.0,Uh_red.gids)
    dv_h_red    = get_dirichlet_dof_values(Uh_red)
    glue        = mh.levels[lev].red_glue

    cache_redist = fv_h_red, dv_h_red, Uh_red, model_h_red, glue
  else
    cache_redist = nothing
  end
  return cache_redist
end

function setup_transfer_operators(sh::FESpaceHierarchy, qdegree::Int)
  mh = sh.mh
  restrictions  = Vector{DistributedGridTransferOperator}(undef,num_levels(sh)-1)
  prolongations = Vector{DistributedGridTransferOperator}(undef,num_levels(sh)-1)
  for lev in 1:num_levels(sh)-1
    parts = get_level_parts(mh,lev)
    if GridapP4est.i_am_in(parts)
      restrictions[lev]  = RestrictionOperator(lev,sh,qdegree)
      prolongations[lev] = ProlongationOperator(lev,sh,qdegree)
    end
  end
  return restrictions, prolongations
end

### Applying the operators: 

# A) Prolongation, without redistribution
function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{Val{:prolongation},Val{false}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine

  copy!(fv_H,x) # Matrix layout -> FE layout
  uH = FEFunction(UH,fv_H,dv_H)
  uh = interpolate!(uH,fv_h,Uh)
  copy!(y,fv_h) # FE layout -> Matrix layout

  return y
end

# B) Restriction, without redistribution
function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{Val{:restriction},Val{false}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, VH, AH, lH, xH = cache_refine

  copy!(fv_h,x) # Matrix layout -> FE layout
  uh = FEFunction(Uh,fv_h,dv_h)
  rhs(v) = lH(v,uh)
  bH = assemble_vector(rhs,VH) # Matrix layout
  IterativeSolvers.cg!(xH,AH,bH;reltol=1.0e-06)
  copy!(y,xH) # TO UNDERSTAND: Why can't we use directly y instead of xH?
  
  return y
end

# C) Prolongation, with redistribution
function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{Val{:prolongation},Val{true}},x::Union{PVector,Nothing})
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  fv_h_red, dv_h_red, Uh_red, model_h_red, glue = cache_redist

  # 1 - Solve c2f projection in coarse partition
  if !isa(x,Nothing)
    copy!(fv_H,x) # Matrix layout -> FE layout
    uH = FEFunction(UH,fv_H,dv_H)
    uh = interpolate!(uH,fv_h,Uh)
  end

  # 2 - Redistribute from coarse partition to fine partition
  redistribute_free_values!(fv_h_red,Uh_red,fv_h,dv_h,Uh,model_h_red,glue;reverse=false)
  copy!(y,fv_h_red) # FE layout -> Matrix layout

  return y
end

# D) Restriction, with redistribution
function LinearAlgebra.mul!(y::Union{PVector,Nothing},A::DistributedGridTransferOperator{Val{:restriction},Val{true}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, VH, AH, lH, xH = cache_refine
  fv_h_red, dv_h_red, Uh_red, model_h_red, glue = cache_redist

  # 1 - Redistribute from fine partition to coarse partition
  copy!(fv_h_red,x)
  redistribute_free_values!(fv_h,Uh,fv_h_red,dv_h_red,Uh_red,model_h,glue;reverse=true)

  # 2 - Solve f2c projection in fine partition
  if !isa(y,Nothing)
    uh = FEFunction(Uh,fv_h,dv_h)
    rhs(v) = lH(v,uh)
    bH = assemble_vector(rhs,VH) # Matrix layout
    IterativeSolvers.cg!(xH,AH,bH;reltol=1.0e-06)
    copy!(y,xH)
  end

  return y 
end
