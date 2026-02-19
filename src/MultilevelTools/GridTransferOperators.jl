
struct RedistributionOperator{A,B}
  indices_to   :: A
  indices_from :: B
  reverse      :: Bool
end

Base.reverse(op::RedistributionOperator) = RedistributionOperator(op.indices_from, op.indices_to, !op.reverse)

function RedistributionOperator(
  model_to, space_to, space_from, glue::RedistributeGlue, reverse::Bool
)
  if !isnothing(space_to)
    dof_ids_to = partition(get_free_dof_ids(space_to))
    to_cell_to_to_lid = get_cell_dof_ids(space_to)
  else
    dof_ids_to, to_cell_to_to_lid = nothing, nothing, nothing
  end
  if !isnothing(space_from)
    dof_ids_from = partition(get_free_dof_ids(space_from))
    from_cell_to_from_lid = get_cell_dof_ids(space_from)
  else
    dof_ids_from, from_cell_to_from_lid = nothing, nothing, nothing
  end
  indices_from, indices_to = GridapDistributed.redistribute_indices(
    dof_ids_from, from_cell_to_from_lid, to_cell_to_to_lid, model_to, glue; reverse,
  )
  return RedistributionOperator(indices_to, indices_from, reverse)
end

function RedistributionOperator(sh::FESpaceHierarchyLevel,reverse::Bool)
  if !reverse
    model_to = get_model(sh)
    space_to = get_fe_space(sh)
    space_from = get_fe_space_before_redist(sh)
  else
    model_to = get_model_before_redist(sh)
    space_to = get_fe_space_before_redist(sh)
    space_from = get_fe_space(sh)
  end
  glue = sh.mh_level.red_glue
  return RedistributionOperator(model_to, space_to, space_from, glue, reverse)
end

function change_parts_indices(indices, new_parts)
  ng = !isnothing(indices) ? map(global_length,indices) : nothing
  l2g = !isnothing(indices) ? map(local_to_global, indices) : nothing
  l2o = !isnothing(indices) ? map(local_to_owner, indices) : nothing

  ng = emit(change_parts(ng,new_parts;default=0))
  new_l2g = change_parts(l2g, new_parts; default=Int[])
  new_l2o = change_parts(l2o, new_parts; default=Int32[])
  return map(LocalIndices,ng, new_parts, new_l2g, new_l2o)
end

function redistribution_cache(x_to, x_from, indices_to, indices_from)
  parts = linear_indices(indices_to)
  x_ids_from = change_parts_indices(!isnothing(x_from) ? partition(axes(x_from,1)) : nothing, parts)
  if !matching_local_indices(PRange(x_ids_from), PRange(indices_from))
    @assert matching_own_indices(PRange(x_ids_from), PRange(indices_from))
    x_ids_to = GridapDistributed.reindex_partition(x_ids_from, indices_from, indices_to)
  else
    x_ids_to = indices_to
  end
  T = isnothing(x_from) ? eltype(x_to) : eltype(x_from)
  values_from = change_parts(partition, x_from, parts; default = T[])
  values_to = change_parts(partition, x_to, parts; default = T[])
  caches = GridapDistributed.p_vector_redistribution_cache(values_from, indices_from, indices_to)
  return (values_to, values_from), (x_ids_to, x_ids_from), (x_to, x_from), caches
end

function redistribution_cache(x_to, op::RedistributionOperator, x_from)
  return redistribution_cache(x_to, x_from, op.indices_to, op.indices_from)
end

function GridapDistributed.redistribute!(x_to,op::RedistributionOperator,x_from,cache)
  values, indices, owners, caches = cache
  @assert (x_to, x_from) == owners
  t = GridapDistributed.redistribute!(values..., indices..., caches)
  wait(t)
  return x_to
end

function GridapDistributed.redistribute(x_to,op::RedistributionOperator,x_from)
  caches = redistribution_cache(x_to, op, x_from)
  return redistribute!(x_to,op,x_from,caches)
end

"""
"""
struct DistributedGridTransferOperator{T,R,M,A,B}
  sh     :: A
  cache  :: B

  function DistributedGridTransferOperator(
    op_type::Symbol,redist::Bool,restriction_method::Symbol,sh::FESpaceHierarchy,cache
  )
    T = typeof(Val(op_type))
    R = typeof(Val(redist))
    M = typeof(Val(restriction_method))
    A = typeof(sh)
    B = typeof(cache)
    new{T,R,M,A,B}(sh,cache)
  end
end

### Constructors

"""
"""
function RestrictionOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int;kwargs...)
  return DistributedGridTransferOperator(lev,sh,qdegree,:restriction;kwargs...)
end

"""
"""
function ProlongationOperator(lev::Int,sh::FESpaceHierarchy,qdegree::Int;kwargs...)
  return DistributedGridTransferOperator(lev,sh,qdegree,:prolongation;kwargs...)
end

function DistributedGridTransferOperator(
  lev::Int,sh::FESpaceHierarchy,qdegree::Int,op_type::Symbol;
  mode::Symbol=:solution,restriction_method::Symbol=:projection,
  solver=LUSolver()
)
  @check lev < num_levels(sh)
  @check op_type ∈ [:restriction, :prolongation]
  @check mode ∈ [:solution, :residual]
  @check restriction_method ∈ [:projection, :interpolation, :dof_mask]

  # Refinement
  if (op_type == :prolongation) || (restriction_method ∈ [:interpolation,:dof_mask])
    cache_refine = _get_interpolation_cache(lev,sh,qdegree,mode)
  elseif mode == :solution
    cache_refine = _get_projection_cache(lev,sh,qdegree,mode,solver)
  else
    cache_refine = _get_dual_projection_cache(lev,sh,qdegree,solver)
    restriction_method = :dual_projection
  end

  # Redistribution
  redist = has_redistribution(sh,lev)
  cache_redist = _get_redistribution_cache(lev,sh,mode,op_type,restriction_method,cache_refine)

  cache = cache_refine, cache_redist
  return DistributedGridTransferOperator(op_type,redist,restriction_method,sh,cache)
end

function _get_interpolation_cache(lev::Int,sh::FESpaceHierarchy,qdegree,mode::Symbol)
  cparts = get_level_parts(sh,lev+1)

  if i_am_in(cparts)
    model_h = get_model_before_redist(sh,lev)
    Uh   = get_fe_space_before_redist(sh,lev)
    fv_h = zero_free_values(Uh)
    dv_h = (mode == :solution) ? get_dirichlet_dof_values(Uh) : zero_dirichlet_values(Uh)

    UH   = get_fe_space(sh,lev+1)
    fv_H = zero_free_values(UH)
    dv_H = (mode == :solution) ? get_dirichlet_dof_values(UH) : zero_dirichlet_values(UH)

    cache_refine = model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H
  else
    model_h = get_model_before_redist(sh,lev)
    Uh      = get_fe_space_before_redist(sh,lev)
    cache_refine = model_h, Uh, nothing, nothing, nothing, nothing, nothing
  end

  return cache_refine
end

function _get_projection_cache(lev::Int,sh::FESpaceHierarchy,qdegree,mode::Symbol,solver)
  cparts = get_level_parts(sh,lev+1)

  if i_am_in(cparts)
    model_h = get_model_before_redist(sh,lev)
    Uh   = get_fe_space_before_redist(sh,lev)
    Ωh   = Triangulation(model_h)
    fv_h = zero_free_values(Uh)
    dv_h = (mode == :solution) ? get_dirichlet_dof_values(Uh) : zero_dirichlet_values(Uh)

    model_H = get_model(sh,lev+1)
    UH   = get_fe_space(sh,lev+1)
    VH   = get_fe_space(sh,lev+1)
    ΩH   = Triangulation(model_H)
    dΩH  = Measure(ΩH,qdegree)
    dΩhH = Measure(ΩH,Ωh,qdegree)

    aH(u,v)  = ∫(v⋅u)*dΩH
    lH(v,uh) = ∫(v⋅uh)*dΩhH
    assem    = SparseMatrixAssembler(UH,VH)

    fv_H   = zero_free_values(UH)
    dv_H   = zero_dirichlet_values(UH)
    u0     = FEFunction(UH,fv_H)      # Zero at free dofs
    u00    = FEFunction(UH,fv_H,dv_H) # Zero everywhere

    u_dir  = (mode == :solution) ? u0 : u00
    u,v    = get_trial_fe_basis(UH), get_fe_basis(VH)
    data   = collect_cell_matrix_and_vector(UH,VH,aH(u,v),lH(v,u00),u_dir)
    AH,bH0 = assemble_matrix_and_vector(assem,data)
    AH_ns  = numerical_setup(symbolic_setup(solver,AH),AH)
    xH     = allocate_in_domain(AH); fill!(xH,zero(eltype(xH)))
    bH     = copy(bH0)

    cache_refine = model_h, Uh, fv_h, dv_h, VH, AH_ns, lH, xH, bH, bH0, assem
  else
    model_h = get_model_before_redist(sh,lev)
    Uh      = get_fe_space_before_redist(sh,lev)
    cache_refine = model_h, Uh, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
  end

  return cache_refine
end

function _get_dual_projection_cache(lev::Int,sh::FESpaceHierarchy,qdegree,solver)
  cparts = get_level_parts(sh,lev+1)

  if i_am_in(cparts)
    model_h = get_model_before_redist(sh,lev)
    Uh   = get_fe_space_before_redist(sh,lev)
    Ωh   = Triangulation(model_h)
    dΩh  = Measure(Ωh,qdegree)
    uh   = FEFunction(Uh,zero_free_values(Uh),zero_dirichlet_values(Uh))

    model_H = get_model(sh,lev+1)
    UH   = get_fe_space(sh,lev+1)
    ΩH   = Triangulation(model_H)
    dΩhH = Measure(ΩH,Ωh,qdegree)

    Mh = assemble_matrix((u,v)->∫(v⋅u)*dΩh,Uh,Uh)
    Mh_ns = numerical_setup(symbolic_setup(solver,Mh),Mh)

    assem = SparseMatrixAssembler(UH,UH)
    rh = allocate_in_domain(Mh); fill!(rh,zero(eltype(rh)))
    cache_refine = model_h, Uh, UH, Mh_ns, rh, uh, assem, dΩhH
  else
    model_h = get_model_before_redist(sh,lev)
    Uh      = get_fe_space_before_redist(sh,lev)
    cache_refine = model_h, Uh, nothing, nothing, nothing, nothing, nothing, nothing
  end

  return cache_refine
end

function _get_redistribution_cache(lev::Int,sh::FESpaceHierarchy,mode::Symbol,op_type::Symbol,restriction_method::Symbol,cache_refine)
  redist = has_redistribution(sh,lev)
  if !redist 
    cache_redist = nothing
    return cache_redist
  end

  Uh_red      = get_fe_space(sh,lev)
  fv_h_red    = zero_free_values(Uh_red)

  if (op_type == :prolongation)
    model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
    op = RedistributionOperator(sh[lev],false)
    cache_exchange = redistribution_cache(fv_h_red, op, fv_h)
  elseif restriction_method ∈ [:interpolation,:dof_mask]
    model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
    op = RedistributionOperator(sh[lev],true)
    cache_exchange = redistribution_cache(fv_h, op, fv_h_red)
  elseif restriction_method == :projection
    model_h, Uh, fv_h, dv_h, VH, AH, lH, xH, bH, bH0, assem = cache_refine
    op = RedistributionOperator(sh[lev],true)
    cache_exchange = redistribution_cache(fv_h, op, fv_h_red)
  else
    model_h, Uh, UH, Mh, rh, uh, assem, dΩhH = cache_refine
    fv_h = isa(uh,Nothing) ? nothing : get_free_dof_values(uh)
    op = RedistributionOperator(sh[lev],true)
    cache_exchange = redistribution_cache(fv_h, op, fv_h_red)
  end

  cache_redist = fv_h_red, op, cache_exchange
  return cache_redist
end

# TODO: Please replace this type of functions by a map functionality over hierarchies
function setup_transfer_operators(sh::FESpaceHierarchy,qdegree;kwargs...)
  prolongations = setup_prolongation_operators(sh,qdegree;kwargs...)
  restrictions  = setup_restriction_operators(sh,qdegree;kwargs...)
  return restrictions, prolongations
end

function setup_prolongation_operators(sh::FESpaceHierarchy,qdegree::Integer;kwargs...)
  qdegrees = Fill(qdegree,num_levels(sh))
  return setup_prolongation_operators(sh,qdegrees;kwargs...)
end

function setup_prolongation_operators(sh::FESpaceHierarchy,qdegrees::AbstractArray{<:Integer};kwargs...)
  @check length(qdegrees) == num_levels(sh)
  map(view(linear_indices(sh),1:num_levels(sh)-1)) do lev
    qdegree = qdegrees[lev]
    ProlongationOperator(lev,sh,qdegree;kwargs...)
  end
end

function setup_restriction_operators(sh::FESpaceHierarchy,qdegree::Integer;kwargs...)
  qdegrees = Fill(qdegree,num_levels(sh))
  return setup_restriction_operators(sh,qdegrees;kwargs...)
end

function setup_restriction_operators(sh::FESpaceHierarchy,qdegrees::AbstractArray{<:Integer};kwargs...)
  @check length(qdegrees) == num_levels(sh)
  map(view(linear_indices(sh),1:num_levels(sh)-1)) do lev
    qdegree = qdegrees[lev]
    RestrictionOperator(lev,sh,qdegree;kwargs...)
  end
end

function update_transfer_operator!(
  op::DistributedGridTransferOperator,x::AbstractVector
)
  nothing
end

### Applying the operators:

# A) Prolongation, without redistribution
function LinearAlgebra.mul!(y::AbstractVector,A::DistributedGridTransferOperator{Val{:prolongation},Val{false}},x::AbstractVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine

  copy!(fv_H,x) # Matrix layout -> FE layout
  uH = FEFunction(UH,fv_H,dv_H)
  uh = interpolate!(uH,fv_h,Uh)
  copy!(y,fv_h) # FE layout -> Matrix layout

  return y
end

# B.1) Restriction, without redistribution, by interpolation
function LinearAlgebra.mul!(y::AbstractVector,A::DistributedGridTransferOperator{Val{:restriction},Val{false},Val{:interpolation}},x::AbstractVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine

  copy!(fv_h,x) # Matrix layout -> FE layout
  uh = FEFunction(Uh,fv_h,dv_h)
  uH = interpolate!(uh,fv_H,UH)
  copy!(y,fv_H) # FE layout -> Matrix layout

  return y
end

# B.2) Restriction, without redistribution, by projection
function LinearAlgebra.mul!(y::AbstractVector,A::DistributedGridTransferOperator{Val{:restriction},Val{false},Val{:projection}},x::AbstractVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, VH, AH_ns, lH, xH, bH, bH0, assem = cache_refine

  copy!(fv_h,x) # Matrix layout -> FE layout
  uh = FEFunction(Uh,fv_h,dv_h)
  v  = get_fe_basis(VH)
  vec_data = collect_cell_vector(VH,lH(v,uh))
  assemble_vector!(bH,assem,vec_data) # Matrix layout
  bH .+= bH0
  solve!(xH,AH_ns,bH)
  copy!(y,xH)
  
  return y
end

# B.3) Restriction, without redistribution, by dof selection (only nodal dofs)
function LinearAlgebra.mul!(y::AbstractVector,A::DistributedGridTransferOperator{Val{:restriction},Val{false},Val{:dof_mask}},x::AbstractVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine

  copy!(fv_h,x) # Matrix layout -> FE layout
  consistent!(fv_h) |> fetch
  restrict_dofs!(fv_H,fv_h,dv_h,Uh,UH,get_adaptivity_glue(model_h))
  copy!(y,fv_H) # FE layout -> Matrix layout

  return y
end

# C) Prolongation, with redistribution
function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{Val{:prolongation},Val{true}},x::Union{PVector,Nothing})
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  fv_h_red, op_redist, cache_exchange = cache_redist

  # 1 - Interpolate in coarse partition
  if !isa(x,Nothing)
    copy!(fv_H,x) # Matrix layout -> FE layout
    uH = FEFunction(UH,fv_H,dv_H)
    uh = interpolate!(uH,fv_h,Uh)
  end

  # 2 - Redistribute from coarse partition to fine partition
  redistribute!(fv_h_red,op_redist,fv_h,cache_exchange)
  copy!(y,fv_h_red) # FE layout -> Matrix layout

  return y
end

# D.1) Restriction, with redistribution, by interpolation
function LinearAlgebra.mul!(y::Union{PVector,Nothing},A::DistributedGridTransferOperator{Val{:restriction},Val{true},Val{:interpolation}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  fv_h_red, op_redist, cache_exchange = cache_redist

  # 1 - Redistribute from fine partition to coarse partition
  copy!(fv_h_red,x)
  consistent!(fv_h_red) |> fetch
  redistribute!(fv_h,op_redist,fv_h_red,cache_exchange)

  # 2 - Interpolate in coarse partition
  if !isa(y,Nothing)
    uh = FEFunction(Uh,fv_h,dv_h)
    uH = interpolate!(uh,fv_H,UH)
    copy!(y,fv_H) # FE layout -> Matrix layout
  end

  return y
end

# D.2) Restriction, with redistribution, by projection
function LinearAlgebra.mul!(y::Union{PVector,Nothing},A::DistributedGridTransferOperator{Val{:restriction},Val{true},Val{:projection}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, VH, AH_ns, lH, xH, bH, bH0, assem = cache_refine
  fv_h_red, op_redist, cache_exchange = cache_redist

  # 1 - Redistribute from fine partition to coarse partition
  copy!(fv_h_red,x)
  consistent!(fv_h_red) |> fetch
  redistribute!(fv_h,op_redist,fv_h_red,cache_exchange)

  # 2 - Solve f2c projection coarse partition
  if !isa(y,Nothing)
    consistent!(fv_h) |> fetch
    uh = FEFunction(Uh,fv_h,dv_h)
    v  = get_fe_basis(VH)
    vec_data = collect_cell_vector(VH,lH(v,uh))
    copy!(bH,bH0)
    assemble_vector_add!(bH,assem,vec_data) # Matrix layout
    solve!(xH,AH_ns,bH)
    copy!(y,xH)
  end

  return y
end

# D.3) Restriction, with redistribution, by dof selection (only nodal dofs)
function LinearAlgebra.mul!(y::Union{PVector,Nothing},A::DistributedGridTransferOperator{Val{:restriction},Val{true},Val{:dof_mask}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  fv_h_red, op_redist, cache_exchange = cache_redist

  # 1 - Redistribute from fine partition to coarse partition
  copy!(fv_h_red,x)
  consistent!(fv_h_red) |> fetch
  redistribute!(fv_h,op_redist,fv_h_red,cache_exchange)

  # 2 - Interpolate in coarse partition
  if !isa(y,Nothing)
    consistent!(fv_h) |> fetch
    restrict_dofs!(fv_H,fv_h,dv_h,Uh,UH,get_adaptivity_glue(model_h))
    copy!(y,fv_H) # FE layout -> Matrix layout
  end

  return y 
end

###############################################################

function LinearAlgebra.mul!(y::AbstractVector,A::DistributedGridTransferOperator{Val{:restriction},Val{false},Val{:dual_projection}},x::AbstractVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, VH, Mh_ns, rh, uh, assem, dΩhH = cache_refine
  fv_h = get_free_dof_values(uh)

  solve!(rh,Mh_ns,x)
  copy!(fv_h,rh)
  v = get_fe_basis(VH)
  assemble_vector!(y,assem,collect_cell_vector(VH,∫(v⋅uh)*dΩhH))
  
  return y
end

function LinearAlgebra.mul!(y::PVector,A::DistributedGridTransferOperator{Val{:restriction},Val{false},Val{:dual_projection}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, VH, Mh_ns, rh, uh, assem, dΩhH = cache_refine
  fv_h = get_free_dof_values(uh)

  solve!(rh,Mh_ns,x)
  copy!(fv_h,rh)
  consistent!(fv_h) |> fetch
  v = get_fe_basis(VH)
  assemble_vector!(y,assem,collect_cell_vector(VH,∫(v⋅uh)*dΩhH))
  
  return y
end

function LinearAlgebra.mul!(y::Union{PVector,Nothing},A::DistributedGridTransferOperator{Val{:restriction},Val{true},Val{:dual_projection}},x::PVector)
  cache_refine, cache_redist = A.cache
  model_h, Uh, VH, Mh_ns, rh, uh, assem, dΩhH = cache_refine
  fv_h_red, op_redist, cache_exchange = cache_redist
  fv_h = isa(uh,Nothing) ? nothing : get_free_dof_values(uh)

  # 1 - Redistribute from fine partition to coarse partition
  copy!(fv_h_red,x)
  consistent!(fv_h_red) |> fetch
  redistribute!(fv_h,op_redist,fv_h_red,cache_exchange)

  # 2 - Solve f2c projection coarse partition
  if !isa(y,Nothing)
    solve!(rh,Mh_ns,fv_h)
    copy!(fv_h,rh)
    consistent!(fv_h) |> fetch
    v = get_fe_basis(VH)
    assemble_vector!(y,assem,collect_cell_vector(VH,∫(v⋅uh)*dΩhH))
  end

  return y
end
