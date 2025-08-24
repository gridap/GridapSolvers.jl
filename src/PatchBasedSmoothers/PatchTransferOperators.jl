
"""
    struct PatchProlongationOperator end

A `PatchProlongationOperator` is a modified prolongation operator such that given a coarse
solution `xH`, it maps it to a fine mesh solution `xh` given by

```
xh = Ih(xH) - yh
```

where `yh` is a subspace-based correction computed by solving local problems on coarse cell 
patches within the fine mesh.
"""
mutable struct PatchProlongationOperator{R,A,B}
  sh    :: A
  assem :: B
  lhs   :: Union{Nothing,Function}
  rhs   :: Union{Nothing,Function}
  is_nonlinear :: Bool
  collect_factorizations :: Bool
  op_redist
  caches

  function PatchProlongationOperator{R}(
    sh,assem,lhs,rhs,is_nonlinear,collect_factorizations,op_redist,caches
  ) where R
    A, B = typeof(sh), typeof(assem)
    new{R,A,B}(sh,assem,lhs,rhs,is_nonlinear,collect_factorizations,op_redist,caches)
  end
end

@doc """
    function PatchProlongationOperator(
      lev   :: Integer,
      sh    :: FESpaceHierarchy,
      ptopo :: PatchTopology,
      lhs   :: Function,
      rhs   :: Function;
      is_nonlinear=false,
      collect_factorizations=false
    )

Returns an instance of `PatchProlongationOperator` for a given level `lev` and a given 
FESpaceHierarchy `sh`. The subspace-based correction on a solution `uH` is computed 
by solving local problems given by 

```
  lhs(u_i,v_i) = rhs(uH,v_i)  ∀ v_i ∈ V_i
```

where `V_i` is a restriction of `sh[lev]` onto the patches given by the PatchTopology `ptopo`.
"""
function PatchProlongationOperator(
  lev,sh,ptopo::Union{<:Geometry.PatchTopology,<:GridapDistributed.DistributedPatchTopology},
  lhs,rhs;is_nonlinear=false,collect_factorizations=false
)
  Vh = MultilevelTools.get_fe_space_before_redist(sh,lev)
  assem = FESpaces.PatchAssembler(ptopo,Vh,Vh,assembly=:star)
  PatchProlongationOperator(
    lev,sh,assem,lhs,rhs;is_nonlinear,collect_factorizations
  )
end

function PatchProlongationOperator(
  lev,sh,assem,lhs,rhs;is_nonlinear=false,collect_factorizations=false
)

  cache_refine = MultilevelTools._get_interpolation_cache(lev,sh,0,:residual)
  cache_redist = MultilevelTools._get_redistribution_cache(lev,sh,:residual,:prolongation,:interpolation,cache_refine)
  cache_patch = _get_patch_cache(lev,sh,assem,lhs,rhs,is_nonlinear,collect_factorizations,cache_refine)
  caches = cache_refine, cache_patch, cache_redist

  redist = has_redistribution(sh,lev)
  if redist
    op_redist = MultilevelTools.RedistributionOperator(sh[lev],true)
  else
    op_redist = nothing
  end
  R = typeof(Val(redist))
  return PatchProlongationOperator{R}(sh,assem,lhs,rhs,is_nonlinear,collect_factorizations,op_redist,caches)
end

function _get_patch_cache(lev,sh,assem,lhs,rhs,is_nonlinear,collect_factorizations,cache_refine)
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine

  cparts = get_level_parts(sh,lev+1)
  if i_am_in(cparts)
    
    xh = zero(Uh)
    uh = FEFunction(Uh,fv_h,dv_h)
    uH = FEFunction(UH,fv_H,dv_H)
    dx_h = zero_free_values(Uh)
    @check get_free_dof_values(uh) === fv_h # Check correct aliasing
    @check get_free_dof_values(uH) === fv_H # Check correct aliasing

    biform(u,v) = is_nonlinear ? lhs(xh,u,v) : lhs(u,v)
    liform(v) = is_nonlinear ? rhs(xh,uh,v) : rhs(uh,v)

    patch_mats = assemble_matrix(biform,assem,Uh,Uh)
    patch_rows, patch_cols, patch_f, caches = patch_solver_caches(
      assem, patch_mats; collect_factorizations
    )

    if isa(dx_h,PVector)
      patch_ids = map(eachindex,patch_f)
    else
      patch_ids = eachindex(patch_rows)
    end
    
    return uh, uH, xh, dx_h, liform, patch_rows, patch_cols, patch_f, patch_ids, caches
  else
    return nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing, nothing
  end
end

# Please make this a standard API or something
function MultilevelTools.update_transfer_operator!(op::PatchProlongationOperator,x::Union{AbstractVector,Nothing})
  cache_refine, cache_patch, cache_redist = op.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  uh, uH, xh, dx_h, _, _, _, _, patch_ids, _ = cache_patch

  if !op.is_nonlinear
    return # Nothing to update
  end

  if !isa(cache_redist,Nothing)
    fv_h_red, op_redist, cache_exchange = cache_redist
    copy!(fv_h_red,x)
    consistent!(fv_h_red) |> fetch
    redistribute(fv_h,op.op_redist,fv_h_red)
  else
    copy!(fv_h,x)
    consistent!(fv_h) |> fetch
  end

  if !isa(fv_h,Nothing)
    copy!(get_free_dof_values(xh),fv_h)
    @check get_free_dof_values(uh) === fv_h # Check correct aliasing
    biform(u,v) = op.lhs(xh,u,v)
    liform(v) = op.rhs(xh,uh,v)

    patch_mats = assemble_matrix(biform,op.assem,Uh,Uh)
    patch_rows, patch_cols, patch_f, caches = patch_solver_caches(
      op.assem, patch_mats; op.collect_factorizations
    )

    cache_patch = (uh, uH, xh, dx_h, liform, patch_rows, patch_cols, patch_f, patch_ids, caches)
    op.caches = (cache_refine, cache_patch, cache_redist)
  end
end

function LinearAlgebra.mul!(y::AbstractVector,A::PatchProlongationOperator{Val{false}},x::AbstractVector)
  cache_refine, cache_patch, cache_redist = A.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  uh, uH, _, dx_h, liform, _, patch_cols, patch_f, patch_ids, caches = cache_patch

  copy!(fv_H,x) # Matrix layout -> FE layout
  consistent!(fv_H) |> fetch
  interpolate!(uH,fv_h,Uh)

  @check get_free_dof_values(uh) === fv_h # Check correct aliasing
  patch_b = assemble_vector(liform, A.assem, Uh)
  solve_patch_overlapping!(
    dx_h, patch_cols, patch_f, patch_b, patch_ids, caches
  )
  
  fv_h .= fv_h .- dx_h
  copy!(y,fv_h)

  return y
end

function LinearAlgebra.mul!(y::PVector,A::PatchProlongationOperator{Val{true}},x::Union{PVector,Nothing})
  cache_refine, cache_patch, cache_redist = A.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  fv_h_red, op_redist, cache_exchange = cache_redist
  _, uH, _, dx_h, liform, _, patch_cols, patch_f, patch_ids, caches = cache_patch

  # 1 - Interpolate in coarse partition
  if !isa(x,Nothing)
    copy!(fv_H,x) # Matrix layout -> FE layout
    consistent!(fv_H) |> fetch
    interpolate!(uH,fv_h,Uh)

    patch_b = assemble_vector(liform, A.assem, Uh)
    solve_patch_overlapping!(
      dx_h, patch_cols, patch_f, patch_b, patch_ids, caches
    )

    fv_h .= fv_h .- dx_h
  end

  # 2 - Redistribute from coarse partition to fine partition
  redistribute!(fv_h_red,op_redist,fv_h,cache_exchange)
  copy!(y,fv_h_red) # FE layout -> Matrix layout

  return y
end

function setup_patch_prolongation_operators(
  sh,lhs,rhs,qdegrees;is_nonlinear=false,collect_factorizations=false
)
  map(view(linear_indices(sh),1:num_levels(sh)-1)) do lev
    qdegree = isa(qdegrees,Vector) ? qdegrees[lev] : qdegrees
    cparts = get_level_parts(sh,lev+1)
    if i_am_in(cparts)
      model = get_model_before_redist(sh,lev)
      ptopo = CoarsePatchTopology(model)
      Ω = Geometry.PatchTriangulation(model,ptopo)
      dΩ = Measure(Ω,qdegree)
      lhs_i = is_nonlinear ? (u,du,dv) -> lhs(u,du,dv,dΩ) : (u,v) -> lhs(u,v,dΩ)
      rhs_i = is_nonlinear ? (u,du,dv) -> rhs(u,du,dv,dΩ) : (u,v) -> rhs(u,v,dΩ)
    else
      ptopo, lhs_i, rhs_i = nothing, nothing, nothing
    end
    PatchProlongationOperator(
      lev,sh,ptopo,lhs_i,rhs_i;is_nonlinear,collect_factorizations
    )
  end
end

############################################################################################

mutable struct PatchRestrictionOperator{R,A,B}
  Ip :: A
  caches :: B

  function PatchRestrictionOperator{R}(
    Ip::PatchProlongationOperator{R},
    caches
  ) where R
    A = typeof(Ip)
    B = typeof(caches)
    new{R,A,B}(Ip,caches)
  end
end

function PatchRestrictionOperator(lev,sh,Ip,qdegree;solver=LUSolver())
  cache_refine = MultilevelTools._get_dual_projection_cache(lev,sh,qdegree,solver)
  cache_redist = MultilevelTools._get_redistribution_cache(lev,sh,:residual,:restriction,:dual_projection,cache_refine)
  cache_patch = Ip.caches[2]
  caches = cache_refine, cache_patch, cache_redist

  redist = has_redistribution(sh,lev)
  R = typeof(Val(redist))
  return PatchRestrictionOperator{R}(Ip,caches)
end

function MultilevelTools.update_transfer_operator!(op::PatchRestrictionOperator,x::Union{PVector,Nothing})
  cache_refine, cache_patch, cache_redist = op.caches
  op.caches = (cache_refine, op.Ip.caches[2], cache_redist)
  return nothing
end

function setup_patch_restriction_operators(sh,patch_prolongations,qdegrees;kwargs...)
  map(view(linear_indices(sh),1:num_levels(sh)-1)) do lev
    qdegree = isa(qdegrees,Vector) ? qdegrees[lev] : qdegrees
    Ip = patch_prolongations[lev]
    PatchRestrictionOperator(lev,sh,Ip,qdegree;kwargs...)
  end
end

function LinearAlgebra.mul!(y::AbstractVector,A::PatchRestrictionOperator{Val{false}},x::AbstractVector)
  cache_refine, cache_patch, _ = A.caches
  model_h, Uh, VH, Mh_ns, rh, _, assem, dΩhH = cache_refine
  uh, uH, _, dx_h, liform, _, patch_cols, patch_f, patch_ids, caches = cache_patch
  fv_h = get_free_dof_values(uh)

  copy!(fv_h,x)
  consistent!(fv_h) |> fetch
  patch_b = assemble_vector(liform, A.Ip.assem, Uh)
  solve_patch_overlapping!(
    dx_h, patch_cols, patch_f, patch_b, patch_ids, caches
  )
  fv_h .= fv_h .- dx_h

  solve!(rh,Mh_ns,fv_h)
  copy!(fv_h,rh)
  isa(y,PVector) && wait(consistent!(fv_h))
  v = get_fe_basis(VH)
  assemble_vector!(y,assem,collect_cell_vector(VH,∫(v⋅uh)*dΩhH))

  return y
end

function LinearAlgebra.mul!(y::Union{PVector,Nothing},A::PatchRestrictionOperator{Val{true}},x::PVector)
  cache_refine, cache_patch, cache_redist = A.caches
  model_h, Uh, VH, Mh_ns, rh, _, assem, dΩhH = cache_refine
  fv_h_red, op_redist, cache_exchange = cache_redist
  uh, uH, _, dx_h, liform, _, patch_cols, patch_f, patch_ids, caches = cache_patch
  fv_h = isa(uh,Nothing) ? nothing : get_free_dof_values(uh)

  copy!(fv_h_red,x)
  consistent!(fv_h_red) |> fetch
  redistribute!(fv_h,op_redist,fv_h_red,cache_exchange)

  if !isa(y,Nothing)
    consistent!(fv_h) |> fetch
    patch_b = assemble_vector(liform, A.Ip.assem, Uh)
    solve_patch_overlapping!(
      dx_h, patch_cols, patch_f, patch_b, patch_ids, caches
    )
    fv_h .= fv_h .- dx_h
    
    solve!(rh,Mh_ns,fv_h)
    copy!(fv_h,rh)
    consistent!(fv_h) |> fetch
    v = get_fe_basis(VH)
    assemble_vector!(y,assem,collect_cell_vector(VH,∫(v⋅uh)*dΩhH))
  end

  return y
end
