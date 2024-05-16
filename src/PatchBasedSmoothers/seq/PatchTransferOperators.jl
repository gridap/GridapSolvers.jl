
"""
    struct PatchProlongationOperator end

A `PatchProlongationOperator` is a modified prolongation operator such that given a coarse
solution `xH` returns 

```
xh = Ih(xH) - yh
```

where `yh` is a subspace-based correction computed by solving local problems on coarse cells 
within the fine mesh.

"""
struct PatchProlongationOperator{R,A,B,C}
  sh :: A
  PD :: B
  lhs :: Union{Nothing,Function}
  rhs :: Union{Nothing,Function}
  is_nonlinear :: Bool
  caches :: C

  function PatchProlongationOperator{R}(sh,PD,lhs,rhs,is_nonlinear,caches) where R
    A, B, C = typeof(sh), typeof(PD), typeof(caches)
    new{R,A,B,C}(sh,PD,lhs,rhs,is_nonlinear,caches)
  end
end

@doc """
    function PatchProlongationOperator(
      lev :: Integer,
      sh  :: FESpaceHierarchy,
      PD  :: PatchDecomposition,
      lhs :: Function,
      rhs :: Function;
      is_nonlinear=false
    )

Returns an instance of `PatchProlongationOperator` for a given level `lev` and a given 
FESpaceHierarchy `sh`. The subspace-based correction on a solution `uH` is computed 
by solving local problems given by 

```
  lhs(u_i,v_i) = rhs(uH,v_i) ∀ v_i ∈ V_i
```

where `V_i` is the patch-local space indiced by the PatchDecomposition `PD`.

"""
function PatchProlongationOperator(lev,sh,PD,lhs,rhs;is_nonlinear=false)

  cache_refine = MultilevelTools._get_interpolation_cache(lev,sh,0,:residual)
  cache_redist = MultilevelTools._get_redistribution_cache(lev,sh,:residual,:prolongation,:interpolation,cache_refine)
  cache_patch = _get_patch_cache(lev,sh,PD,lhs,rhs,is_nonlinear,cache_refine)
  caches = cache_refine, cache_patch, cache_redist

  redist = has_redistribution(sh,lev)
  R = typeof(Val(redist))
  return PatchProlongationOperator{R}(sh,PD,lhs,rhs,is_nonlinear,caches)
end

function _get_patch_cache(lev,sh,PD,lhs,rhs,is_nonlinear,cache_refine)
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine

  cparts = get_level_parts(sh,lev+1)
  if i_am_in(cparts)
    # Patch-based correction fespace
    glue = sh[lev].mh_level.ref_glue
    patches_mask = get_coarse_node_mask(model_h,glue)
    cell_conformity = MultilevelTools.get_cell_conformity_before_redist(sh,lev)
    Ph = PatchFESpace(Uh,PD,cell_conformity;patches_mask)

    # Solver caches
    u, v = get_trial_fe_basis(Uh), get_fe_basis(Uh)
    contr = is_nonlinear ? lhs(zero(Uh),u,v) : lhs(u,v)
    matdata = collect_cell_matrix(Ph,Ph,contr)
    Ap_ns, Ap = map(local_views(Ph),matdata) do Ph, matdata
      assem = SparseMatrixAssembler(Ph,Ph)
      Ap    = assemble_matrix(assem,matdata)
      Ap_ns = numerical_setup(symbolic_setup(LUSolver(),Ap),Ap)
      return Ap_ns, Ap
    end |> tuple_of_arrays
    Ap = is_nonlinear ? Ap : nothing

    duh = zero(Uh)
    dxp, rp = zero_free_values(Ph), zero_free_values(Ph)
    return  Ph, Ap_ns, Ap, duh, dxp, rp
  else
    return nothing, nothing, nothing, nothing, nothing, nothing
  end
end

# Please make this a standard API or something
function MultilevelTools.update_transfer_operator!(op::PatchProlongationOperator,x::Union{PVector,Nothing})
  cache_refine, cache_patch, cache_redist = op.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  Ph, Ap_ns, Ap, duh, dxp, rp = cache_patch

  if !isa(cache_redist,Nothing)
    fv_h_red, dv_h_red, Uh_red, model_h_red, glue, cache_exchange = cache_redist
    copy!(fv_h_red,x)
    consistent!(fv_h_red) |> fetch
    GridapDistributed.redistribute_free_values(fv_h,Uh,fv_h_red,dv_h_red,Uh_red,model_h,glue;reverse=true)
  else
    copy!(fv_h,x)
  end

  if !isa(fv_h,Nothing)
    u, v = get_trial_fe_basis(Uh), get_fe_basis(Uh)
    contr = op.is_nonlinear ? op.lhs(FEFunction(Uh,fv_h),u,v) : op.lhs(u,v)
    matdata = collect_cell_matrix(Ph,Ph,contr)
    map(Ap_ns,Ap,local_views(Ph),matdata) do Ap_ns, Ap, Ph, matdata
      assem = SparseMatrixAssembler(Ph,Ph)
      assemble_matrix!(Ap,assem,matdata)
      numerical_setup!(Ap_ns,Ap)
    end
  end
end

function LinearAlgebra.mul!(y::PVector,A::PatchProlongationOperator{Val{false}},x::PVector)
  cache_refine, cache_patch, cache_redist = A.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  Ph, Ap_ns, Ap, duh, dxp, rp = cache_patch
  dxh = get_free_dof_values(duh)

  copy!(fv_H,x) # Matrix layout -> FE layout
  uH = FEFunction(UH,fv_H,dv_H)
  uh = interpolate!(uH,fv_h,Uh)

  assemble_vector!(v->A.rhs(uh,v),rp,Ph)
  map(solve!,partition(dxp),Ap_ns,partition(rp))
  inject!(dxh,Ph,dxp)
  fv_h .= fv_h .- dxh
  copy!(y,fv_h)

  return y
end

function LinearAlgebra.mul!(y::PVector,A::PatchProlongationOperator{Val{true}},x::Union{PVector,Nothing})
  cache_refine, cache_patch, cache_redist = A.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  fv_h_red, dv_h_red, Uh_red, model_h_red, glue, cache_exchange = cache_redist
  Ph, Ap_ns, Ap, duh, dxp, rp = cache_patch
  dxh  = isa(duh,Nothing) ? nothing : get_free_dof_values(duh)

  # 1 - Interpolate in coarse partition
  if !isa(x,Nothing)
    copy!(fv_H,x) # Matrix layout -> FE layout
    uH = FEFunction(UH,fv_H,dv_H)
    uh = interpolate!(uH,fv_h,Uh)

    assemble_vector!(v->A.rhs(uh,v),rp,Ph)
    map(solve!,partition(dxp),Ap_ns,partition(rp))
    inject!(dxh,Ph,dxp)
    fv_h .= fv_h .- dxh
  end

  # 2 - Redistribute from coarse partition to fine partition
  GridapDistributed.redistribute_free_values!(cache_exchange,fv_h_red,Uh_red,fv_h,dv_h,Uh,model_h_red,glue;reverse=false)
  copy!(y,fv_h_red) # FE layout -> Matrix layout

  return y
end

function setup_patch_prolongation_operators(sh,lhs,rhs,qdegrees;is_nonlinear=false)
  map(view(linear_indices(sh),1:num_levels(sh)-1)) do lev
    qdegree = isa(qdegrees,Number) ? qdegrees : qdegrees[lev]
    cparts = get_level_parts(sh,lev+1)
    if i_am_in(cparts)
      model = get_model_before_redist(sh,lev)
      PD = PatchDecomposition(model)
      Ω = Triangulation(PD)
      dΩ = Measure(Ω,qdegree)
      lhs_i = is_nonlinear ? (u,du,dv) -> lhs(u,du,dv,dΩ) : (u,v) -> lhs(u,v,dΩ)
      rhs_i = (u,v) -> rhs(u,v,dΩ)
    else
      PD, lhs_i, rhs_i = nothing, nothing, nothing
    end
    PatchProlongationOperator(lev,sh,PD,lhs_i,rhs_i;is_nonlinear)
  end
end

function get_coarse_node_mask(fmodel::GridapDistributed.DistributedDiscreteModel,glue)
  gids = get_face_gids(fmodel,0)
  mask = map(local_views(fmodel),glue,partition(gids)) do fmodel, glue, gids
    mask = get_coarse_node_mask(fmodel,glue)
    mask[ghost_to_local(gids)] .= true # Mask ghost nodes as well
    return mask
  end
  return mask
end

# Coarse nodes are the ones that are shared by fine cells that do not belong to the same coarse cell. 
# Conversely, fine nodes are the ones shared by fine cells that all have the same parent coarse cell.
function get_coarse_node_mask(fmodel::DiscreteModel{Dc},glue) where Dc
  ftopo = get_grid_topology(fmodel)
  n2c_map = Gridap.Geometry.get_faces(ftopo,0,Dc)
  n2c_map_cache = array_cache(n2c_map)
  f2c_cells   = glue.n2o_faces_map[Dc+1]
  is_boundary = get_isboundary_face(ftopo,0)

  is_coarse = map(1:length(n2c_map)) do n
    nbor_cells = getindex!(n2c_map_cache,n2c_map,n)
    parent = f2c_cells[first(nbor_cells)]
    return is_boundary[n] || any(c -> f2c_cells[c] != parent, nbor_cells)
  end

  return is_coarse
end

# PatchRestrictionOperator

struct PatchRestrictionOperator{R,A,B}
  Ip :: A
  rhs :: Union{Function,Nothing}
  caches :: B

  function PatchRestrictionOperator{R}(
    Ip::PatchProlongationOperator{R},
    rhs,
    caches
  ) where R
    A = typeof(Ip)
    B = typeof(caches)
    new{R,A,B}(Ip,rhs,caches)
  end
end

function PatchRestrictionOperator(lev,sh,Ip,rhs,qdegree;solver=LUSolver())

  cache_refine = MultilevelTools._get_dual_projection_cache(lev,sh,qdegree,solver)
  cache_redist = MultilevelTools._get_redistribution_cache(lev,sh,:residual,:restriction,:dual_projection,cache_refine)
  cache_patch = Ip.caches[2]
  caches = cache_refine, cache_patch, cache_redist

  redist = has_redistribution(sh,lev)
  R = typeof(Val(redist))
  return PatchRestrictionOperator{R}(Ip,rhs,caches)
end

function MultilevelTools.update_transfer_operator!(op::PatchRestrictionOperator,x::Union{PVector,Nothing})
  nothing
end

function setup_patch_restriction_operators(sh,patch_prolongations,rhs,qdegrees;kwargs...)
  map(view(linear_indices(sh),1:num_levels(sh)-1)) do lev
    qdegree = isa(qdegrees,Number) ? qdegrees : qdegrees[lev]
    cparts = get_level_parts(sh,lev+1)
    if i_am_in(cparts)
      model = get_model_before_redist(sh,lev)
      Ω  = Triangulation(model)
      dΩ = Measure(Ω,qdegree)
      rhs_i = (u,v) -> rhs(u,v,dΩ)
    else
      rhs_i = nothing
    end
    Ip = patch_prolongations[lev]
    PatchRestrictionOperator(lev,sh,Ip,rhs_i,qdegree;kwargs...)
  end
end

function LinearAlgebra.mul!(y::PVector,A::PatchRestrictionOperator{Val{false}},x::PVector)
  cache_refine, cache_patch, _ = A.caches
  model_h, Uh, VH, Mh_ns, rh, uh, assem, dΩhH = cache_refine
  Ph, Ap_ns, Ap, duh, dxp, rp = cache_patch
  fv_h = get_free_dof_values(uh)
  dxh = get_free_dof_values(duh)

  copy!(fv_h,x)
  fill!(rp,0.0)
  prolongate!(rp,Ph,fv_h)
  map(solve!,partition(dxp),Ap_ns,partition(rp))
  inject!(dxh,Ph,dxp)
  consistent!(dxh) |> fetch

  assemble_vector!(v->A.rhs(duh,v),rh,Uh)
  fv_h .= fv_h .- rh
  consistent!(fv_h) |> fetch

  solve!(rh,Mh_ns,fv_h)
  copy!(fv_h,rh)
  consistent!(fv_h) |> fetch
  v = get_fe_basis(VH)
  assemble_vector!(y,assem,collect_cell_vector(VH,∫(v⋅uh)*dΩhH))

  return y
end

function LinearAlgebra.mul!(y::Union{PVector,Nothing},A::PatchRestrictionOperator{Val{true}},x::PVector)
  cache_refine, cache_patch, cache_redist = A.caches
  model_h, Uh, VH, Mh_ns, rh, uh, assem, dΩhH = cache_refine
  Ph, Ap_ns, Ap, duh, dxp, rp = cache_patch
  fv_h_red, dv_h_red, Uh_red, model_h_red, glue, cache_exchange = cache_redist
  fv_h = isa(uh,Nothing) ? nothing : get_free_dof_values(uh)
  dxh  = isa(duh,Nothing) ? nothing : get_free_dof_values(duh)

  copy!(fv_h_red,x)
  consistent!(fv_h_red) |> fetch
  GridapDistributed.redistribute_free_values!(cache_exchange,fv_h,Uh,fv_h_red,dv_h_red,Uh_red,model_h,glue;reverse=true)

  if !isa(y,Nothing)
    fill!(rp,0.0)
    prolongate!(rp,Ph,fv_h;is_consistent=true)
    map(solve!,partition(dxp),Ap_ns,partition(rp))
    inject!(dxh,Ph,dxp)
    consistent!(dxh) |> fetch
  
    assemble_vector!(v->A.rhs(duh,v),rh,Uh)
    fv_h .= fv_h .- rh
    consistent!(fv_h) |> fetch

    solve!(rh,Mh_ns,fv_h)
    copy!(fv_h,rh)
    consistent!(fv_h) |> fetch
    v = get_fe_basis(VH)
    assemble_vector!(y,assem,collect_cell_vector(VH,∫(v⋅uh)*dΩhH))
  end

  return y
end
