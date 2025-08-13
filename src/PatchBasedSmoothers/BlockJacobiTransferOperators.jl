
"""
    struct BlockJacobiProlongationOperator end

A `BlockJacobiProlongationOperator` is a modified prolongation operator such that given a coarse
solution `xH`, it maps it to a fine mesh solution `xh` given by

```
xh = Ih(xH) - yh
```

where `yh` is a subspace-based correction computed by solving local problems on coarse cell 
patches within the fine mesh.
"""
mutable struct BlockJacobiProlongationOperator{R,A,B}
  sh     :: A
  solver :: B
  caches

  function BlockJacobiProlongationOperator{R}(
    sh, solver::BlockJacobiSolver, caches
  ) where R
    A, B = typeof(sh), typeof(solver)
    new{R,A,B}(sh,solver,caches)
  end
end

@doc """
    BlockJacobiProlongationOperator(lev::Integer,sh::FESpaceHierarchy,solver::BlockJacobiSolver)

    function BlockJacobiProlongationOperator(lev::Integer,sh::FESpaceHierarchy,args...;kwargs...)
      solver = BlockJacobiSolver(get_fe_space_before_redist(sh,lev),args...;kwargs...)
      BlockJacobiProlongationOperator(lev,sh,solver)
    end

Returns an instance of `BlockJacobiProlongationOperator` for a given level `lev` and a given 
FESpaceHierarchy `sh`.

"""
function BlockJacobiProlongationOperator(lev,sh,args...;kwargs...)
  Vh = MultilevelTools.get_fe_space_before_redist(sh,lev)
  solver = BlockJacobiSolver(Vh,args...;kwargs...)
  BlockJacobiProlongationOperator(lev,sh,solver)
end

function BlockJacobiProlongationOperator(
  lev,sh,solver
)

  cache_refine = MultilevelTools._get_interpolation_cache(lev,sh,0,:residual)
  cache_redist = MultilevelTools._get_redistribution_cache(lev,sh,:residual,:prolongation,:interpolation,cache_refine)
  cache_patch = _get_patch_cache(lev,sh,assem,lhs,rhs,is_nonlinear,collect_factorizations,cache_refine)
  caches = cache_refine, cache_patch, cache_redist

  redist = has_redistribution(sh,lev)
  R = typeof(Val(redist))
  return BlockJacobiProlongationOperator{R}(sh,assem,lhs,rhs,is_nonlinear,collect_factorizations,caches)
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
function MultilevelTools.update_transfer_operator!(op::BlockJacobiProlongationOperator,A::Union{AbstractMatrix,Nothing})
  cache_refine, cache_patch, cache_redist = op.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  uh, uH, xh, dx_h, _, _, _, _, patch_ids, _ = cache_patch

  if !op.is_nonlinear
    return # Nothing to update
  end

  if !isa(cache_redist,Nothing)
    fv_h_red, dv_h_red, Uh_red, model_h_red, glue, cache_exchange = cache_redist
    copy!(fv_h_red,x)
    consistent!(fv_h_red) |> fetch
    GridapDistributed.redistribute_free_values(fv_h,Uh,fv_h_red,dv_h_red,Uh_red,model_h,glue;reverse=true)
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

function LinearAlgebra.mul!(y::AbstractVector,A::BlockJacobiProlongationOperator{Val{false}},x::AbstractVector)
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

function LinearAlgebra.mul!(y::PVector,A::BlockJacobiProlongationOperator{Val{true}},x::Union{PVector,Nothing})
  cache_refine, cache_patch, cache_redist = A.caches
  model_h, Uh, fv_h, dv_h, UH, fv_H, dv_H = cache_refine
  fv_h_red, dv_h_red, Uh_red, model_h_red, glue, cache_exchange = cache_redist
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
  GridapDistributed.redistribute_free_values!(cache_exchange,fv_h_red,Uh_red,fv_h,dv_h,Uh,model_h_red,glue;reverse=false)
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
    BlockJacobiProlongationOperator(
      lev,sh,ptopo,lhs_i,rhs_i;is_nonlinear,collect_factorizations
    )
  end
end
