
# DistributedAdaptedTriangulations

const DistributedAdaptedTriangulation{Dc,Dp} = GridapDistributed.DistributedTriangulation{Dc,Dp,<:AbstractArray{<:AdaptedTriangulation{Dc,Dp}}}

# Restriction of dofs

function restrict_dofs!(fv_c::PVector,
                        fv_f::PVector,
                        dv_f::AbstractArray,
                        U_f ::GridapDistributed.DistributedSingleFieldFESpace,
                        U_c ::GridapDistributed.DistributedSingleFieldFESpace,
                        glue::AbstractArray{<:AdaptivityGlue})

  map(restrict_dofs!,local_views(fv_c),local_views(fv_f),dv_f,local_views(U_f),local_views(U_c),glue)
  consistent!(fv_c) |> fetch

  return fv_c
end

function restrict_dofs!(fv_c::AbstractVector,
                        fv_f::AbstractVector,
                        dv_f::AbstractVector,
                        U_f ::FESpace,
                        U_c ::FESpace,
                        glue::AdaptivityGlue)

  fine_cell_ids    = get_cell_dof_ids(U_f)
  fine_cell_values = Gridap.Arrays.Table(lazy_map(Gridap.Arrays.PosNegReindex(fv_f,dv_f),fine_cell_ids.data),fine_cell_ids.ptrs)
  coarse_rrules    = Gridap.Adaptivity.get_old_cell_refinement_rules(glue)
  f2c_cell_values  = Gridap.Adaptivity.n2o_reindex(fine_cell_values,glue)
  child_ids        = Gridap.Adaptivity.n2o_reindex(glue.n2o_cell_to_child_id,glue)

  f2c_maps = lazy_map(FineToCoarseDofMap,coarse_rrules)
  caches   = lazy_map(Gridap.Arrays.return_cache,f2c_maps,f2c_cell_values,child_ids)
  coarse_cell_values = lazy_map(Gridap.Arrays.evaluate!,caches,f2c_maps,f2c_cell_values,child_ids)
  fv_c = gather_free_values!(fv_c,U_c,coarse_cell_values)
  
  return fv_c
end

struct FineToCoarseDofMap{A}
  rr::A
end

function Gridap.Arrays.return_cache(m::FineToCoarseDofMap,fine_cell_vals,child_ids)
  return fill(0.0,Gridap.Adaptivity.num_subcells(m.rr))
end

function Gridap.Arrays.evaluate!(cache,m::FineToCoarseDofMap,fine_cell_vals,child_ids)
  fill!(cache,0.0)
  for (k,i) in enumerate(child_ids)
    cache[i] = fine_cell_vals[k][i]
  end
  return cache
end
