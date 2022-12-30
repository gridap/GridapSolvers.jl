
# DistributedRefinedDiscreteModels

const DistributedAdaptedDiscreteModel{Dc,Dp} = GridapDistributed.DistributedDiscreteModel{Dc,Dp,<:AbstractPData{<:AdaptedDiscreteModel{Dc,Dp}}}

function DistributedAdaptedDiscreteModel(model::GridapDistributed.AbstractDistributedDiscreteModel,
                                         parent_models::AbstractPData{<:DiscreteModel},
                                         glue::AbstractPData{<:AdaptivityGlue})
  models = map_parts(local_views(model),parent_models,glue) do model, parent, glue
    AdaptedDiscreteModel(model,parent,glue)
  end
  return GridapDistributed.DistributedDiscreteModel(models,get_cell_gids(model))
end

function DistributedAdaptedDiscreteModel(model::GridapDistributed.AbstractDistributedDiscreteModel,
                                         parent::GridapDistributed.AbstractDistributedDiscreteModel,
                                         glue::AbstractPData{<:Union{AdaptivityGlue,Nothing}})
  mparts = get_parts(model)
  pparts = get_parts(parent)

  !i_am_in(mparts)    && (return VoidDistributedDiscreteModel(model))
  (mparts === pparts) && (return DistributedAdaptedDiscreteModel(model,local_views(parent),glue))

  parent_models, glues = map_parts(local_views(model)) do m
    if i_am_in(pparts)
      parent_models = local_views(parent)
      parent_models.part, glue.part
    else
      void(typeof(m)), void(AdaptivityGlue)
    end
  end
  return DistributedAdaptedDiscreteModel(model,parent_models,glues)
end

function Gridap.Adaptivity.get_adaptivity_glue(model::DistributedAdaptedDiscreteModel)
  return map_parts(Gridap.Adaptivity.get_adaptivity_glue,local_views(model))
end

# DistributedRefinedTriangulations

const DistributedRefinedTriangulation{Dc,Dp} = GridapDistributed.DistributedTriangulation{Dc,Dp,<:AbstractPData{<:AdaptedTriangulation{Dc,Dp}}}

# ChangeDomain

function Gridap.Adaptivity.change_domain_o2n(c_cell_field,
              ftrian::GridapDistributed.DistributedTriangulation{Dc,Dp},
              glue::AbstractPData{Gridap.Adaptivity.AdaptivityGlue}) where {Dc,Dp}

  i_am_in_coarse = !isa(c_cell_field, Nothing)

  fields = map_parts(local_views(ftrian)) do Ω
    if (i_am_in_coarse)
      c_cell_field.fields.part
    else
      Gridap.Helpers.@check num_cells(Ω) == 0
      Gridap.CellData.GenericCellField(Fill(Gridap.Fields.ConstantField(0.0),num_cells(Ω)),Ω,ReferenceDomain())
    end
  end
  c_cell_field_fine = GridapDistributed.DistributedCellField(fields)

  dfield = map_parts(Gridap.Adaptivity.change_domain_o2n,local_views(c_cell_field_fine),local_views(ftrian),glue)
  return GridapDistributed.DistributedCellField(dfield)
end

# Restriction of dofs

function restrict_dofs!(fv_c::PVector,
                        fv_f::PVector,
                        dv_f::AbstractPData,
                        U_f ::GridapDistributed.DistributedSingleFieldFESpace,
                        U_c ::GridapDistributed.DistributedSingleFieldFESpace,
                        glue::AbstractPData{<:AdaptivityGlue})

  map_parts(restrict_dofs!,local_views(fv_c),local_views(fv_f),dv_f,local_views(U_f),local_views(U_c),glue)
  exchange!(fv_c)

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
  f2c_cell_values  = Gridap.Adaptivity.f2c_reindex(fine_cell_values,glue)
  child_ids        = Gridap.Adaptivity.f2c_reindex(glue.n2o_cell_to_child_id,glue)

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
