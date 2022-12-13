
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


# DistributedRefinedTriangulations

const DistributedRefinedTriangulation{Dc,Dp} = GridapDistributed.DistributedTriangulation{Dc,Dp,<:AbstractPData{<:AdaptedTriangulation{Dc,Dp}}}

# ChangeDomain

function Gridap.Adaptivity.change_domain_o2n(c_cell_field,
              ftrian::GridapDistributed.DistributedTriangulation{Dc,Dp},
              glue::AbstractPData{Gridap.Adaptivity.AdaptivityGlue}) where {Dc,Dp}

  i_am_in_coarse = (c_cell_field != nothing)

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
