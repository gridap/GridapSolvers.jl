

struct DistributedRefinedDiscreteModel{Dc,Dp,A,B,C} <: GridapDistributed.AbstractDistributedDiscreteModel{Dc,Dp}
  model  :: A
  parent :: B
  glue   :: C

  function DistributedRefinedDiscreteModel(model ::GridapDistributed.AbstractDistributedDiscreteModel{Dc,Dp},
                                           parent::GridapDistributed.AbstractDistributedDiscreteModel{Dc,Dp},
                                           glue) where {Dc,Dp}
    A = typeof(model)
    B = typeof(parent)
    C = typeof(glue)
    return new{Dc,Dp,A,B,C}(model,parent,glue)
  end
end


function Base.getproperty(x::DistributedRefinedDiscreteModel, sym::Symbol)
  if sym === :parts
    return x.model.parts
  else
    getfield(x, sym)
  end
end

function Base.propertynames(x::DistributedRefinedDiscreteModel, private::Bool=false)
  (fieldnames(typeof(x))...,:parts)
end

Gridap.Geometry.num_cells(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.num_cells(model.model)
Gridap.Geometry.num_facets(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.num_facets(model.model)
Gridap.Geometry.num_edges(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.num_edges(model.model)
Gridap.Geometry.num_vertices(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.num_vertices(model.model)
Gridap.Geometry.num_faces(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.num_faces(model.model)
Gridap.Geometry.get_grid(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.get_grid(model.model)
Gridap.Geometry.get_grid_topology(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.get_grid_topology(model.model)
Gridap.Geometry.get_face_labeling(model::DistributedRefinedDiscreteModel) = Gridap.Geometry.get_face_labeling(model.model)

GridapDistributed.local_views(model::DistributedRefinedDiscreteModel) = GridapDistributed.local_views(model.model)
GridapDistributed.get_cell_gids(model::DistributedRefinedDiscreteModel) = GridapDistributed.get_cell_gids(model.model)
GridapDistributed.get_face_gids(model::DistributedRefinedDiscreteModel,dim::Integer) = GridapDistributed.get_face_gids(model.model,dim)
GridapDistributed.generate_gids(model::DistributedRefinedDiscreteModel,spaces) = GridapDistributed.generate_gids(model.model,spaces)
