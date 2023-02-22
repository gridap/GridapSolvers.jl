
# DistributedCompositeMeasure

function Gridap.CellData.Measure(tt::GridapDistributed.DistributedTriangulation{Dc,Dp},
                                 it::GridapDistributed.DistributedTriangulation{Dc,Dp},
                                 args...) where {Dc,Dp}
  itrians = change_parts(local_views(it),get_parts(tt);default=void(BodyFittedTriangulation{Dc,Dp}))

  measures = map_parts(local_views(tt),itrians) do ttrian, itrian
    Measure(ttrian,itrian,args...)
  end
  return GridapDistributed.DistributedMeasure(measures)
end

# change_parts

function change_parts(x::Union{AbstractPData,Nothing}, new_parts; default=nothing)
  x_new = map_parts(new_parts) do _p
    if isa(x,AbstractPData)
      x.part
    else
      default
    end
  end
  return x_new
end

function change_parts(::Type{<:GridapDistributed.DistributedCellField},x,new_parts)
  if isa(x,GridapDistributed.DistributedCellField)
    fields = change_parts(local_views(x),new_parts)
  else
    fields = change_parts(nothing,new_parts;default=void(CellField))
  end
  return GridapDistributed.DistributedCellField(fields)
end

# DistributedFESpaces

function get_test_space(U::GridapDistributed.DistributedSingleFieldFESpace)
  spaces = map_parts(local_views(U)) do U
    U.space
  end
  return GridapDistributed.DistributedSingleFieldFESpace(spaces,U.gids,U.vector_type)
end

# Void GridapDistributed structures

struct VoidDistributedDiscreteModel{Dc,Dp,A} <: GridapDistributed.AbstractDistributedDiscreteModel{Dc,Dp}
  parts::A
  function VoidDistributedDiscreteModel(Dc::Int,Dp::Int,parts)
    A = typeof(parts)
    return new{Dc,Dp,A}(parts)
  end
end

function VoidDistributedDiscreteModel(model::GridapDistributed.AbstractDistributedDiscreteModel{Dc,Dp}) where {Dc,Dp}
  return VoidDistributedDiscreteModel(Dc,Dp,get_parts(model))
end

function GridapDistributed.get_parts(x::VoidDistributedDiscreteModel)
  return x.parts
end

struct VoidDistributedTriangulation{Dc,Dp,A} <: GridapDistributed.DistributedGridapType
  parts::A
  function VoidDistributedTriangulation(Dc::Int,Dp::Int,parts)
    A = typeof(parts)
    return new{Dc,Dp,A}(parts)
  end
end

function GridapDistributed.get_parts(x::VoidDistributedTriangulation)
  return x.parts
end

function VoidDistributedTriangulation(trian::GridapDistributed.DistributedTriangulation{Dc,Dp}) where {Dc,Dp}
  return VoidDistributedTriangulation(Dc,Dp,get_parts(trian))
end

function Gridap.Geometry.Triangulation(model::VoidDistributedDiscreteModel{Dc,Dp}) where {Dc,Dp}
  return VoidDistributedTriangulation(Dc,Dp,get_parts(model))
end

struct VoidDistributedFESpace{A} <: GridapDistributed.DistributedGridapType
  parts::A
end

function GridapDistributed.get_parts(x::VoidDistributedFESpace)
  return x.parts
end

function Gridap.FESpaces.TestFESpace(model::VoidDistributedDiscreteModel,args...;kwargs...)
  return VoidDistributedFESpace(get_parts(model))
end

function Gridap.FESpaces.TrialFESpace(space::VoidDistributedFESpace,args...;kwargs...)
  return VoidDistributedFESpace(get_parts(space))
end

function FESpaces.get_triangulation(f::VoidDistributedFESpace,model::VoidDistributedDiscreteModel)
  return VoidDistributedTriangulation(model)
end

# Void Gridap structures

function void(::Type{<:CartesianDiscreteModel{Dc,Dp}}) where {Dc,Dp}
  #domain    = Tuple(fill(0.0,2*Dc))
  domain    = Tuple(repeat([0,1],Dc))
  partition = Tuple(fill(0,Dc))
  return CartesianDiscreteModel(domain,partition)
end

function void(::Type{<:UnstructuredDiscreteModel{Dc,Dp}}) where {Dc,Dp}
  cmodel = void(CartesianDiscreteModel{Dc,Dp})
  return UnstructuredDiscreteModel(cmodel)
end

function void(::Type{<:AdaptivityGlue})
  f2c_faces_map      = [Int32[],Int32[],Int32[]]
  fcell_to_child_id  = Int32[]
  rrules             = Fill(void(RefinementRule),0)
  return AdaptivityGlue(f2c_faces_map,fcell_to_child_id,rrules)
end

function void(::Type{<:RefinementRule})
  reffe = Gridap.ReferenceFEs.LagrangianRefFE(Float64,QUAD,1)
  return RefinementRule(reffe,1)
end

function void(::Type{<:BodyFittedTriangulation{Dc,Dp}}) where {Dc,Dp}
  model = void(UnstructuredDiscreteModel{Dc,Dp})
  return Gridap.Geometry.Triangulation(model)
end

function void(::Type{<:CellField})
  trian = void(BodyFittedTriangulation{2,2})
  return Gridap.CellData.CellField(0.0,trian,ReferenceDomain())
end
