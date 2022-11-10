
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

# get_parts

function get_parts(x::GridapDistributed.DistributedDiscreteModel)
  return PartitionedArrays.get_part_ids(x.models)
end

function get_parts(x::GridapDistributed.DistributedTriangulation)
  return PartitionedArrays.get_part_ids(x.trians)
end

function get_parts(x::GridapP4est.OctreeDistributedDiscreteModel)
  return x.parts
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

function get_parts(x::VoidDistributedDiscreteModel)
  return x.parts
end

struct VoidDistributedTriangulation{Dc,Dp,A} <: GridapType
  parts::A
  function VoidDistributedTriangulation(Dc::Int,Dp::Int,parts)
    A = typeof(parts)
    return new{Dc,Dp,A}(parts)
  end
end

function get_parts(x::VoidDistributedTriangulation)
  return x.parts
end

function VoidDistributedTriangulation(trian::GridapDistributed.DistributedTriangulation{Dc,Dp}) where {Dc,Dp}
  return VoidDistributedTriangulation(Dc,Dp,get_parts(trian))
end

function Triangulation(model::VoidDistributedDiscreteModel{Dc,Dp}) where {Dc,Dp}
  return VoidDistributedTriangulation(Dc,Dp,get_parts(model))
end


# Void Gridap structures

function void(::Type{<:UnstructuredDiscreteModel{Dc,Dp}}) where {Dc,Dp}
  # This should work but does not.....
  """
  node_coordinates = Vector{Point{Dp,Dp}}(undef,0)
  cell_node_ids    = Table(Vector{Int32}(undef,0),Vector{Int32}(undef,0))
  reffes           = Vector{LagrangianRefFE{Dc}}(undef,0)
  cell_types       = Vector{Int8}(undef,0)
  grid = UnstructuredGrid(node_coordinates,cell_node_ids,reffes,cell_types)
  """
  grid = UnstructuredGrid(Gridap.ReferenceFEs.LagrangianRefFE(Float64,QUAD,1))
  return UnstructuredDiscreteModel(grid)
end

function void(::Type{<:AdaptivityGlue})
  f2c_faces_map = [Int32[1]]
  fcell_to_child_id = Int32[1]
  f2c_reference_cell_map = Int32[1]
  return AdaptivityGlue(f2c_faces_map,fcell_to_child_id,f2c_reference_cell_map)
end
