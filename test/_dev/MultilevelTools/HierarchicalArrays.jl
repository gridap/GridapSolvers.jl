using PartitionedArrays
using GridapDistributed
using GridapSolvers, GridapSolvers.MultilevelTools
using Gridap, Gridap.Geometry


struct HierarchyLevel{A,B,C}
  object     :: A
  object_red :: B
  red_glue   :: C
end

############################################################################################

np = (2,1)
ranks = DebugArray(LinearIndices((prod(np),)))

dmodel = CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4))
model1 = CartesianDiscreteModel((0,1,0,1),(4,4))
model2 = UnstructuredDiscreteModel(model1)

a = HierarchicalArray(fill(dmodel,2),fill(ranks,2))
b = HierarchicalArray([dmodel,nothing],fill(ranks,2))
c = HierarchicalArray([model1,model2],fill(ranks,2))



