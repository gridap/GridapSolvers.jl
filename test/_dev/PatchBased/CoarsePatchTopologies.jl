using Gridap
using Gridap.Geometry, Gridap.Arrays

using GridapSolvers
using GridapSolvers: PatchBasedSmoothers  
using PartitionedArrays, GridapDistributed

parts = (2,2,1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(parts),)))
end

cmodel = CartesianDiscreteModel((0,1,0,1),(5,5);isperiodic=(true,true))
cmodel = CartesianDiscreteModel(ranks,parts,(0,1,0,1,0,1),(4,4,3);isperiodic=(false,false,true))
model = Gridap.Adaptivity.refine(cmodel)

glue = Gridap.Adaptivity.get_adaptivity_glue(model)
ptopo = PatchBasedSmoothers.CoarsePatchTopology(model);

