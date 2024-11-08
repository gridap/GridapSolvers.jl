using Gridap
using Gridap.Geometry, Gridap.Arrays

using GridapSolvers
using GridapSolvers: PatchBasedSmoothers  

cmodel = CartesianDiscreteModel((0,1,0,1),(2,2))
model = Gridap.Adaptivity.refine(cmodel)

glue = Gridap.Adaptivity.get_adaptivity_glue(model)

PD = PatchBasedSmoothers.CoarsePatchDecomposition(model)

reffe = ReferenceFE(lagrangian,Float64,2)
Vh = FESpace(model,reffe)
Ph = PatchFESpace(Vh,PD,reffe)
