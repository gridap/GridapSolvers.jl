module ModelHierarchiesTests

using Test
using MPI
using Gridap
using Gridap.FESpaces, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

using GridapSolvers
using GridapSolvers.MultilevelTools

function main(distribute,np,np_per_level)
  parts = distribute(LinearIndices((prod(np),)))
  
  domain = (0,1,0,1)
  nc = (4,4)
  mh = CartesianModelHierarchy(parts,np_per_level,domain,nc)

  @test isa(mh,ModelHierarchy)

  reffe = ReferenceFE(lagrangian,Float64,1)
  sh = TestFESpace(mh,reffe)
  @test isa(sh,FESpaceHierarchy)

  th = Triangulation(mh)
  @test isa(th,TriangulationHierarchy)

  sh = FESpace(th,reffe)
  @test isa(sh,FESpaceHierarchy)
end

end