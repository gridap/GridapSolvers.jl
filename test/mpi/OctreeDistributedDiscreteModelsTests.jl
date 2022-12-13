module OctreeDistributedDiscreteModelsTests
using MPI
using Test
using Gridap
using Gridap.ReferenceFEs
using Gridap.FESpaces
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.MultilevelTools

function run(parts,subdomains,num_parts_x_level)
  if length(subdomains) == 2
    domain=(0,1,0,1)
  else
    @assert length(subdomains) == 3
    domain=(0,1,0,1,0,1)
  end

  # Generate model
  level_parts  = generate_level_parts(parts,num_parts_x_level)
  coarse_model = CartesianDiscreteModel(domain,subdomains)
  model        = GridapP4est.OctreeDistributedDiscreteModel(level_parts[2],coarse_model,1)

  # Refining and distributing
  fmodel1 , rglue1  = refine(model,level_parts[1])
  dfmodel1, dglue1  = redistribute(fmodel1)

  fmodel2 , rglue2  = refine(model)
  dfmodel2, dglue2  = redistribute(fmodel2,level_parts[1])

  # FESpaces tests
  sol(x) = x[1] + x[2]
  reffe  = ReferenceFE(lagrangian,Float64,1)
  Vh     = TestFESpace(dfmodel2, reffe; conformity=:H1)
  Uh     = TrialFESpace(sol,Vh)
  Ω      = Triangulation(dfmodel2)
  dΩ     = Measure(Ω,3)
  
  a(u,v) = ∫(v⋅u)*dΩ
  assemble_matrix(a,Uh,Vh)
end

prun(run,mpi,4,(2,2),[4,2])
MPI.Finalize()
end # module
