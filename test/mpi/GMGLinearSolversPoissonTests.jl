module GMGLinearSolverPoissonTests
using MPI
using Test
using LinearAlgebra
using IterativeSolvers
using FillArrays

using Gridap
using Gridap.ReferenceFEs
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers


u(x) = x[1] + x[2]
f(x) = -Δ(u)(x)

function main(parts, coarse_grid_partition, num_parts_x_level, num_refs_coarse, order)
  domain       = (0,1,0,1)
  num_levels   = length(num_parts_x_level)
  level_parts  = generate_level_parts(parts,num_parts_x_level)
  cmodel       = CartesianDiscreteModel(domain,coarse_grid_partition)
  coarse_model = OctreeDistributedDiscreteModel(level_parts[num_levels],cmodel,num_refs_coarse)
  mh = ModelHierarchy(coarse_model,level_parts)

  qdegree   = 2*(order+1)
  reffe     = ReferenceFE(lagrangian,Float64,order)
  tests     = TestFESpace(mh,reffe,conformity=:H1,dirichlet_tags="boundary")
  trials    = TrialFESpace(tests,u)

  biform(u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ
  liform(v,dΩ)   = ∫(v*f)dΩ

  # Finest level problem
  model = get_model(mh,1)
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,qdegree)
  Vh = get_fe_space(tests,1)
  Uh = get_fe_space(trials,1)

  a(u,v) = biform(u,v,dΩ)
  l(v)   = liform(v,dΩ)
  op     = AffineFEOperator(a,l,Uh,Vh)
  A, b   = get_matrix(op), get_vector(op)
  
  # Preconditioner
  smatrices = compute_hierarchy_matrices(trials,biform,qdegree)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),num_levels-1)
  restrictions, prolongations = setup_transfer_operators(trials,qdegree;mode=:residual)

  gmg = GMGLinearSolver(mh,
                        smatrices,
                        prolongations,
                        restrictions,
                        pre_smoothers=smoothers,
                        post_smoothers=smoothers,
                        maxiter=1,
                        rtol=1.0e-10,
                        verbose=false,
                        mode=:preconditioner)
  ss = symbolic_setup(gmg,A)
  ns = numerical_setup(ss,A)

  # Solve 
  x = PVector(0.0,A.cols)
  x, history = IterativeSolvers.cg!(x,A,b;
                        verbose=GridapP4est.i_am_main(parts),
                        reltol=1.0e-12,
                        Pl=ns,
                        log=true)

  # Error norms and print solution
  uh   = FEFunction(Uh,x)
  e    = u-uh
  e_l2 = sum(∫(e*e)dΩ)
  tol  = 1.0e-9
  @test e_l2 < tol
  if GridapP4est.i_am_main(parts)
    println("L2 error = ", e_l2)
  end
end

##############################################

if !MPI.Initialized()
  MPI.Init()
end

# Parameters
order = 2
coarse_grid_partition = (2,2)
num_refs_coarse = 2

num_parts_x_level = [4,2,1]
ranks = num_parts_x_level[1]
prun(main,mpi,ranks,coarse_grid_partition,num_parts_x_level,num_refs_coarse,order)


MPI.Finalize()
end