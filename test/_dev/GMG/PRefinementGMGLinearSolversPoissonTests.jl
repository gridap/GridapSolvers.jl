module PRefinementGMGLinearSolverPoissonTests
using MPI
using Test
using LinearAlgebra
using IterativeSolvers
using FillArrays

using Gridap
using Gridap.Helpers
using Gridap.ReferenceFEs
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools

u(x) = x[1] + x[2]
f(x) = -Δ(u)(x)

function main(parts, coarse_grid_partition, num_parts_x_level, num_refs_coarse, max_order)
  GridapP4est.with(parts) do
    domain       = (0,1,0,1)
    num_levels   = length(num_parts_x_level)
    cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
    cmodel       = CartesianDiscreteModel(domain,coarse_grid_partition)
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
    mh = ModelHierarchy(parts,coarse_model,num_parts_x_level;mesh_refinement=false)

    orders    = collect(max_order:-1:1)
    qdegrees  = map(o->2*(o+1),orders)
    reffes    = map(o->ReferenceFE(lagrangian,Float64,o),orders)
    tests     = TestFESpace(mh,reffes;conformity=:H1,dirichlet_tags="boundary")
    trials    = TrialFESpace(tests,u)

    biform(u,v,dΩ)  = ∫(∇(v)⋅∇(u))dΩ
    liform(v,dΩ)    = ∫(v*f)dΩ
    smatrices, A, b = compute_hierarchy_matrices(trials,biform,liform,qdegrees)
    
    # Preconditioner
    smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),num_levels-1)
    restrictions, prolongations = setup_transfer_operators(trials,qdegrees;
                                                           mode=:residual,
                                                           restriction_method=:interpolation)

    gmg = GMGLinearSolver(
      smatrices,
      prolongations,
      restrictions,
      pre_smoothers=smoothers,
      post_smoothers=smoothers,
      maxiter=1,
      rtol=1.0e-10,
      verbose=false,
      mode=:preconditioner
    )
    ss = symbolic_setup(gmg,A)
    ns = numerical_setup(ss,A)

    # Solve 
    x = pfill(0.0,partition(axes(A,2)))
    x, history = IterativeSolvers.cg!(x,A,b;
                          verbose=i_am_main(parts),
                          reltol=1.0e-12,
                          Pl=ns,
                          log=true)

    # Error norms and print solution
    model = get_model(mh,1)
    Uh    = get_fe_space(trials,1)
    Ω     = Triangulation(model)
    dΩ    = Measure(Ω,qdegrees[1])
    uh    = FEFunction(Uh,x)
    e     = u-uh
    e_l2  = sum(∫(e⋅e)dΩ)
    tol   = 1.0e-9
    @test e_l2 < tol
    if i_am_main(parts)
      println("L2 error = ", e_l2)
    end
  end
end

##############################################

if !MPI.Initialized()
  MPI.Init()
end

# Parameters
max_order = 3
coarse_grid_partition = (2,2)
num_refs_coarse = 2

num_parts_x_level = [4,4,1]
num_ranks = num_parts_x_level[1]
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(num_ranks),)))
end
main(parts,coarse_grid_partition,num_parts_x_level,num_refs_coarse,max_order)


MPI.Finalize()
end
