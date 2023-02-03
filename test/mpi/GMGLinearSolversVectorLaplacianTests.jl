module GMGLinearSolverVectorLaplacianTests
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


u(x) = VectorValue(x[1],x[2])
f(x) = VectorValue(2.0*x[2]*(1.0-x[1]*x[1]),2.0*x[1]*(1-x[2]*x[2]))

function main(parts, coarse_grid_partition, num_parts_x_level, num_refs_coarse, order, α)
  GridapP4est.with(parts) do
    domain       = (0,1,0,1)
    num_levels   = length(num_parts_x_level)
    cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
    cmodel       = CartesianDiscreteModel(domain,coarse_grid_partition)
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
    mh = ModelHierarchy(parts,coarse_model,num_parts_x_level)

    qdegree   = 2*(order+1)
    reffe     = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
    tests     = TestFESpace(mh,reffe;conformity=:H1,dirichlet_tags="boundary")
    trials    = TrialFESpace(tests,u)

    biform(u,v,dΩ)  = ∫(v⋅u)dΩ + ∫(α*∇(v)⊙∇(u))dΩ
    liform(v,dΩ)    = ∫(v⋅f)dΩ
    smatrices, A, b = compute_hierarchy_matrices(trials,biform,liform,qdegree)
    
    # Preconditioner
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
                          verbose=i_am_main(parts),
                          reltol=1.0e-12,
                          Pl=ns,
                          log=true)

    # Error norms and print solution
    model = get_model(mh,1)
    Uh    = get_fe_space(trials,1)
    Ω     = Triangulation(model)
    dΩ    = Measure(Ω,qdegree)
    uh    = FEFunction(Uh,x)
    e     = u-uh
    e_l2  = sum(∫(e⋅e)dΩ)
    tol   = 1.0e-9
    #@test e_l2 < tol
    if i_am_main(parts)
      println("L2 error = ", e_l2)
    end

    return history.iters, num_free_dofs(Uh)
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

α = 1.0
num_parts_x_level = [4,2,1]
ranks = num_parts_x_level[1]
num_iters, num_free_dofs2 = with_backend(main,MPIBackend(),ranks,coarse_grid_partition,num_parts_x_level,num_refs_coarse,order,α)

"""

num_refinements = [1,2,3,4]
alpha_exps = [0,1,2,3]
nr = length(num_refinements)
na = length(alpha_exps)

# Do experiments
iter_matrix = zeros(Int,nr,na)
free_dofs   = Vector{Int64}(undef,nr)
for ref = 1:nr
  num_parts_x_level = [1 for i=1:num_refinements[ref]+1]
  for alpha_exp = 1:na
    α = 10.0^alpha_exps[alpha_exp]

    num_iters, num_free_dofs2 = with_backend(main,MPIBackend(),ranks,coarse_grid_partition,num_parts_x_level,order,α)
    free_dofs[ref] = num_free_dofs2
    iter_matrix[ref,alpha_exp] = num_iters
  end
end

# Display results
if i_am_main(parts)
  println("> α = ", map(exp->10.0^exp,alpha_exp))
end

for ref = 1:nr
  if i_am_main(parts)
    println("> Num Refinements: ", num_refinements[ref])
    println("  > Num free dofs         : ", free_dofs[ref])
    println("  > Num Refinements       : ", num_refinements[ref])
    println("  > Num Iters (per alpha) : ", iter_matrix[ref,:])
  end
end
"""


MPI.Finalize()
end
