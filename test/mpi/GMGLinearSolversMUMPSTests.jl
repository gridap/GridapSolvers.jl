module GMGLinearSolversMUMPSTests
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

using GridapPETSc

u(x) = x[1] + x[2]
f(x) = -Δ(u)(x)

function set_ksp_options(ksp)
  pc       = Ref{GridapPETSc.PETSC.PC}()
  mumpsmat = Ref{GridapPETSc.PETSC.Mat}()
  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPPREONLY)
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCLU)
  @check_error_code GridapPETSc.PETSC.PCFactorSetMatSolverType(pc[],GridapPETSc.PETSC.MATSOLVERMUMPS)
  @check_error_code GridapPETSc.PETSC.PCFactorSetUpMatSolverType(pc[])
  @check_error_code GridapPETSc.PETSC.PCFactorGetMatrix(pc[],mumpsmat)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  4, 1)
  # percentage increase in the estimated working space
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[],  14, 1000)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 28, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetIcntl(mumpsmat[], 29, 2)
  @check_error_code GridapPETSc.PETSC.MatMumpsSetCntl(mumpsmat[], 3, 1.0e-6)
end

function main(parts, coarse_grid_partition, num_parts_x_level, num_refs_coarse, order)
  GridapPETSc.with() do
    domain       = (0,1,0,1)
    num_levels   = length(num_parts_x_level)
    cparts       = generate_subparts(parts,num_parts_x_level[num_levels])
    cmodel       = CartesianDiscreteModel(domain,coarse_grid_partition)
    coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
    mh = ModelHierarchy(parts,coarse_model,num_parts_x_level)

    qdegree   = 2*(order+1)
    reffe     = ReferenceFE(lagrangian,Float64,order)
    tests     = TestFESpace(mh,reffe,conformity=:H1,dirichlet_tags="boundary")
    trials    = TrialFESpace(tests,u)

    biform(u,v,dΩ)  = ∫(∇(v)⋅∇(u))dΩ
    liform(v,dΩ)    = ∫(v*f)dΩ
    smatrices, A, b = compute_hierarchy_matrices(trials,biform,liform,qdegree)
    
    # Preconditioner
    smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),num_levels-1)
    restrictions, prolongations = setup_transfer_operators(trials,qdegree;mode=:residual)
    mumps_solver = PETScLinearSolver(set_ksp_options)

    gmg = GMGLinearSolver(mh,
                          smatrices,
                          prolongations,
                          restrictions,
                          pre_smoothers=smoothers,
                          post_smoothers=smoothers,
                          coarsest_solver=mumps_solver,
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
    model = get_model(mh,1)
    Uh    = get_fe_space(trials,1)
    Ω     = Triangulation(model)
    dΩ    = Measure(Ω,qdegree)
    uh    = FEFunction(Uh,x)
    e     = u-uh
    e_l2  = sum(∫(e⋅e)dΩ)
    tol   = 1.0e-9
    @test e_l2 < tol
    if GridapP4est.i_am_main(parts)
      println("L2 error = ", e_l2)
    end
  end
end

##############################################

if !MPI.Initialized()
  MPI.Init()
end

# Parameters
order = 2
coarse_grid_partition = (2,2)
num_refs_coarse = 3

num_parts_x_level = [4,2]
ranks = num_parts_x_level[1]
prun(main,mpi,ranks,coarse_grid_partition,num_parts_x_level,num_refs_coarse,order)


MPI.Finalize()
end