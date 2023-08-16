module RichardsonSmoothersTests

using Test
using MPI
using Gridap
using GridapDistributed
using PartitionedArrays
using IterativeSolvers

using GridapSolvers
using GridapSolvers.LinearSolvers

using GridapPETSc

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

function main(parts,partition)
  GridapPETSc.with() do
    domain = (0,1,0,1)
    model  = CartesianDiscreteModel(parts,domain,partition)

    sol(x) = x[1] + x[2]
    f(x)   = -Δ(sol)(x)

    order  = 1
    qorder = order*2 + 1
    reffe  = ReferenceFE(lagrangian,Float64,order)
    Vh     = TestFESpace(model,reffe,conformity=:H1,dirichlet_tags="boundary")
    Uh     = TrialFESpace(Vh,sol)
    u      = interpolate(sol,Uh)

    Ω      = Triangulation(model)
    dΩ     = Measure(Ω,qorder)
    a(u,v) = ∫(∇(v)⋅∇(u))*dΩ
    l(v)   = ∫(v⋅f)*dΩ

    op = AffineFEOperator(a,l,Uh,Vh)
    A, b = get_matrix(op), get_vector(op)

    P  = PETScLinearSolver(set_ksp_options)
    ss = symbolic_setup(P,A)
    ns = numerical_setup(ss,A)

    x = pfill(0.0,partition(axes(A,2)))
    solve!(x,ns,b)

    u  = interpolate(sol,Uh)
    uh = FEFunction(Uh,x)
    eh = uh - u
    E  = sum(∫(eh*eh)*dΩ)
    if i_am_main(parts)
      println("L2 Error: ", E)
    end
    
    @test E < 1.e-8
  end
end

partition = (32,32)
num_ranks = (2,2)
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(num_ranks),)))
end
main(parts,partition)
MPI.Finalize()

end