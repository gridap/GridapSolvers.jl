module BlockDiagonalSmoothersPETScTests

using Gridap
using Gridap.MultiField
using BlockArrays
using LinearAlgebra
using FillArrays
using IterativeSolvers

using GridapPETSc

using GridapSolvers

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

u(x) = VectorValue(x[1],x[2])
f(x) = VectorValue(2.0*x[2]*(1.0-x[1]*x[1]),2.0*x[1]*(1-x[2]*x[2]))

p(x) = x[1] + x[2]
g(x) = -Δ(p)(x)

GridapPETSc.with() do
  D = 2
  n = 10
  domain = Tuple(repeat([0,1],D))
  partition = (n,n)
  model = CartesianDiscreteModel(domain,partition)

  order  = 2
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["boundary"])

  reffeₚ = ReferenceFE(lagrangian,Float64,order)
  Q = TestFESpace(model,reffeₚ,conformity=:H1,dirichlet_tags=["boundary"])

  U = TrialFESpace(V,u)
  P = TrialFESpace(Q,p)

  Y = MultiFieldFESpace([V, Q])
  X = MultiFieldFESpace([U, P])

  degree = 2*(order + 1)
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,degree)

  a((u,p),(v,q)) = ∫( v⊙u + ∇(v)⊙∇(u) + q⋅p + ∇(q)⊙∇(p))dΩ
  l((v,q)) = ∫( v⋅f + q⋅g)dΩ

  op = AffineFEOperator(a,l,X,Y)
  A,b = get_matrix(op), get_vector(op)
  xh_star = solve(op)
  x_star = get_free_dof_values(xh_star)

  dof_ids = get_free_dof_ids(X)
  ranges  = map(i->dof_ids[Block(i)],1:blocklength(dof_ids))
  solvers = Fill(PETScLinearSolver(set_ksp_options),2)

  BDS   = BlockDiagonalSmoother(A,ranges,solvers;lazy_mode=true)
  BDSss = symbolic_setup(BDS,A)
  BDSns = numerical_setup(BDSss,A)

  x = get_free_dof_values(zero(X))
  x = cg!(x,A,b;verbose=true,Pl=BDSns,reltol=1.0e-12)

  println("Error: ",norm(x-x_star))
end

end