module BlockDiagonalSmoothersTests

using Test
using Gridap
using Gridap.MultiField
using BlockArrays
using LinearAlgebra
using FillArrays
using IterativeSolvers
using PartitionedArrays

using GridapDistributed
using GridapSolvers
using GridapPETSc

using GridapDistributed: BlockPVector, BlockPMatrix

u(x) = VectorValue(x[1],x[2])
f(x) = VectorValue(2.0*x[2]*(1.0-x[1]*x[1]),2.0*x[1]*(1-x[2]*x[2]))

p(x) = x[1] + x[2]
g(x) = -Δ(p)(x)

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

function _is_same_vector(x1,x2,X1,X2)
  res = true
  for i in 1:length(X1)
    x1i = restrict_to_field(X1,x1,i)
    x2i = restrict_to_field(X2,x2,i)
    res = res & (norm(x1i-x2i) < 1.e-5)
  end
  return res
end

function is_same_vector(x1::BlockVector,x2,X1,X2)
  _is_same_vector(x1,x2,X1,X2)
end

function is_same_vector(x1::BlockPVector,x2,X1,X2)
  _x1 = GridapDistributed.change_ghost(x1,X1.gids;make_consistent=true)
  _x2 = GridapDistributed.change_ghost(x2,X2.gids;make_consistent=true)
  _is_same_vector(_x1,_x2,X1,X2)
end

function get_mesh(parts,np)
  Dc = length(np)
  if Dc == 2
    domain = (0,1,0,1)
    nc = (8,8)
  else
    @assert Dc == 3
    domain = (0,1,0,1,0,1)
    nc = (8,8,8)
  end
  if prod(np) == 1
    model = CartesianDiscreteModel(domain,nc)
  else
    model = CartesianDiscreteModel(parts,np,domain,nc)
  end
  return model
end

function main_driver(D,model,solvers)
  order  = 2
  reffeᵤ = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  V = TestFESpace(model,reffeᵤ,conformity=:H1,dirichlet_tags=["boundary"])

  reffeₚ = ReferenceFE(lagrangian,Float64,order)
  Q = TestFESpace(model,reffeₚ,conformity=:H1,dirichlet_tags=["boundary"])

  U = TrialFESpace(V,u)
  P = TrialFESpace(Q,p)

  Y = MultiFieldFESpace([V,Q])
  X = MultiFieldFESpace([U,P])

  mfs = BlockMultiFieldStyle()
  Yb = MultiFieldFESpace([V,Q];style=mfs)
  Xb = MultiFieldFESpace([U,P];style=mfs)

  degree = 2*(order + 1)
  Ω  = Triangulation(model)
  dΩ = Measure(Ω,degree)

  # Global problem
  a((u,p),(v,q)) = ∫( v⊙u + ∇(v)⊙∇(u) + q⋅p + ∇(q)⊙∇(p))dΩ
  l((v,q)) = ∫( v⋅f + q⋅g)dΩ

  op = AffineFEOperator(a,l,X,Y)
  x_star = get_free_dof_values(solve(op))

  opb = AffineFEOperator(a,l,Xb,Yb)
  A,b = get_matrix(opb), get_vector(opb);

  # Build using local weakforms
  a1(u,v) = ∫(v⊙u + ∇(v)⊙∇(u))dΩ
  a2(p,q) = ∫(q⋅p + ∇(q)⊙∇(p))dΩ
  biforms = [a1,a2]

  BDS   = BlockDiagonalSmoother(biforms,Xb,Yb,solvers)
  BDSss = symbolic_setup(BDS,A)
  BDSns = numerical_setup(BDSss,A)

  x = GridapSolvers.allocate_col_vector(A)
  x = cg!(x,A,b;verbose=true,Pl=BDSns,reltol=1.0e-12)
  @test is_same_vector(x,x_star,Xb,X)

  # Build using BlockMatrixAssemblers
  BDS   = BlockDiagonalSmoother(A,solvers)
  BDSss = symbolic_setup(BDS,A)
  BDSns = numerical_setup(BDSss,A)

  x = GridapSolvers.allocate_col_vector(A)
  x = cg!(x,A,b;verbose=true,Pl=BDSns,reltol=1.0e-12)
  @test is_same_vector(x,x_star,Xb,X)
end

function main(distribute,np,use_petsc::Bool)
  parts = distribute(LinearIndices((prod(np),)))
  Dc = length(np)
  model = get_mesh(parts,np)
  if use_petsc
    GridapPETSc.with() do
      solvers = Fill(PETScLinearSolver(set_ksp_options),2)
      main_driver(Dc,model,solvers)
    end
  else
    solvers = Fill(LUSolver(),2)
    main_driver(Dc,model,solvers)
  end
end

end