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

u(x) = VectorValue(x[1],x[2])
f(x) = VectorValue(2.0*x[2]*(1.0-x[1]*x[1]),2.0*x[1]*(1-x[2]*x[2]))

p(x) = x[1] + x[2]
g(x) = -Δ(p)(x)

function main(model,single_proc::Bool)
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


  # Global problem
  a((u,p),(v,q)) = ∫( v⊙u + ∇(v)⊙∇(u) + q⋅p + ∇(q)⊙∇(p))dΩ
  l((v,q)) = ∫( v⋅f + q⋅g)dΩ

  op = AffineFEOperator(a,l,X,Y)
  A,b = get_matrix(op), get_vector(op);
  xh_star = solve(op)
  x_star = get_free_dof_values(xh_star)

  dof_ids = get_free_dof_ids(X)
  ranges  = map(i->dof_ids[Block(i)],1:blocklength(dof_ids))
  solvers = Fill(BackslashSolver(),2)

  # Build using the global matrix
  if single_proc
    BDS   = BlockDiagonalSmoother(A,ranges,solvers)
    BDSss = symbolic_setup(BDS,A)
    BDSns = numerical_setup(BDSss,A)

    x = get_free_dof_values(zero(X))
    x = cg!(x,A,b;verbose=true,Pl=BDSns,reltol=1.0e-12)

    @test norm(x-x_star) < 1.e-8
  end

  # Build using local weakforms
  a1(u,v) = ∫(v⊙u + ∇(v)⊙∇(u))dΩ
  a2(p,q) = ∫(q⋅p + ∇(q)⊙∇(p))dΩ
  biforms = [a1,a2]

  BDS   = BlockDiagonalSmoother(biforms,X,Y,solvers)
  BDSss = symbolic_setup(BDS,A)
  BDSns = numerical_setup(BDSss,A)

  x = GridapSolvers.LinearSolvers.allocate_col_vector(A)
  x = cg!(x,A,b;verbose=true,Pl=BDSns,reltol=1.0e-12)

  @test norm(x-x_star) < 1.e-8

  # Build using BlockMatrixAssemblers
  mfs = BlockMultiFieldStyle()
  Yb = MultiFieldFESpace([V,Q];style=mfs)
  Xb = MultiFieldFESpace([U,P];style=mfs)

  if single_proc
    assem = SparseMatrixAssembler(Xb,Yb)
  else
    assem = SparseMatrixAssembler(Xb,Yb,FullyAssembledRows())
  end
  op_blocks = AffineFEOperator(a,l,Xb,Yb,assem)
  Ab,bb = get_matrix(op_blocks), get_vector(op_blocks);

  BDS   = BlockDiagonalSmoother(Ab,solvers)
  BDSss = symbolic_setup(BDS,A)
  BDSns = numerical_setup(BDSss,A)

  xb = GridapSolvers.LinearSolvers.allocate_col_vector(Ab)
  xb = cg!(xb,Ab,bb;verbose=true,Pl=BDSns,reltol=1.0e-12)

  @test norm(x-x_star) < 1.e-8
end

num_ranks = (2,2)
parts = with_debug() do distribute
  distribute(LinearIndices((prod(num_ranks),)))
end

D = 2
n = 10
domain = Tuple(repeat([0,1],D))
mesh_partition = (n,n)

# Serial
model = CartesianDiscreteModel(domain,mesh_partition)
main(model,true)

# Distributed, sequential
model = CartesianDiscreteModel(parts,num_ranks,domain,mesh_partition)
main(model,false)

end