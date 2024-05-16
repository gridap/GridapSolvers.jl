module BlockDiagonalSolversTests

using BlockArrays, LinearAlgebra
using Gridap, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays, GridapDistributed
using GridapSolvers

function main(distribute,parts)
  ranks = distribute(LinearIndices((prod(parts),)))

  model = CartesianDiscreteModel(ranks,parts,(0,1,0,1),(8,8))

  reffe = ReferenceFE(lagrangian,Float64,1)
  V = FESpace(model,reffe)

  mfs = BlockMultiFieldStyle()
  Y = MultiFieldFESpace([V,V];style=mfs)

  Ω  = Triangulation(model)
  dΩ = Measure(Ω,4)

  sol(x) = sum(x)
  a((u1,u2),(v1,v2)) = ∫(u1⋅v1 + u2⋅v2)*dΩ
  l((v1,v2)) = ∫(sol⋅v1 - sol⋅v2)*dΩ

  op = AffineFEOperator(a,l,Y,Y)
  A, b = get_matrix(op), get_vector(op);

  # 1) From system blocks
  s1  = BlockDiagonalSolver([LUSolver(),LUSolver()])
  ss1 = symbolic_setup(s1,A)
  ns1 = numerical_setup(ss1,A)
  numerical_setup!(ns1,A)

  x1  = allocate_in_domain(A); fill!(x1,0.0)
  solve!(x1,ns1,b)

  # 2) From matrix blocks
  s2  = BlockDiagonalSolver([A[Block(1,1)],A[Block(2,2)]],[LUSolver(),LUSolver()])
  ss2 = symbolic_setup(s2,A)
  ns2 = numerical_setup(ss2,A)
  numerical_setup!(ns2,A)

  x2  = allocate_in_domain(A); fill!(x2,0.0)
  solve!(x2,ns2,b)

  # 3) From weakform blocks
  aii = (u,v) -> ∫(u⋅v)*dΩ
  s3  = BlockDiagonalSolver([aii,aii],[V,V],[V,V],[LUSolver(),LUSolver()])
  ss3 = symbolic_setup(s3,A)
  ns3 = numerical_setup(ss3,A)
  numerical_setup!(ns3,A)

  x3  = allocate_in_domain(A); fill!(x3,0.0)
  solve!(x3,ns3,b)
end

end # module