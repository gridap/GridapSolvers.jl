
module StokesApplication

using Test
using LinearAlgebra
using FillArrays, BlockArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers, GridapSolvers.MultilevelTools
using GridapSolvers.BlockSolvers: LinearSystemBlock, BiformBlock, BlockTriangularSolver

function main(distribute,np,nc)
  parts = distribute(LinearIndices((prod(np),)))

  # Geometry
  model = CartesianDiscreteModel(parts,np,(0,1,0,1),nc)
  labels = get_face_labeling(model);
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"walls",[1,2,5,7,8])

  # FE spaces
  order = 2
  qdegree = 2*(order+1)
  Dc = length(nc)
  reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
  reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

  u_wall = VectorValue(0.0,0.0)
  u_top = VectorValue(1.0,0.0)

  V = TestFESpace(model,reffe_u,dirichlet_tags=["walls","top"]);
  U = TrialFESpace(V,[u_wall,u_top]);
  Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

  mfs = Gridap.MultiField.BlockMultiFieldStyle()
  X = MultiFieldFESpace([U,Q];style=mfs)
  Y = MultiFieldFESpace([V,Q];style=mfs)

  # Weak formulation
  α = 1.e2
  f = VectorValue(1.0,1.0)
  Π_Qh = LocalProjectionMap(QUAD,lagrangian,Float64,order-1;quad_order=qdegree,space=:P)
  graddiv(u,v,dΩ) = ∫(α*Π_Qh(divergence(u))⋅Π_Qh(divergence(v)))dΩ
  biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ + graddiv(u,v,dΩ)
  biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
  liform((v,q),dΩ) = ∫(v⋅f)dΩ

  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)

  a(u,v) = biform(u,v,dΩ)
  l(v) = liform(v,dΩ)
  op = AffineFEOperator(a,l,X,Y)
  A, b = get_matrix(op), get_vector(op);

  # Solver
  solver_u = LUSolver() # or mumps
  solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
  solver_p.log.depth = 2

  diag_blocks  = [LinearSystemBlock(),BiformBlock((p,q) -> ∫(-1.0/α*p*q)dΩ,Q,Q)]
  bblocks = map(CartesianIndices((2,2))) do I
    (I[1] == I[2]) ? diag_blocks[I[1]] : LinearSystemBlock()
  end
  coeffs = [1.0 1.0;
            0.0 1.0]  
  P = BlockTriangularSolver(bblocks,[solver_u,solver_p],coeffs,:upper)
  solver = FGMRESSolver(20,P;atol=1e-14,rtol=1.e-8,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)

  x = allocate_in_domain(A); fill!(x,0.0)
  solve!(x,ns,b)

  xh = FEFunction(X,x)
  @test norm(b - A*x) < 1.e-8
end

end # module