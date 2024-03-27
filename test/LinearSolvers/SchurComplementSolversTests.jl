module SchurComplementSolversTests

using Test
using BlockArrays
using Gridap
using Gridap.MultiField, Gridap.Algebra
using Gridap.Algebra
using Gridap.Geometry
using Gridap.FESpaces
using Gridap.ReferenceFEs

using PartitionedArrays
using GridapDistributed

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools

function l2_error(xh,sol,dΩ)
  eh = xh - sol
  e  = sum(∫(eh⋅eh)dΩ)
  return e
end

function l2_error(x,sol,X,dΩ)
  xh = FEFunction(X,x)
  return l2_error(xh,sol,dΩ)
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

# Darcy solution
const β_U = 50.0
const γ = 100.0

u_ref(x) = VectorValue(x[1]+x[2],-x[2])
p_ref(x) = 2.0*x[1]-1.0
f_ref(x) = u_ref(x) + ∇(p_ref)(x)

function main(distribute,np)
  parts = distribute(LinearIndices((prod(np),)))
  model = get_mesh(parts,np)

  labels = get_face_labeling(model)
  add_tag_from_tags!(labels,"dirichlet",[1,2,3,4,5,6,7])
  add_tag_from_tags!(labels,"newmann",[8,])

  order  = 0
  reffeᵤ = ReferenceFE(raviart_thomas,Float64,order)
  V  = TestFESpace(model,reffeᵤ,conformity=:HDiv,dirichlet_tags="dirichlet")
  U  = TrialFESpace(V,u_ref)

  reffeₚ = ReferenceFE(lagrangian,Float64,order;space=:P)
  Q = TestFESpace(model,reffeₚ,conformity=:L2)
  P = TrialFESpace(Q,p_ref)

  mfs = BlockMultiFieldStyle()
  Y = MultiFieldFESpace([V, Q];style=mfs)
  X = MultiFieldFESpace([U, P];style=mfs)

  qdegree = 4
  Ω   = Triangulation(model)
  dΩ  = Measure(Ω,qdegree)

  Γ_N  = BoundaryTriangulation(model;tags="newmann")
  dΓ_N = Measure(Γ_N,qdegree)
  n_Γ_N = get_normal_vector(Γ_N)

  a(u,v) = ∫(v⊙u)dΩ + ∫(γ*(∇⋅v)*(∇⋅u))dΩ
  b(p,v) = ∫(-(∇⋅v)*p)dΩ
  c(u,q) = ∫(- q*(∇⋅u))dΩ

  biform((u,p),(v,q)) = a(u,v) + b(p,v) + c(u,q)
  liform((v,q)) = ∫(f_ref⋅v)dΩ - ∫((v⋅n_Γ_N)⋅p_ref)dΓ_N

  op = AffineFEOperator(biform,liform,X,Y)
  sysmat, sysvec = get_matrix(op), get_vector(op);

  ############################################################################################
  # Solve by GMRES preconditioned with inexact Schur complement

  s(p,q) = ∫(γ*p*q)dΩ
  PS = assemble_matrix(s,P,Q)
  PS_solver = LUSolver()
  PS_ns = numerical_setup(symbolic_setup(PS_solver,PS),PS)

  A = sysmat[Block(1,1)]
  A_solver = LUSolver()
  A_ns = numerical_setup(symbolic_setup(A_solver,A),A)

  B = sysmat[Block(1,2)]; C = sysmat[Block(2,1)]
  psc_solver = SchurComplementSolver(A_ns,B,C,PS_ns);

  gmres = GMRESSolver(20;Pr=psc_solver,rtol=1.e-10,verbose=i_am_main(parts))
  gmres_ns = numerical_setup(symbolic_setup(gmres,sysmat),sysmat)

  x = allocate_in_domain(sysmat)
  solve!(x,gmres_ns,sysvec)

  xh = FEFunction(X,x)
  uh, ph = xh
  #@test l2_error(uh,u_ref,dΩ) < 1.e-4
  #@test l2_error(ph,p_ref,dΩ) < 1.e-4
end

end