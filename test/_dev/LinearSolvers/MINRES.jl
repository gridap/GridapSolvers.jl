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

function add_labels_2d!(labels)
  add_tag_from_tags!(labels,"top",[3,4,6])
  add_tag_from_tags!(labels,"walls",[1,2,5,7,8])
end

function add_labels_3d!(labels)
  add_tag_from_tags!(labels,"top",[5,6,7,8,11,12,15,16,22])
  add_tag_from_tags!(labels,"walls",[1,2,3,4,9,10,13,14,17,18,19,20,21,23,24,25,26])
end

nc = (10,10)
Dc = length(nc)
domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)

model = CartesianDiscreteModel(domain,nc)
add_labels! = (Dc == 2) ? add_labels_2d! : add_labels_3d!
add_labels!(get_face_labeling(model))

order = 2
qdegree = 2*(order+1)
reffe_u = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

u_walls = (Dc==2) ? VectorValue(0.0,0.0) : VectorValue(0.0,0.0,0.0)
u_top = (Dc==2) ? VectorValue(1.0,0.0) : VectorValue(1.0,0.0,0.0)

V = TestFESpace(model,reffe_u,dirichlet_tags=["walls","top"]);
U = TrialFESpace(V,[u_walls,u_top]);
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([U,Q];style=mfs)
Y = MultiFieldFESpace([V,Q];style=mfs)

α = 10.0
f = (Dc==2) ? VectorValue(1.0,1.0) : VectorValue(1.0,1.0,1.0)
poly = (Dc==2) ? QUAD : HEX
Π_Qh = LocalProjectionMap(divergence,lagrangian,Float64,order-1;space=:P)
graddiv(u,v,dΩ) = ∫(α*Π_Qh(u,dΩ)⋅Π_Qh(v,dΩ))dΩ
biform_u(u,v,dΩ) = ∫(∇(v)⊙∇(u))dΩ + graddiv(u,v,dΩ)
biform((u,p),(v,q),dΩ) = biform_u(u,v,dΩ) - ∫(divergence(v)*p)dΩ - ∫(divergence(u)*q)dΩ
liform((v,q),dΩ) = ∫(v⋅f)dΩ

Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

a(u,v) = biform(u,v,dΩ)
l(v) = liform(v,dΩ)
op = AffineFEOperator(a,l,X,Y)
A, b = get_matrix(op), get_vector(op);

solver_u = LUSolver()
solver_p = CGSolver(JacobiLinearSolver();maxiter=20,atol=1e-14,rtol=1.e-6,verbose=true)
solver_p.log.depth = 2

β = max(1.0,α)
bblocks = [LinearSystemBlock(),BiformBlock((p,q) -> ∫((1.0/β)*p*q)dΩ,Q,Q)] 
P = BlockDiagonalSolver(bblocks,[solver_u,solver_p])
solver = MINRESSolver(;Pl=P,atol=1e-14,rtol=1.e-8,verbose=true)
ns = numerical_setup(symbolic_setup(solver,A),A)

x = allocate_in_domain(A); fill!(x,0.0)
solve!(x,ns,b);


###############################################

X2 = MultiFieldFESpace([U,Q])
Y2 = MultiFieldFESpace([V,Q])

op2 = AffineFEOperator(a,l,X2,Y2)
A2, b2 = get_matrix(op2), get_vector(op2);
x_ref = A2\b2

p((u,p),(v,q)) = biform_u(u,v,dΩ) + ∫((1.0/β)*p*q)dΩ
P2_mat = assemble_matrix(p,X2,Y2)

P2 = LinearSolvers.MatrixSolver(P2_mat)
solver2 = MINRESSolver(;Pl=P2,atol=1e-14,rtol=1.e-8,verbose=true)
ns2 = numerical_setup(symbolic_setup(solver2,A2),A2)

x2 = allocate_in_domain(A2); fill!(x2,0.0)
solve!(x2,ns2,b2);

norm(A*x-b), norm(A2*x_ref-b2)
norm(x-x_ref), norm(x2-x_ref)
