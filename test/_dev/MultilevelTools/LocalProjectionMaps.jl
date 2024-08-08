
using Gridap
using GridapDistributed, PartitionedArrays
using GridapSolvers
using GridapSolvers.MultilevelTools

layer(x) = sign(x)*abs(x)^(1/3)
cmap(x) = VectorValue(layer(x[1]),layer(x[2]))
model = CartesianDiscreteModel((0,-1,0,-1),(10,10),map=cmap)

Ω = Triangulation(model)
dΩ = Measure(Ω,2)

order = 2
qdegree = 2*(order+1)
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

V = TestFESpace(model,reffe_u,dirichlet_tags="boundary");
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean) 

mfs = Gridap.MultiField.BlockMultiFieldStyle()
X = MultiFieldFESpace([V,Q];style=mfs)

#Π_Qh = LocalProjectionMap(divergence,reffe_p)
Π_Qh = LocalProjectionMap(divergence,Q)

A = assemble_matrix((u,v) -> ∫(∇(v)⊙∇(u))dΩ, V, V)
D = assemble_matrix((u,v) -> ∫(Π_Qh(u,dΩ)⋅Π_Qh(v,dΩ))dΩ, V, V)
B = assemble_matrix((u,q) -> ∫(divergence(u)*q)dΩ, V, Q)
Bt = assemble_matrix((p,v) -> ∫(divergence(v)*p)dΩ, Q, V)
Mp = assemble_matrix((p,q) -> ∫(p⋅q)dΩ, Q, Q)
@assert Bt ≈ B'

F = Bt*inv(Matrix(Mp))*B
G = Matrix(D)

F-G
norm(F-G)
