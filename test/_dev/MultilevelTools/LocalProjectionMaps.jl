
using Gridap
using GridapDistributed, PartitionedArrays
using GridapSolvers
using GridapSolvers.MultilevelTools
using Gridap.Arrays, Gridap.CellData, Gridap.FESpaces

np = (2,2)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

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

Π_Qh = LocalProjectionMap(divergence,reffe_p,qdegree)
#Π_Qh = LocalProjectionMap(divergence,Q,qdegree)

A = assemble_matrix((u,v) -> ∫(∇(v)⊙∇(u))dΩ, V, V)
D = assemble_matrix((u,v) -> ∫(Π_Qh(u)⋅(∇⋅v))dΩ, V, V)
B = assemble_matrix((u,q) -> ∫(divergence(u)*q)dΩ, V, Q)
Bt = assemble_matrix((p,v) -> ∫(divergence(v)*p)dΩ, Q, V)
Mp = assemble_matrix((p,q) -> ∫(p⋅q)dΩ, Q, Q)
@assert Bt ≈ B'

F = Bt*inv(Matrix(Mp))*B
G = Matrix(D)

F-G
norm(F-G)
maximum(abs.(F-G))

#######################################################################

Ωf = Triangulation(model,tags="interior")
Vf = TestFESpace(Ωf,reffe_u);
Qf = TestFESpace(Ωf,reffe_p;conformity=:L2)
Xf = MultiFieldFESpace([Vf,Qf];style=mfs)

Ωv = Triangulation(model,[1,2,3])
dΩv = Measure(Ωv,2)

u1 = get_trial_fe_basis(Vf)
cf1 = Π_Qh(u1)
∫(cf1)dΩv # OK

u2 = get_trial_fe_basis(Xf)[1]
cf2 = Π_Qh(u2)
∫(cf2)dΩv # Not OK

cf2_bis = GenericCellField(cf2.cell_field.args[1].args[1],Ωf,ReferenceDomain())
∫(cf2_bis)dΩv # OK

u3 = get_trial_fe_basis(Xf)[2]
∫(u3)dΩv # OK

u3_bis = ∇⋅(u3)
u3_bisbis = change_domain(u3_bis,Ωv,ReferenceDomain())

###################################

eltype(cf2.cell_field.args[1].args[1])
eltype(cf2.cell_field.args[1])

val = testitem(cf2.cell_field.args[1])
Gridap.Geometry._similar_empty(val)

val_copy = deepcopy(val)

Gridap.Geometry._similar_empty(val.array[1])

val1 = val.array[1]
zs = 0 .* size(val1)
void = similar(val,eltype(val1),zs)
