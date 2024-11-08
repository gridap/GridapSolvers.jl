
using Gridap
using GridapDistributed, PartitionedArrays
using GridapSolvers
using GridapSolvers.MultilevelTools
using Gridap.Arrays, Gridap.CellData, Gridap.FESpaces

np = (2,2,1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

function half_empty_trian(ranks,model)
  cell_ids = get_cell_gids(model)
  trians = map(ranks,local_views(model),partition(cell_ids)) do rank, model, ids
    cell_mask = zeros(Bool, num_cells(model))
    if rank ∈ (3,4)
      cell_mask[own_to_local(ids)] .= true
    end
    Triangulation(model,cell_mask)
  end
  GridapDistributed.DistributedTriangulation(trians,model)
end

layer(x) = sign(x)*abs(x)^(1/3)
cmap(x) = VectorValue(layer(x[1]),layer(x[2]),x[3])
model = CartesianDiscreteModel(ranks,np,(0,1,0,1,0,1),(5,5,5),map=cmap)

#Ω = Triangulation(model)
Ω = half_empty_trian(ranks,model)
dΩ = Measure(Ω,2)

order = 2
qdegree = 2*(order)
reffe_u = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1;space=:P)

V = TestFESpace(Ω,reffe_u,dirichlet_tags="boundary");
Q = TestFESpace(Ω,reffe_p;conformity=:L2) 

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

u,p = get_trial_fe_basis(X)
v,q = get_fe_basis(X)

Πu = Π_Qh(u)
∇v = ∇⋅(v)
Πu ⋅ ∇v

∫(Π_Qh(u)⋅(∇⋅v))dΩ

cf = Π_Qh(u)⋅(∇⋅v)

i = 1
integrate(cf.fields.items[i],dΩ.measures.items[i])

map(num_cells,local_views(Ω))

data = Πu.fields.items[1].cell_field.args[1].args[1]
testitem(data)

length(u.fields.items[1].cell_basis)

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
testitem(cf2.cell_field.args[1].args[1])
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
