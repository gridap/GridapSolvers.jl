
using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra, Gridap.Geometry, Gridap.FESpaces
using Gridap.CellData, Gridap.MultiField, Gridap.Algebra
using PartitionedArrays
using GridapDistributed
using GridapP4est

np = 4
parts = with_mpi() do distribute 
  distribute(LinearIndices((np,)))
end

nc = (8,8)
domain = (0,1,0,1)
cmodel = CartesianDiscreteModel(domain,nc)

num_refs_coarse = 0
model = OctreeDistributedDiscreteModel(parts,cmodel,num_refs_coarse)
#model = CartesianDiscreteModel(parts,(2,2),domain,nc) # ALL TESTS RUN OK

order = 2
reffe_u = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
reffe_p = ReferenceFE(lagrangian,Float64,order-1,space=:P)

V = TestFESpace(model,reffe_u)
Q = TestFESpace(model,reffe_p;conformity=:L2,constraint=:zeromean)
#Q = TestFESpace(model,reffe_p;conformity=:L2) # ALL TESTS RUN OK

mfs = Gridap.MultiField.BlockMultiFieldStyle()
#mfs = Gridap.MultiField.ConsecutiveMultiFieldStyle() # ALL TESTS RUN OK
X = MultiFieldFESpace([V,Q];style=mfs)
Y = MultiFieldFESpace([Q,Q];style=mfs)

qdegree = 4
Ω = Triangulation(model)
dΩ = Measure(Ω,qdegree)

m(p,q) = ∫(p*q)dΩ
M = assemble_matrix(m,Q,Q) # OK

n(u,q) = ∫((∇⋅u)*q)dΩ
N = assemble_matrix(n,V,Q) # OK

l((p1,p2),(q1,q2)) = ∫(p1*q1 + p2*q2 + p1*q2)dΩ
L = assemble_matrix(l,Y,Y) # OK

b((u,p),(v,q)) = ∫(∇(v)⊙∇(u))dΩ + m(p,q)
B = assemble_matrix(b,X,X) # OK

a((u,p),(v,q)) = ∫(∇(v)⊙∇(u))dΩ + m(p,q) - ∫((∇⋅v)*p)dΩ - ∫((∇⋅u)*q)dΩ
A = assemble_matrix(a,X,X) # FAILS
