module Debugging

using IterativeSolvers
using FillArrays
using Gridap
using Gridap.Adaptivity
using Gridap.FESpaces

using GridapDistributed
using PartitionedArrays

function assemble_matrix_and_vector_bis(a,l,U,V)
  u_dir = zero(UH)
  u = get_trial_fe_basis(U)
  v = get_fe_basis(V)

  assem = SparseMatrixAssembler(U,V)

  matcontribs, veccontribs = a(u,v), l(v)
  data = collect_cell_matrix_and_vector(U,V,matcontribs,veccontribs,u_dir)
  A,b = assemble_matrix_and_vector(assem,data)
  return A,b
end



backend = SequentialBackend()
parts   = get_part_ids(backend,(1,2))

domain = (0,1,0,1)
partition = Tuple(fill(4,2))
model = CartesianDiscreteModel(parts,domain,partition)

order = 1
u(x)  = 1.0
reffe = ReferenceFE(lagrangian,Float64,order)

V = TestFESpace(model,reffe;dirichlet_tags="boundary")
U = TrialFESpace(V,u)

uh = interpolate(u,U)

qorder = order*2+1
Ω  = Triangulation(model)
dΩ = Measure(Ω,qorder)

a(u,v) = ∫(v⋅u)*dΩ
l(v)   = ∫(v⋅uh)*dΩ
h(v)   = ∫(v⋅v)*dΩ


SAR = SparseMatrixAssembler(U,V)
FAR = SparseMatrixAssembler(U,V,FullyAssembledRows())

v = get_fe_basis(V)
vecdata = collect_cell_vector(V,l(v))

v_sar = assemble_vector(SAR,vecdata)
v_far = assemble_vector(FAR,vecdata)


end