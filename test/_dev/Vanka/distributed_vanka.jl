
using Gridap
using PartitionedArrays
using GridapDistributed


np = (2,1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(4,4))

order = 1
reffe = ReferenceFE(lagrangian,Float64,order)
Vh = TestFESpace(model,reffe)

Ω = Triangulation(model)
dΩ = Measure(Ω,2*order)

a(u,v) = ∫(v⋅u)*dΩ
A = assemble_matrix(a,Vh,Vh)


rows, cols = axes(A)

nbors_snd, nbors_rcv = assembly_neighbors(partition(rows))

map(partition(rows),partition(cols),partition(A)) do rows,cols,mat
  own_rows = own_to_local(rows)
end

