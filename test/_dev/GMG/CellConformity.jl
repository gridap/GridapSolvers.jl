
using FillArrays
using Gridap
using Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.CellData

model = CartesianDiscreteModel((0,1,0,1),(3,3))

reffe = LagrangianRefFE(Float64,QUAD,1)
conf = H1Conformity()

V = FESpace(model,reffe)

cell_conformity = CellConformity(Fill(reffe,num_cells(model)),conf)
cell_dofs = get_fe_dof_basis(V)

data = CellData.get_data(cell_dofs)
dofs = data[1]


