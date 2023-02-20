module Debugging

using IterativeSolvers
using FillArrays
using Gridap
using Gridap.Adaptivity
using Gridap.FESpaces


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

"""
function Gridap.Adaptivity.FineToCoarseField(fine_fields,rrule::RefinementRule)
  return Gridap.Adaptivity.FineToCoarseField(collect(fine_fields),rrule)
end
"""

domain = (0,1,0,1)
partition = Tuple(fill(4,2))
model_H = CartesianDiscreteModel(domain,partition)
model_h = refine(model_H)

order = 1
u(x)  = 1.0
reffe = ReferenceFE(lagrangian,Float64,order)

VH = TestFESpace(model_H,reffe;dirichlet_tags="boundary")
UH = TrialFESpace(VH,u)
Vh = TestFESpace(model_h,reffe;dirichlet_tags="boundary")
Uh = TrialFESpace(Vh,u)

uh = interpolate(u,Uh)
uH = interpolate(u,UH)

qorder = order*2+1
ΩH  = Triangulation(model_H)
dΩH = Measure(ΩH,qorder)
Ωh  = Triangulation(model_h)
dΩh = Measure(Ωh,qorder)

dΩHh = Measure(ΩH,Ωh,qorder)

a(u,v) = ∫(v⋅u)*dΩH
lh(v)  = ∫(v⋅uh)*dΩHh
lH(v)  = ∫(v⋅uH)*dΩH

op = AffineFEOperator(a,lH,UH,VH)

AH, bH = assemble_matrix_and_vector_bis(a,lh,UH,VH)

xH = zeros(size(bH))
rH = AH*xH - bH
xH, hist = cg!(xH,AH,bH;log=true)
xH

uH2 = FEFunction(UH,xH)

pts = get_cell_points(dΩH)
uH(pts)
uH2(pts)


end