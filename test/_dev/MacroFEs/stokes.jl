
using Gridap
using Gridap.Adaptivity, Gridap.Geometry, Gridap.ReferenceFEs, Gridap.FESpaces, Gridap.Arrays

using FillArrays

u_sol(x) = VectorValue(x[1]^2*x[2], -x[1]*x[2]^2)
p_sol(x) = (x[1] - 1.0/2.0)

model = simplexify(CartesianDiscreteModel((0,1,0,1),(50,50)))
labels = get_face_labeling(model)
add_tag_from_tags!(labels,"top",[6])
add_tag_from_tags!(labels,"walls",[1,2,3,4,5,7,8])

order = 3
rrule = Adaptivity.BarycentricRefinementRule(TRI)
reffes = Fill(LagrangianRefFE(VectorValue{2,Float64},TRI,order),Adaptivity.num_subcells(rrule))
reffe_u = Adaptivity.MacroReferenceFE(rrule,reffes)

reffe_p = LagrangianRefFE(Float64,TRI,order-1)
#reffe_p = Adaptivity.MacroReferenceFE(rrule,reffes;conformity=L2Conformity())

qdegree = 2*order
quad  = Quadrature(TRI,Adaptivity.CompositeQuadrature(),rrule,qdegree)

V = FESpace(model,reffe_u,dirichlet_tags=["boundary"])
Q = FESpace(model,reffe_p,conformity=:L2,constraint=:zeromean)
U = TrialFESpace(V,u_sol)

Ω = Triangulation(model)
dΩ = Measure(Ω,quad)
@assert abs(sum(∫(p_sol)dΩ)) < 1.e-15
@assert abs(sum(∫(divergence(u_sol))dΩ)) < 1.e-15

X = MultiFieldFESpace([U,Q])
Y = MultiFieldFESpace([V,Q])

f(x) = -Δ(u_sol)(x) + ∇(p_sol)(x)
lap(u,v) = ∫(∇(u)⊙∇(v))dΩ
a((u,p),(v,q)) = lap(u,v) + ∫(divergence(u)*q - divergence(v)*p)dΩ
l((v,q)) = ∫(f⋅v)dΩ

op = AffineFEOperator(a,l,X,Y)
xh = solve(op)
uh, ph = xh
sum(∫(uh⋅uh)dΩ)
sum(∫(ph)dΩ)

eh_u = uh - u_sol
eh_p = ph - p_sol
sum(∫(eh_u⋅eh_u)dΩ)
sum(∫(eh_p*eh_p)dΩ)

writevtk(
  Ω,"stokes",
  cellfields=["u"=>uh,"p"=>ph,"eu"=>eh_u,"ep"=>eh_p,"u_sol"=>u_sol,"p_sol"=>p_sol],
  append=false,
)
