
using Gridap
using Gridap.Geometry, Gridap.FESpaces, Gridap.Adaptivity

u_h1(x) = x[1]*(1-x[1])*x[2]*(1-x[2]) #x[1] + x[2]
u_hdiv(x) = VectorValue(x[1],x[2])

a_h1(u,v,dΩ) = ∫(u⋅v)*dΩ
a_hdiv(u,v,dΩ) = ∫(u⋅v + divergence(u)⋅divergence(v))*dΩ

conf  = :hdiv
order = 1

cmodel = CartesianDiscreteModel((0,1,0,1),(8,8))
fmodel = refine(cmodel)

Ωh = Triangulation(fmodel)
ΩH = Triangulation(cmodel)

qdegree = 2*(order+1)
dΩh = Measure(Ωh,qdegree)
dΩH = Measure(ΩH,qdegree)
dΩHh = Measure(ΩH,Ωh,qdegree)

if conf == :h1
  u_bc  = u_h1
  reffe = ReferenceFE(lagrangian,Float64,order)
  ah(u,v) = a_h1(u,v,dΩh)
  aH(u,v) = a_h1(u,v,dΩH)
  f = zero(Float64)
else
  u_bc  = u_hdiv
  reffe = ReferenceFE(raviart_thomas,Float64,order)
  ah(u,v) = a_hdiv(u,v,dΩh)
  aH(u,v) = a_hdiv(u,v,dΩH)
  f = zero(VectorValue{2,Float64})
end
lh(v) = ∫(v⋅f)*dΩh
lH(v) = ∫(v⋅f)*dΩH

VH = TestFESpace(cmodel,reffe,dirichlet_tags="boundary")
UH = TrialFESpace(VH,u_bc)
Vh = TestFESpace(fmodel,reffe,dirichlet_tags="boundary")
Uh = TrialFESpace(Vh,u_bc)

oph = AffineFEOperator(ah,lh,Uh,Vh)
opH = AffineFEOperator(aH,lH,UH,VH)

xh_star = get_free_dof_values(solve(oph))
xH_star = get_free_dof_values(solve(opH))

Ah, bh = get_matrix(oph), get_vector(oph);
AH, bH = get_matrix(opH), get_vector(opH);

uh = interpolate(u_bc,Uh)
uH = interpolate(u_bc,UH)

xh = get_free_dof_values(uh)
xH = get_free_dof_values(uH)

eh = xh_star - xh
eH = xH_star - xH

rh = bh - Ah*xh
rH = bH - AH*xH
norm(Ah*eh - rh)
norm(AH*eH - rH)

uh0 = FEFunction(Vh,xh)
yh  = bh - assemble_vector(v -> ah(uh0,v),Vh)
norm(rh-yh)

function project_sol_c2f(xH)
  uH = FEFunction(UH,xH)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩh,v->∫(v⋅uH)*dΩh,Uh,Vh)
  return get_matrix(op)\get_vector(op)
end

function project_err_c2f(xH)
  uH = FEFunction(VH,xH)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩh,v->∫(v⋅uH)*dΩh,Vh,Vh)
  return get_matrix(op)\get_vector(op)
end

#function project_err_f2c(xh)
#  uh = FEFunction(Vh,xh)
#  uH = interpolate(uh,VH)
#  return get_free_dof_values(uH)
#end

function project_err_f2c(xh)
  uh = FEFunction(Vh,xh)
  op = AffineFEOperator((u,v)->∫(u⋅v)*dΩH,v->∫(v⋅uh)*dΩHh,VH,VH)
  return get_matrix(op)\get_vector(op)
end

yh = project_sol_c2f(xH)
norm(xh-yh)

yh = project_err_c2f(eH)
norm(eh-yh)
norm(rh-Ah*yh)

########################################

hH = get_cell_measure(ΩH)
hh = get_cell_measure(Ωh)

function dotH(a::AbstractVector,b::AbstractVector)
  _a = FEFunction(VH,a)
  _b = FEFunction(VH,b)
  dotH(_a,_b)
end

function dotH(a,b)
  sum(∫(a⋅b)*dΩH)
end

function doth(a::AbstractVector,b::AbstractVector)
  _a = FEFunction(Vh,a)
  _b = FEFunction(Vh,b)
  doth(_a,_b)
end

function doth(a,b)
  sum(∫(a⋅b)*dΩh)
end

g = rh
z = Ah\g
zm1 = fill(0.1,size(Ah,2))

gH = project_err_f2c(g-Ah*zm1)
qH = AH\gH

wH = randn(size(AH,2))
wh = get_free_dof_values(interpolate(FEFunction(VH,wH),Vh))
#wh = project_err_c2f(wH)

dotH(AH*qH,wH)
dotH(gH,wH)
doth(g-Ah*zm1,wh)
doth(Ah*(z-zm1),wh)


gH = project_err_f2c(g)
dotH(gH,wH)
doth(g,wh)
gh = project_err_c2f(gH)
doth(gh-g,wh)
