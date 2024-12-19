
using Gridap
using Gridap.Algebra

using GridapSolvers
using GridapSolvers.SolverInterfaces, GridapSolvers.LinearSolvers

using GridapSolvers.SolverInterfaces: NullSpace, is_orthonormal, is_orthogonal, gram_schmidt!, modified_gram_schmidt!
using GridapSolvers.SolverInterfaces: project, project!, make_orthogonal!, reconstruct, reconstruct!

A = [1.0 0.0 0.0; 0.0 0.0 0.0; 0.0 0.0 0.0]
N = NullSpace(A)

is_orthonormal(N)
is_orthogonal(N,[1.0,0.0,0.0])
is_orthogonal(N,[0.0,1.0,0.0])

v = [1.0,2.0,3.0]
p, α = project(N,v)
w, β = make_orthogonal!(N,copy(v))
u = reconstruct(N,w,α)
u ≈ v

V1 = [[2.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,1.0]]
gram_schmidt!(V1)
is_orthonormal(NullSpace(V1))

V2 = [[2.0,1.0,1.0],[1.0,2.0,1.0],[1.0,1.0,1.0]]
modified_gram_schmidt!(V2)
is_orthonormal(NullSpace(V2))
V1 ≈ V2


############################################################################################

model = CartesianDiscreteModel((0,1,0,1),(4,4))
Ω = Triangulation(model)
Γ = BoundaryTriangulation(model)

dΩ = Measure(Ω,4)
dΓ = Measure(Γ,4)
n = get_normal_vector(Γ)

V = FESpace(model, ReferenceFE(lagrangian,Float64,1))

u_exact(x) = x[1]*x[2]
f(x) = Δ(u_exact)(x)
a(u,v) = ∫(∇(u)⋅∇(v))*dΩ
l(v) = ∫(f*v)*dΩ + ∫(v*(∇(u_exact)⋅n))*dΓ

op = AffineFEOperator(a,l,V,V)
A, b = get_matrix(op), get_vector(op)

N = NullSpace(ones(size(A,2)))
K = SolverInterfaces.matrix_representation(N)
norm(A*K)

s = GMRESSolver(10;verbose=true)
s = LUSolver()
solver = NullspaceSolver(s,N;constrain_matrix=true)
ns = numerical_setup(symbolic_setup(solver,A),A)

x = allocate_in_domain(A)
fill!(x,0.0)
solve!(x,ns,b)
norm(A*x-b)
