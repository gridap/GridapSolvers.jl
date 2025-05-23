module SmoothersTests

using Test
using MPI
using Gridap, Gridap.Algebra
using GridapDistributed
using PartitionedArrays

using GridapSolvers
using GridapSolvers.LinearSolvers

function smoothers_driver(parts,model,P)
  sol(x) = x[1] + x[2]
  f(x)   = -Δ(sol)(x)

  order  = 1
  qorder = order*2 + 1
  reffe  = ReferenceFE(lagrangian,Float64,order)
  Vh     = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
  Uh     = TrialFESpace(Vh,sol)

  Ω      = Triangulation(model)
  dΩ     = Measure(Ω,qorder)
  a(u,v) = ∫(∇(v)⋅∇(u))*dΩ
  l(v)   = ∫(v⋅f)*dΩ

  op = AffineFEOperator(a,l,Uh,Vh)
  A, b = get_matrix(op), get_vector(op)

  solver = CGSolver(P;rtol=1.0e-8,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)
  x = allocate_in_domain(A); fill!(x,zero(eltype(x)))
  solve!(x,ns,b)

  u  = interpolate(sol,Uh)
  uh = FEFunction(Uh,x)
  eh = uh - u
  E  = sum(∫(eh*eh)*dΩ)
  if i_am_main(parts)
    println("L2 Error: ", E)
  end
  
  @test E < 1.e-8
end

function main_smoother_driver(parts,model,smoother)
  if smoother === :richardson
    S = RichardsonSmoother(JacobiLinearSolver(),5,2.0/3.0)
  elseif smoother === :sym_gauss_seidel
    S = SymGaussSeidelSmoother(5)
  else
    error("Unknown smoother")
  end
  P = LinearSolverFromSmoother(S)
  smoothers_driver(parts,model,P)
end

function get_mesh(parts,np)
  Dc = length(np)
  if Dc == 2
    domain = (0,1,0,1)
    nc = (8,8)
  else
    @assert Dc == 3
    domain = (0,1,0,1,0,1)
    nc = (8,8,8)
  end
  if prod(np) == 1
    model = CartesianDiscreteModel(domain,nc)
  else
    model = CartesianDiscreteModel(parts,np,domain,nc)
  end
  return model
end

function main(distribute,np)
  parts = distribute(LinearIndices((prod(np),)))
  model = get_mesh(parts,np)

  for smoother in [:richardson,:sym_gauss_seidel]
    if i_am_main(parts)
      println(repeat("=",80))
      println("Testing smoother $smoother with Dc=$(length(np))")
    end
    main_smoother_driver(parts,model,smoother)
  end
end

end # module SmoothersTests