module NonlinearSolversTests

using Test
using LinearAlgebra
using FillArrays, BlockArrays
using LineSearches: BackTracking

using GridapDistributed, PartitionedArrays

using Gridap
using Gridap.Algebra

using GridapSolvers
using GridapSolvers.NonlinearSolvers, GridapSolvers.LinearSolvers

const ModelTypes = Union{<:DiscreteModel,<:GridapDistributed.DistributedDiscreteModel}

function main(ranks,model::ModelTypes,solver::Symbol)
  u_sol(x) = sum(x)

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  Vh = TestFESpace(model,reffe,dirichlet_tags=["boundary"])
  Uh = TrialFESpace(Vh,u_sol)
  
  degree = 4*order
  Ω = Triangulation(model)
  dΩ = Measure(Ω,degree)

  α(u) = (1 + u + u^2)
  f(x) = α(u_sol(x))
  res(u,v) = ∫((α∘u)⋅v - f*v)dΩ
  
  op = FEOperator(res,Uh,Vh)
  ls = GMRESSolver(10,Pr=JacobiLinearSolver(),maxiter=50,verbose=false)

  if solver == :nlsolvers_newton
    nls = NLsolveNonlinearSolver(ls; show_trace=true, method=:newton, linesearch=BackTracking(), iterations=20)
  elseif solver == :nlsolvers_trust_region
    nls = NLsolveNonlinearSolver(ls; show_trace=true, method=:trust_region, linesearch=BackTracking(), iterations=20)
  elseif solver == :nlsolvers_anderson
    nls = NLsolveNonlinearSolver(ls; show_trace=true, method=:anderson, linesearch=BackTracking(), iterations=40, m=10)
  elseif solver == :newton
    nls = NewtonSolver(ls,maxiter=20,verbose=true)
  elseif solver == :newton_continuation
    op = ContinuationFEOperator(op,op,2;reuse_caches=true)
    nls = NewtonSolver(ls,maxiter=20,verbose=true)
  else
    @error "Unknown solver"
  end

  solver = FESolver(nls)
  uh0 = interpolate(0.01,Uh)
  uh, = solve!(uh0,solver,op)

  @test norm(residual(op,uh)) < 1e-6
end

# Serial
function main(solver::Symbol)
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  ranks = DebugArray([1])
  main(ranks,model,solver)
end

# Distributed
function main(distribute,np,solver::Symbol)
  ranks = distribute(LinearIndices((prod(np),)))
  model = CartesianDiscreteModel((0,1,0,1),(8,8))
  main(ranks,model,solver)
end

end #module