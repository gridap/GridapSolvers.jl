"""
    struct RichardsonLinearSolver <: LinearSolver 
      ...
    end

    RichardsonLinearSolver(ω,IterMax;Pl=nothing,rtol=1e-10,atol=1e-6,verbose=true,name = "RichardsonLinearSolver")

  Richardson Iteration, with an optional left preconditioners `Pl`.

  The relaxation parameter (ω) can either be of type Float64 or Vector{Float64}. 
  This gives flexiblity in relaxation.  
"""

struct RichardsonLinearSolver<:Gridap.Algebra.LinearSolver
    ω::Union{Vector{Float64},Float64}
    Pl::Union{Gridap.Algebra.LinearSolver,Nothing}
    IterMax::Int64
    log::ConvergenceLog{Float64}
end

function RichardsonLinearSolver(ω,IterMax;Pl=nothing,rtol=1e-10,atol=1e-6,verbose=true,name = "RichardsonLinearSolver")
    tols = SolverTolerances{Float64}(maxiter=IterMax,atol=atol,rtol=rtol)
    log  = ConvergenceLog(name,tols,verbose=verbose)
    return RichardsonLinearSolver(ω,Pl,IterMax,log)
end

struct RichardsonLinearSymbolicSetup<:Gridap.Algebra.SymbolicSetup
    solver
end

function Gridap.Algebra.symbolic_setup(solver::RichardsonLinearSolver,A::AbstractMatrix)
    return RichardsonLinearSymbolicSetup(solver)
end

function get_solver_caches(solver::RichardsonLinearSolver, A::AbstractMatrix)
    ω = solver.ω
    return ω
end

mutable struct RichardsonLinearNumericalSetup<:Gridap.Algebra.NumericalSetup
    solver
    A
    Pl_ns
    caches
end

function Gridap.Algebra.numerical_setup(ss::RichardsonLinearSymbolicSetup, A::AbstractMatrix)
    solver = ss.solver
    Pl_ns = !isnothing(solver.Pl) ? numerical_setup(symbolic_setup(solver.Pl,A),A) : nothing 
    caches = get_solver_caches(solver,A)
    return RichardsonLinearNumericalSetup(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup(ss::RichardsonLinearSymbolicSetup, A::AbstractMatrix, x::AbstractVector)
    solver = ss.solver
    Pl_ns = !isnothing(solver.Pl) ? numerical_setup(symbolic_setup(solver.Pl,A,x),A,x) : nothing 
    caches = get_solver_caches(solver,A)
    return RichardsonLinearNumericalSetup(solver,A,Pl_ns,caches)
end

function Gridap.Algebra.numerical_setup!(ns::RichardsonLinearNumericalSetup, A::AbstractMatrix)
    if !isa(ns.Pl_ns,Nothing)
        numerical_setup!(ns.Pl_ns,A)
    end
    ns.A = A 
    return ns 
end

function Gridap.Algebra.numerical_setup!(ns::RichardsonLinearNumericalSetup, A::AbstractMatrix, x::AbstractVector)
    if !isa(ns.Pl_ns,Nothing)
        numerical_setup!(ns.Pl_ns,A,x)
    end
    ns.A = A 
    return ns 
end

function Gridap.Algebra.solve!(x::AbstractVector, ns:: RichardsonLinearNumericalSetup, b::AbstractVector)
    solver,A,Pl,caches = ns.solver,ns.A,ns.Pl_ns,ns.caches
    ω = caches
    log = solver.log
    # Relaxation parameters if ω is of type Float64
    if isa(ω,Float64)
        temp = ω
        ω = temp.*ones(eltype(x), size(x))
    end
    # Initialization 
    z = 0*similar(x)
    r = 0*similar(x)
    # residual
    copyto!(r,b - mul!(r,A,x))
    done = init!(log,norm(r))
    if !isa(ns.Pl_ns,Nothing) # Case when a preconditioner is applied
        while !done
            solve!(z, Pl, r) # Apply preconditioner r = PZ 
            copyto!(x,x + ω.* z)
            copyto!(r,b - mul!(r,A,x))
            done = update!(log,norm(r))
        end
        finalize!(log,norm(r))
        return x
    else    # Case when no preconditioner is applied
        while !done
            copyto!(z,r)
            copyto!(x,x + ω.* z)
            copyto!(r,b - mul!(r,A,x))
            done = update!(log,norm(r))
        end
        finalize!(log,norm(r))
        return x
    end
end