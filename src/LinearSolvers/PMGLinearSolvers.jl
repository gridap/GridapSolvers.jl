"""
    struct PMG{S1,S2,CS} <: Gridap.Algebra.LinearSolver

Implementation of a P-MultiGrid solver. 

Through the constructor kwargs, one can specify the smoothers and 
solver used for the pre, post smoothing steps and the coarsest level solve. 
"""
struct PMG{S1,S2,CS} <: Gridap.Algebra.LinearSolver
    sh            ::FESpaceHierarchy
    pre_smoother  ::S1
    post_smoother ::S2
    coarse_solver ::CS
    rtol          ::Float64
    maxiter       ::Int
    verbose       ::Bool
    mode          ::Symbol
end

function PMG(
    sh::FESpaceHierarchy;
    pre_smoother=JacobiSmoother(5),
    post_smoother=pre_smoother,
    coarse_solver=BackslashSolver(),
    rtol=1.0e-6,
    maxiter=1000,
    verbose=false,
    mode=:preconditioner)
    Gridap.Helpers.@check mode==:preconditioner || mode==:solver
    return PMG(sh,pre_smoother,post_smoother,coarse_solver,rtol,maxiter,verbose,mode)
end

struct PMGSymbolicSetup{S1,S2,CS} <: Gridap.Algebra.SymbolicSetup
    pmg               :: PMG
    ss_pre_smoothers  :: Vector{S1}
    ss_post_smoothers :: Vector{S2}
    ss_coarse_solver  :: CS
end

function Gridap.Algebra.symbolic_setup(pmg::PMG, sysmats)
    nlev = get_num_levels(pmg.sh)

    ss_pre_smoothers = map(mat -> symbolic_setup(pmg.pre_smoother,mat),sysmats[1:nlev-1])
    if pmg.post_smoother === pmg.pre_smoother
        ss_post_smoothers = ss_pre_smoothers
    else
        ss_post_smoothers = map(mat -> symbolic_setup(pmg.post_smoother,mat),sysmats[1:nlev-1])
    end
    ss_coarse_solver = symbolic_setup(pmg.coarse_solver,sysmats[nlev])

    return PMGSymbolicSetup(pmg,ss_pre_smoothers,ss_post_smoothers,ss_coarse_solver)
end

mutable struct PMGNumericalSetup{M,C,T,S1,S2,CS} <: Gridap.Algebra.NumericalSetup
    pmg               :: PMG
    sysmats           :: Vector{M}
    caches            :: Vector{C}
    transfers         :: Vector{T}
    ns_pre_smoothers  :: Vector{S1}
    ns_post_smoothers :: Vector{S2}
    ns_coarse_solver  :: CS
end

function get_pmg_caches(lev::Int, sysmats, sh::FESpaceHierarchy)
    nlev = length(sysmats)
    Adxh = fill(0.0,size(sysmats[lev],1))
    dxh  = fill(0.0,size(sysmats[lev],2))
    if (lev != nlev) # Not the coarsest level
        dxH  = fill(0.0,size(sysmats[lev+1],2))
        rH   = fill(0.0,size(sysmats[lev+1],2))
    else
        dxH, rH = nothing, nothing
    end
    return Adxh, dxh, dxH, rH
end

function get_pmg_caches(lev::Int, sysmats::Vector{T}, sh::FESpaceHierarchy) where T <: PSparseMatrix
    nlev = length(sysmats)
    Adxh = PVector(0.0,sysmats[lev].rows)
    dxh  = PVector(0.0,sysmats[lev].cols)
    if (lev != nlev) # Not the coarsest level
        dxH  = PVector(0.0,sysmats[lev+1].cols)
        rH   = PVector(0.0,sysmats[lev+1].rows)
    else
        dxH, rH = nothing, nothing
    end
    return Adxh, dxh, dxH, rH
end

function Gridap.Algebra.numerical_setup(ss::PMGSymbolicSetup, sysmats)
    pmg = ss.pmg
    nlev = get_num_levels(pmg.sh)

    # Caches
    caches = map(k -> get_pmg_caches(k,sysmats,pmg.sh), collect(1:nlev))

    # Transfer Operators
    transfers = get_transfer_operators(pmg.sh)

    # Smoother/Solvers setups
    ns_pre_smoothers = map((ss,mat) -> numerical_setup(ss,mat),ss.ss_pre_smoothers,sysmats[1:nlev-1])
    if pmg.post_smoother === pmg.pre_smoother
        ns_post_smoothers = ns_pre_smoothers
    else
        ns_post_smoothers = map((ss,mat) -> numerical_setup(ss,mat),ss.ss_post_smoothers,sysmats[1:nlev-1])
    end
    ns_coarse_solver = numerical_setup(ss.ss_coarse_solver,sysmats[nlev])

    return PMGNumericalSetup(pmg,sysmats,caches,transfers,ns_pre_smoothers,ns_post_smoothers,ns_coarse_solver)
end

function solve!(x::AbstractVector,ns::PMGNumericalSetup,b::AbstractVector)
    maxiter = ns.pmg.maxiter
    rtol    = ns.pmg.rtol
    verbose = ns.pmg.verbose
    mode    = ns.pmg.mode

    if mode == :preconditioner
        fill!(x,0.0)
        r  = copy(b)
    else
        A  = ns.sysmats[1]
        r  = similar(b); mul!(r,A,x); r .= b .- r
    end

    iter   = 0
    err    = 1.0
    nrm_r0 = norm(r)
    verbose && println("> PMG: Starting convergence loop.")
    while err > rtol && iter < maxiter
        solve!(1,x,ns,r)

        nrm_r = norm(r)
        err   = nrm_r/nrm_r0
        verbose && println("  > Iteration ", iter, ": (eAbs, eRel) = (", nrm_r, " , ", err, ")")
        iter = iter + 1
    end

    converged = (err < rtol)
    return iter, converged
end

function solve!(lev::Int,xh::AbstractVector,ns::PMGNumericalSetup,rh::AbstractVector)
    nlev = get_num_levels(ns.pmg.sh)

    ### Coarsest level
    if (lev == nlev)
        solve!(xh,ns.ns_coarse_solver,rh)
        return
    end

    ### Fine levels
    Ah = ns.sysmats[lev]
    Adxh, dxh, dxH, rH = ns.caches[lev]
    R, Rt = ns.transfers[lev]

    # Pre-smooth current solution
    solve!(xh, ns.ns_pre_smoothers[lev], rh)

    # Restrict the residual
    mul!(rH, R, rh)

    # Apply next level
    fill!(dxH,0.0)
    solve!(lev+1,dxH,ns,rH)

    # Interpolate dxH in finer space
    mul!(dxh, Rt, dxH)

    # Update solution & residual
    xh .= xh .+ dxh
    mul!(Adxh, Ah, dxh)
    rh .= rh .- Adxh

    # Post-smooth current solution
    solve!(xh, ns.ns_post_smoothers[lev], rh)
end

function LinearAlgebra.ldiv!(x::AbstractVector,ns::PMGNumericalSetup,b::AbstractVector)
    solve!(x,ns,b)
end
