
"""
    abstract type StaggeredFEOperator{NB,SB} <: FESpaces.FEOperator end

Staggered operator, used to solve staggered problems. 

We define a staggered problem as a multi-variable non-linear problem where the equation 
for the k-th variable `u_k` only depends on the previous variables `u_1,...,u_{k-1}` (and itself).

Such a problem can then be solved by solving each variable sequentially, 
using the previous variables as input. The most common examples of staggered problems 
are one-directional coupling problems, where the variables are coupled in a chain-like manner.

Two types of staggered operators are currently supported: 

- [`StaggeredAffineFEOperator`](@ref): when the k-th equation is linear in `u_k`.
- [`StaggeredNonlinearFEOperator`](@ref): when the k-th equation is non-linear in `u_k`.

"""
abstract type StaggeredFEOperator{NB,SB} <: FESpaces.FEOperator end

function MultiField.get_block_ranges(::StaggeredFEOperator{NB,SB}) where {NB,SB}
  MultiField.get_block_ranges(NB,SB,Tuple(1:sum(SB)))
end

function get_solution(op::StaggeredFEOperator{NB,SB}, xh::MultiFieldFEFunction, k) where {NB,SB}
  r = MultiField.get_block_ranges(op)[k]
  if isone(length(r)) # SingleField
    xh_k = xh[r[1]]
  else # MultiField
    fv_k = blocks(get_free_dof_values(xh))[k]
    xh_k = MultiFieldFEFunction(fv_k, op.trials[k], xh.single_fe_functions[r])
  end
  return xh_k
end

function get_solution(op::StaggeredFEOperator{NB,SB}, xh::DistributedMultiFieldFEFunction, k) where {NB,SB}
  r = MultiField.get_block_ranges(op)[k]
  if isone(length(r)) # SingleField
    xh_k = xh[r[1]]
  else # MultiField
    sf_k = xh.field_fe_fun[r]
    fv_k = blocks(get_free_dof_values(xh))[k]
    mf_k = map(local_views(op.trials[k]),partition(fv_k),map(local_views,sf_k)...) do Vk, fv_k, sf_k...
      MultiFieldFEFunction(fv_k, Vk, [sf_k...])
    end
    xh_k = DistributedMultiFieldFEFunction(sf_k, mf_k, fv_k)
  end
  return xh_k
end

# StaggeredFESolver

"""
    struct StaggeredFESolver{NB} <: FESpaces.FESolver
      solvers :: Vector{<:Union{LinearSolver,NonlinearSolver}}
    end

Solver for staggered problems. See [`StaggeredFEOperator`](@ref) for more details.
"""
struct StaggeredFESolver{NB} <: FESpaces.FESolver
  solvers :: Vector{<:NonlinearSolver}
  function StaggeredFESolver(solvers::Vector{<:NonlinearSolver})
    NB = length(solvers)
    new{NB}(solvers)
  end
end

"""
    solve(solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB})
    solve!(xh, solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB}, cache::Nothing) where NB
    solve!(xh, solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB}, cache) where NB
"""
function Algebra.solve!(xh, solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB}, ::Nothing) where NB
  solvers = solver.solvers
  xhs, caches, operators = (), (), ()
  for k in 1:NB
    xh_k = get_solution(op,xh,k)
    op_k = get_operator(op,xhs,k)
    xh_k, cache_k = solve!(xh_k,solvers[k],op_k,nothing)
    copy!(get_free_dof_values(get_solution(op,xh,k)),get_free_dof_values(xh_k))
    xhs, caches, operators = (xhs...,xh_k), (caches...,cache_k), (operators...,op_k)
  end
  return xh, (caches,operators)
end

function Algebra.solve!(xh, solver::StaggeredFESolver{NB}, op::StaggeredFEOperator{NB}, cache) where NB
  last_caches, last_operators = cache
  solvers = solver.solvers
  xhs, caches, operators = (), (), ()
  for k in 1:NB
    xh_k = get_solution(op,xh,k)
    op_k = get_operator!(last_operators[k],op,xhs,k)
    xh_k, cache_k = solve!(xh_k,solvers[k],op_k,last_caches[k])
    xhs, caches, operators = (xhs...,xh_k), (caches...,cache_k), (operators...,op_k)
  end
  return xh, (caches,operators)
end

# StaggeredAffineFEOperator

"""
    struct StaggeredAffineFEOperator{NB,SB} <: StaggeredFEOperator{NB,SB}
      ...
    end

Affine staggered operator, used to solve staggered problems 
where the k-th equation is linear in `u_k`.

Such a problem is formulated by a set of bilinear/linear form pairs:

    a_k((u_1,...,u_{k-1}),u_k,v_k) = ∫(...)
    l_k((u_1,...,u_{k-1}),v_k) = ∫(...)

than cam be assembled into a set of linear systems: 

    A_k u_k = b_k

where `A_k` and `b_k` only depend on the previous variables `u_1,...,u_{k-1}`.
"""
struct StaggeredAffineFEOperator{NB,SB} <: StaggeredFEOperator{NB,SB}
  biforms :: Vector{<:Function}
  liforms :: Vector{<:Function}
  trials  :: Vector{<:FESpace}
  tests   :: Vector{<:FESpace}
  assems  :: Vector{<:Assembler}
  trial   :: BlockFESpaceTypes{NB,SB}
  test    :: BlockFESpaceTypes{NB,SB}

  function StaggeredAffineFEOperator(
    biforms::Vector{<:Function},
    liforms::Vector{<:Function},
    trials::Vector{<:FESpace},
    tests::Vector{<:FESpace},
    assems::Vector{<:Assembler}
  )
    @assert length(biforms) == length(liforms) == length(trials) == length(tests) == length(assems)
    trial = combine_fespaces(trials)
    test  = combine_fespaces(tests)
    NB, SB = length(trials), Tuple(map(num_fields,trials))
    new{NB,SB}(biforms,liforms,trials,tests,assems,trial,test)
  end
end

function StaggeredAffineFEOperator(
  biforms::Vector{<:Function},
  liforms::Vector{<:Function},
  trials::Vector{<:FESpace},
  tests::Vector{<:FESpace}
)
  assems = map(SparseMatrixAssembler,tests,trials)
  return StaggeredAffineFEOperator(biforms,liforms,trials,tests,assems)
end

FESpaces.get_trial(op::StaggeredAffineFEOperator) = op.trial
FESpaces.get_test(op::StaggeredAffineFEOperator) = op.test

# Utils

MultiField.num_fields(space::FESpace) = 1

# TODO: We could reuse gids in distributed
function combine_fespaces(spaces::Vector{<:FESpace})
  NB = length(spaces)
  SB = Tuple(map(num_fields,spaces))

  sf_spaces = vcat(map(split_fespace,spaces)...)
  MultiFieldFESpace(sf_spaces; style = BlockMultiFieldStyle(NB,SB))
end

split_fespace(space::FESpace) = [space]
split_fespace(space::MultiFieldFESpaceTypes) = [space...]

function get_operator(op::StaggeredFEOperator{NB}, xhs, k) where NB
  @assert NB >= k
  a(uk,vk) = op.biforms[k](xhs,uk,vk)
  l(vk) = op.liforms[k](xhs,vk)
  # uhd = zero(op.trials[k])
  # A, b = assemble_matrix_and_vector(a,l,op.assems[k],op.trials[k],op.tests[k],uhd)
  # return AffineOperator(A,b)
  return AffineFEOperator(a,l,op.trials[k],op.tests[k],op.assems[k])
end

function get_operator!(op_k::AffineFEOperator, op::StaggeredFEOperator{NB}, xhs, k) where NB
  @assert NB >= k
  A, b = get_matrix(op_k), get_vector(op_k)
  a(uk,vk) = op.biforms[k](xhs,uk,vk)
  l(vk) = op.liforms[k](xhs,vk)
  uhd = zero(op.trials[k])
  assemble_matrix_and_vector!(a,l,A,b,op.assems[k],op.trials[k],op.tests[k],uhd)
  return AffineOperator(A,b)
end

############################################################################################
# StaggeredFEOperator

"""
    struct StaggeredNonlinearFEOperator{NB,SB} <: StaggeredFEOperator{NB,SB}
      ...
    end

Nonlinear staggered operator, used to solve staggered problems 
where the k-th equation is nonlinear in `u_k`.

Such a problem is formulated by a set of residual/jacobian pairs:

    jac_k((u_1,...,u_{k-1}),u_k,du_k,dv_k) = ∫(...)
    res_k((u_1,...,u_{k-1}),u_k,v_k) = ∫(...)

"""
struct StaggeredNonlinearFEOperator{NB,SB} <: StaggeredFEOperator{NB,SB}
  jacobians :: Vector{<:Function}
  residuals :: Vector{<:Function}
  trials    :: Vector{<:FESpace}
  tests     :: Vector{<:FESpace}
  assems    :: Vector{<:Assembler}
  trial     :: BlockFESpaceTypes{NB,SB}
  test      :: BlockFESpaceTypes{NB,SB}

  function StaggeredNonlinearFEOperator(
    res::Vector{<:Function},
    jac::Vector{<:Function},
    trials::Vector{<:FESpace},
    tests::Vector{<:FESpace},
    assems::Vector{<:Assembler}
  )
    @assert length(res) == length(jac) == length(trials) == length(tests) == length(assems)
    trial = combine_fespaces(trials)
    test  = combine_fespaces(tests)
    NB, SB = length(trials), Tuple(map(num_fields,trials))
    new{NB,SB}(res,jac,trials,tests,assems,trial,test)
  end
end

# TODO: Can be compute jacobians from residuals? 
function StaggeredNonlinearFEOperator(
  res::Vector{<:Function},
  jac::Vector{<:Function},
  trials::Vector{<:FESpace},
  tests::Vector{<:FESpace}
)
  assems = map(SparseMatrixAssembler,tests,trials)
  return StaggeredNonlinearFEOperator(res,jac,trials,tests,assems)
end

FESpaces.get_trial(op::StaggeredNonlinearFEOperator) = op.trial
FESpaces.get_test(op::StaggeredNonlinearFEOperator) = op.test
