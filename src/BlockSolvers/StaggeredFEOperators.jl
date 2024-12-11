
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

struct StaggeredFESolver{NB} <: FESpaces.FESolver
  solvers :: Vector{<:NonlinearSolver}
  function StaggeredFESolver(solvers::Vector{<:NonlinearSolver})
    NB = length(solvers)
    new{NB}(solvers)
  end
end

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
