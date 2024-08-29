
struct VankaSolver{A} <: Algebra.LinearSolver
  patch_ids::A
  function VankaSolver(
    patch_ids::Union{T,AbstractVector{<:T}} # Serial / Distributed
  ) where T <: AbstractVector{<:AbstractVector{<:Integer}}
    A = typeof(patch_ids)
    new{A}(patch_ids)
  end
end

function VankaSolver(space::MultiFieldFESpace)
  trian = get_triangulation(space)
  ncells = num_cells(trian)
  patch_cells = Table(1:ncells,1:ncells+1)
  return VankaSolver(space,patch_cells)
end

function VankaSolver(space::MultiFieldFESpace,patch_decomposition::PatchDecomposition)
  patch_cells = patch_decomposition.patch_cells
  return VankaSolver(space,patch_cells)
end

function VankaSolver(space::MultiFieldFESpace,patch_cells::Table{<:Integer})
  cell_ids = get_cell_dof_ids(space)
  patch_ids = map(patch_cells) do cells
    ids = vcat([vcat(cell_ids[cell].array...) for cell in cells]...)
    filter!(x->x>0,ids)
    sort!(ids)
    unique!(ids)
    return ids
  end
  return VankaSolver(patch_ids)
end

function VankaSolver(space::GridapDistributed.DistributedMultiFieldFESpace)
  patch_ids = map(local_views(space)) do space 
    solver = VankaSolver(space)
    return solver.patch_ids
  end
  return VankaSolver(patch_ids)
end

function VankaSolver(space::GridapDistributed.DistributedMultiFieldFESpace,patch_decomposition::DistributedPatchDecomposition)
  patch_ids = map(local_views(space),local_views(patch_decomposition)) do space, patch_decomposition
    solver = VankaSolver(space,patch_decomposition)
    return solver.patch_ids
  end
  return VankaSolver(patch_ids)
end

struct VankaSS{A} <: Algebra.SymbolicSetup
  solver::VankaSolver{A}
end

Algebra.symbolic_setup(s::VankaSolver,mat::AbstractMatrix) = VankaSS(s)

struct VankaNS{A,B,C} <: Algebra.NumericalSetup
  solver::VankaSolver{A}
  matrix::B
  cache ::C
end

function Algebra.numerical_setup(ss::VankaSS,mat::AbstractMatrix)
  T = eltype(mat)
  cache = (CachedArray(zeros(T,1)), CachedArray(zeros(T,1,1)))
  return VankaNS(ss.solver,mat,cache)
end

function Algebra.solve!(x::AbstractVector,ns::VankaNS,b::AbstractVector)
  A, patch_ids = ns.matrix, ns.solver.patch_ids
  c_xk, c_Ak = ns.cache
  fill!(x,0.0)

  n_patches = length(patch_ids)
  for patch in 1:n_patches
    ids = patch_ids[patch]

    n = length(ids)
    setsize!(c_xk,(n,))
    setsize!(c_Ak,(n,n))
    Ak = c_Ak.array
    bk = c_xk.array
    
    copyto!(Ak,view(A,ids,ids))
    copyto!(bk,view(b,ids))
    f = lu!(Ak,NoPivot();check=false)
    @check issuccess(f) "Factorization failed"
    ldiv!(f,bk)
    
    x[ids] .+= bk
  end

  return x
end

struct DistributedVankaNS{A,B} <: Algebra.NumericalSetup
  solver::VankaSolver{A}
  ns::B
end

function Algebra.numerical_setup(ss::VankaSS,mat::PSparseMatrix)
  ns = map(partition(mat)) do mat
    numerical_setup(ss,mat)
  end
  return DistributedVankaNS(ss.solver,ns)
end

function Algebra.solve!(x::PVector,ns::DistributedVankaNS,b::PVector)
  map(solve!,partition(x),ns.ns,partition(b))
  assemble!(x) |> wait
  consistent!(x) |> wait
  return x
end
