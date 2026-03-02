
"""
"""
mutable struct BlockJacobiProlongationOperator{A,B,C}
  sh     :: A
  solver :: B
  op     :: C
  caches
end

"""
"""
function BlockJacobiProlongationOperator(lev,sh,ptopo)
  Vh = MultilevelTools.get_fe_space(sh,lev)
  solver = BlockJacobiSolver(Vh,ptopo;assembly=:interior)
  BlockJacobiProlongationOperator(lev,sh,solver)
end

function BlockJacobiProlongationOperator(lev,sh,solver::BlockJacobiSolver)
  op = ProlongationOperator(lev,sh,0;mode=:residual)
  return BlockJacobiProlongationOperator(sh,solver,op,nothing)
end

function MultilevelTools.update_transfer_operator!(op::BlockJacobiProlongationOperator,A)
  if isnothing(op.caches)
    op.caches = numerical_setup(symbolic_setup(op.solver,A),A)
  else
    numerical_setup!(op.caches,A)
  end
end

function LinearAlgebra.mul!(y::AbstractVector,op::BlockJacobiProlongationOperator,x::AbstractVector)
  ns = op.caches
  mat, patch_rows, patch_cols, cache = ns.matrix, ns.patch_rows, ns.patch_cols, ns.cache
  
  mul!(y,op.op,x) # Regular prolongation
  z = op.op.cache[1][3] # Awful 
  fill!(z, zero(eltype(z)))
  solve_block_jacobi_projection!(z, mat, y, patch_rows, patch_cols, eachindex(patch_rows), cache)
  y .-= z
  return y
end

function LinearAlgebra.mul!(y::PVector,op::BlockJacobiProlongationOperator,x::Union{Nothing,PVector})
  ns = op.caches
  mat, patch_rows, patch_cols, caches = ns.matrix, ns.patch_rows, ns.patch_cols, ns.cache
  x_c, b_c, cache = caches

  mul!(b_c,op.op,x) # Regular prolongation
  consistent!(b_c) |> wait

  map(partition(x_c), partition(mat), partition(b_c), patch_rows, patch_cols, cache) do x_c, mat, b_c, patch_rows, patch_cols, cache
    fill!(x_c, zero(eltype(x)))
    solve_block_jacobi_projection!(x_c, mat, b_c, patch_rows, patch_cols, eachindex(patch_cols), cache)
  end
  assemble!(x_c) |> wait
  y .= b_c .- x_c
  consistent!(y) |> wait
  return y
end

function setup_block_jacobi_prolongation_operators(sh::FESpaceHierarchy)
  map(view(linear_indices(sh),1:num_levels(sh)-1)) do lev
    mhl = sh[lev].mh_level
    ptopo = CoarsePatchTopology(mhl)
    BlockJacobiProlongationOperator(lev,sh,ptopo)
  end
end
