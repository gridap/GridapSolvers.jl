
"""
    mutable struct ContinuationSwitch{A}
      callback :: Function
      caches   :: A
      switched :: Bool
    end

    ContinuationSwitch(callback::Function, caches)

Switch that implements the switching logic for a ContinuationFEOperator.

The callback function must provide the following signatures:

- Reset: `callback(u::FEFunction,::Nothing,cache) -> (switch::Bool, new_cache)`
- Update: `callback(u::FEFunction,b::AbstractVector,cache) -> (switch::Bool, new_cache)`

where `u` is the current solution (as a FEFunction) and `b` is the current residual. The first 
signature is called by `allocate_residual` and is typically used to reset the switch for a new 
nonlinear solve. The second signature is called by `residual!` and is used to update the switch
based on the current residual.

The cache is (potentially) mutated between each call, and can hold any extra information 
needed for the continuation logic.
"""
mutable struct ContinuationSwitch{A}
  callback :: Function
  caches   :: A
  switched :: Bool

  function ContinuationSwitch(callback::Function, caches)
    A = typeof(caches)
    new{A}(callback, caches, false)
  end
end

has_switched(s::ContinuationSwitch) = s.switched

switch!(s::ContinuationSwitch, u) = switch!(s, u, nothing)

function switch!(s::ContinuationSwitch, u, b)
  has_switched(s) && return s.switched
  s.switched, s.caches = s.callback(u, b, s.caches)
  if has_switched(s)
    @debug "ContinuationFEOperator: Switching operators!"
  end
  return s.switched
end

"""
  ContinuationSwitch(niter::Int)

Switch that will change operators after `niter` iterations.
"""
function ContinuationSwitch(niter::Int)
  caches = (;it = -1, niter = niter)
  callback(u,::Nothing,c) = (false, (;it = -1, niter = c.niter))
  callback(u,b,c) = (c.it+1 >= c.niter, (;it = c.it+1, niter = c.niter))
  return ContinuationSwitch(callback, caches)
end

"""
    struct ContinuationFEOperator <: FESpaces.FEOperator
      op1    :: FEOperator
      op2    :: FEOperator
      switch :: ContinuationSwitch
      reuse_caches :: Bool
    end

FEOperator implementing the continuation method for nonlinear solvers. It switches between 
its two operators when the switch is triggered.

Continuation between more that two operators can be achieved by daisy-chaining two or more 
ContinuationFEOperators.

If `reuse_caches` is `true`, the Jacobian of the first operator is reused for the second 
operator. This is only possible if the sparsity pattern of the Jacobian does not change.
"""
struct ContinuationFEOperator{A,B,C} <: FESpaces.FEOperator
  op1 :: A
  op2 :: B
  switch :: C
  reuse_caches :: Bool

  function ContinuationFEOperator(
    op1::FEOperator,
    op2::FEOperator,
    switch::ContinuationSwitch;
    reuse_caches::Bool = true
  )
    A, B, C = typeof(op1), typeof(op2), typeof(switch)
    new{A,B,C}(op1, op2, switch, reuse_caches)
  end
end

has_switched(op::ContinuationFEOperator) = has_switched(op.switch)

"""
  ContinuationFEOperator(op1::FEOperator, op2::FEOperator, niter::Int; reuse_caches::Bool = true)

ContinuationFEOperator that switches between `op1` and `op2` after `niter` iterations.
"""
function ContinuationFEOperator(
  op1::FEOperator, op2::FEOperator, niter::Int; reuse_caches::Bool = true
)
  switch = ContinuationSwitch(niter)
  return ContinuationFEOperator(op1, op2, switch; reuse_caches)
end

# FEOperator API

function FESpaces.get_test(op::ContinuationFEOperator)
  ifelse(!has_switched(op), get_test(op.op1), get_test(op.op2))
end

function FESpaces.get_trial(op::ContinuationFEOperator)
  ifelse(!has_switched(op), get_trial(op.op1), get_trial(op.op2))
end

function Algebra.allocate_residual(op::ContinuationFEOperator,u)
  switch!(op.switch, u, nothing)
  ifelse(!has_switched(op), allocate_residual(op.op1, u), allocate_residual(op.op2, u))
end

function Algebra.residual!(b::AbstractVector,op::ContinuationFEOperator,u)
  switch!(op.switch, u, b)
  ifelse(!has_switched(op), residual!(b, op.op1, u), residual!(b, op.op2, u))
end

function Algebra.allocate_jacobian(op::ContinuationFEOperator,u)
  ifelse(!has_switched(op), allocate_jacobian(op.op1, u), allocate_jacobian(op.op2, u))
end

function Algebra.jacobian!(A::AbstractMatrix,op::ContinuationFEOperator,u)
  if op.reuse_caches
    ifelse(!has_switched(op), jacobian!(A, op.op1, u), jacobian!(A, op.op2, u))
  else
    ifelse(!has_switched(op), jacobian(op.op1, u), jacobian(op.op2, u))
  end
end
