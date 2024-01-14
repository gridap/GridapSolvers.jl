
"""
 Notes on this structure: 

 When converting julia vectors/PVectors to PETSc vectors, we purposely create aliasing 
 of the vector values. This means we can avoid copying data from one to another before solving, 
 but we need to be careful about it. 

 This structure takes care of this, and makes sure you do not attempt to solve the system 
 with julia vectors that are not the ones you used to create the solver cache.
"""
struct CachedPETScNS{TM,A}
  ns     :: GridapPETSc.PETScLinearSolverNS{TM}
  X      :: PETScVector
  B      :: PETScVector
  owners :: A
  function CachedPETScNS(ns::GridapPETSc.PETScLinearSolverNS{TM},x::AbstractVector,b::AbstractVector) where TM
    X = convert(PETScVector,x)
    B = convert(PETScVector,b)
    owners = (x,b)

    A = typeof(owners)
    new{TM,A}(ns,X,B,owners)
  end
end

function Algebra.solve!(x::AbstractVector,ns::CachedPETScNS,b::AbstractVector)
  @assert x === ns.owners[1]
  @assert b === ns.owners[2]
  solve!(ns.X,ns.ns,ns.B)
  consistent!(x)
  return x
end
