module IterativeSolversTests

using Test
using Gridap
using IterativeSolvers
using LinearAlgebra
using SparseArrays

using GridapSolvers

A = SparseMatrixCSC(Matrix(1.0I,3,3))

# CG
solver = ConjugateGradientSolver(;maxiter=100,reltol=1.e-12)
ss = symbolic_setup(solver,A)
ns = numerical_setup(ss,A)

x = zeros(3)
y = [1.0,2.0,3.0]
solve!(x,ns,y)
@test x ≈ y

# GMRES
solver = GMRESSolver(;maxiter=100,reltol=1.e-12)
ss = symbolic_setup(solver,A)
ns = numerical_setup(ss,A)

x = zeros(3)
y = [1.0,2.0,3.0]
solve!(x,ns,y)
@test x ≈ y

# MINRES
solver = MINRESSolver(;maxiter=100,reltol=1.e-12)
ss = symbolic_setup(solver,A)
ns = numerical_setup(ss,A)

x = zeros(3)
y = [1.0,2.0,3.0]
solve!(x,ns,y)
@test x ≈ y

# SSOR
solver = SSORSolver(2.0/3.0;maxiter=100)
ss = symbolic_setup(solver,A)
ns = numerical_setup(ss,A)

x = zeros(3)
y = [1.0,2.0,3.0]
solve!(x,ns,y)
@test x ≈ y

end