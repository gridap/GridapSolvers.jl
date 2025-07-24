
module HPDDMTests

using Test
using Gridap
using GridapDistributed
using PartitionedArrays
using SparseMatricesCSR, SparseArrays
using GridapPETSc

function main(distribute,np)
  u(x) = x[1] + x[2]

  ranks = distribute(LinearIndices((prod(np),)))
  model = CartesianDiscreteModel(ranks,np,(0,1,0,1),(16,16))

  order = 1
  reffe = ReferenceFE(lagrangian,Float64,order)
  V = FESpace(model,reffe;dirichlet_tags="boundary")
  U = TrialFESpace(V,u)

  # Global assembled problem
  qdegree = 2*(order-1)
  Ω = Triangulation(model)
  dΩ = Measure(Ω,qdegree)

  f(x) = -Δ(u)(x)
  a(u,v) = ∫(∇(u)⋅∇(v))dΩ
  l(v) = ∫(f⋅v)dΩ

  assem = SparseMatrixAssembler(
    SparseMatrixCSR{0,PetscScalar,PetscInt},Vector{PetscScalar},U,V
  )
  op = AffineFEOperator(a,l,U,V,assem)

  # Overlapping Neumann problems require ghost cells in the measure
  Ωg = Triangulation(with_ghost,model)
  dΩg = Measure(Ωg,qdegree)
  a_g(u,v) = ∫(∇(u)⋅∇(v))dΩg

  options = "-ksp_error_if_not_converged true -ksp_converged_reason -ksp_monitor -pc_hpddm_levels_1_eps_nev 10 -pc_hpddm_levels_1_sub_pc_type cholesky -pc_hpddm_define_subdomains"
  GridapPETSc.with(args=split(options)) do
    solver = HPDDMLinearSolver(V,a_g)
    uh = solve(solver,op)

    eh = u - uh
    err_l2 = sqrt(sum(∫(eh⋅eh)dΩ))
    if i_am_main(ranks)
      @info "L2 error: $err_l2"
      @test err_l2 < 1e-6
    end
  end
end

end
