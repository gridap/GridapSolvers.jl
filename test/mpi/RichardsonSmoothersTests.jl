module RichardsonSmoothersTests

using Test
using MPI
using Gridap
using GridapDistributed
using PartitionedArrays
using IterativeSolvers
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers

function main(parts,nranks,domain_partition)
  GridapP4est.with(parts) do
    domain = (0,1,0,1)
    model  = CartesianDiscreteModel(parts,nranks,domain,domain_partition)

    sol(x) = x[1] + x[2]
    f(x)   = -Δ(sol)(x)

    order  = 1
    qorder = order*2 + 1
    reffe  = ReferenceFE(lagrangian,Float64,order)
    Vh     = TestFESpace(model,reffe;conformity=:H1,dirichlet_tags="boundary")
    Uh     = TrialFESpace(Vh,sol)
    u      = interpolate(sol,Uh)

    Ω      = Triangulation(model)
    dΩ     = Measure(Ω,qorder)
    a(u,v) = ∫(∇(v)⋅∇(u))*dΩ
    l(v)   = ∫(v⋅f)*dΩ

    op = AffineFEOperator(a,l,Uh,Vh)
    A, b = get_matrix(op), get_vector(op)

    P  = RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0)
    ss = symbolic_setup(P,A)
    ns = numerical_setup(ss,A)

    x = pfill(1.0,partition(axes(A,2)))
    x, history = IterativeSolvers.cg!(x,A,b;
                                      verbose=i_am_main(parts),
                                      reltol=1.0e-8,
                                      Pl=ns,
                                      log=true)

    u  = interpolate(sol,Uh)
    uh = FEFunction(Uh,x)
    eh = uh - u
    E  = sum(∫(eh*eh)*dΩ)
    if i_am_main(parts)
      println("L2 Error: ", E)
    end
    
    @test E < 1.e-8
  end
end

domain_partition = (32,32)
num_ranks = (2,2)
parts = with_mpi() do distribute
  distribute(LinearIndices((prod(num_ranks),)))
end
main(parts,num_ranks,domain_partition)
MPI.Finalize()

end