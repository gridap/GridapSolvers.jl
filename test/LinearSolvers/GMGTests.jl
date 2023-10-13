module GMGTests

using MPI
using Test
using LinearAlgebra
using IterativeSolvers
using FillArrays

using Gridap
using Gridap.ReferenceFEs
using PartitionedArrays
using GridapDistributed
using GridapP4est

using GridapSolvers
using GridapSolvers.LinearSolvers
using GridapSolvers.MultilevelTools
using GridapSolvers.PatchBasedSmoothers


function get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)
  mh = tests.mh
  nlevs = num_levels(mh)
  smoothers = Vector{RichardsonSmoother}(undef,nlevs-1)
  for lev in 1:nlevs-1
    parts = get_level_parts(mh,lev)
    if i_am_in(parts)
      PD = patch_decompositions[lev]
      Ph = get_fe_space(patch_spaces,lev)
      Vh = get_fe_space(tests,lev)
      Ω  = Triangulation(PD)
      dΩ = Measure(Ω,qdegree)
      a(u,v) = biform(u,v,dΩ)
      local_solver = LUSolver() # IS_ConjugateGradientSolver(;reltol=1.e-6)
      patch_smoother = PatchBasedLinearSolver(a,Ph,Vh,local_solver)
      smoothers[lev] = RichardsonSmoother(patch_smoother,10,0.2)
    end
  end
  return smoothers
end

function gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u,restriction_method)
  tests, trials = spaces

  tic!(t;barrier=true)
  # Integration
  smatrices, A, b = compute_hierarchy_matrices(trials,biform,liform,qdegree)

  # Preconditioner
  coarse_solver = LUSolver()
  restrictions, prolongations = setup_transfer_operators(trials,qdegree;mode=:residual,restriction_method=restriction_method)
  gmg = GMGLinearSolver(mh,
                        smatrices,
                        prolongations,
                        restrictions,
                        pre_smoothers=smoothers,
                        post_smoothers=smoothers,
                        coarsest_solver=coarse_solver,
                        maxiter=1,
                        rtol=1.0e-8,
                        verbose=false,
                        mode=:preconditioner)
  ss = symbolic_setup(gmg,A)
  ns = numerical_setup(ss,A)
  toc!(t,"GMG setup")

  # Solve
  tic!(t;barrier=true)
  x = pfill(0.0,partition(axes(A,2)))
  x, history = IterativeSolvers.cg!(x,A,b;
                                    verbose=i_am_main(parts),
                                    reltol=1.0e-8,
                                    Pl=ns,
                                    log=true)
  toc!(t,"Solver")

  # Error
  model = get_model(mh,1)
  Uh    = get_fe_space(trials,1)
  Ω     = Triangulation(model)
  dΩ    = Measure(Ω,qdegree)
  uh    = FEFunction(Uh,x)
  eh    = u-uh
  e_l2  = sum(∫(eh⋅eh)dΩ)
  if i_am_main(parts)
    println("L2 error = ", e_l2)
  end
  return e_l2
end

function gmg_poisson_driver(t,parts,mh,order,restriction_method)
  tic!(t;barrier=true)
  u(x) = x[1] + x[2]
  f(x) = -Δ(u)(x)
  biform(u,v,dΩ) = ∫(∇(v)⋅∇(u))dΩ
  liform(v,dΩ)   = ∫(v*f)dΩ
  qdegree   = 2*order+1
  reffe     = ReferenceFE(lagrangian,Float64,order)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,9.0/8.0),num_levels(mh)-1)

  tests     = TestFESpace(mh,reffe,dirichlet_tags="boundary")
  trials    = TrialFESpace(tests,u)
  spaces    = tests, trials
  toc!(t,"FESpaces")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u,restriction_method)
end

function gmg_laplace_driver(t,parts,mh,order,restriction_method)
  tic!(t;barrier=true)
  α    = 1.0
  u(x) = x[1] + x[2]
  f(x) = -Δ(u)(x)
  biform(u,v,dΩ) = ∫(v*u)dΩ + ∫(α*∇(v)⋅∇(u))dΩ
  liform(v,dΩ)   = ∫(v*f)dΩ
  qdegree   = 2*order+1
  reffe     = ReferenceFE(lagrangian,Float64,order)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),num_levels(mh)-1)

  tests     = TestFESpace(mh,reffe,dirichlet_tags="boundary")
  trials    = TrialFESpace(tests,u)
  spaces    = tests, trials
  toc!(t,"FESpaces")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u,restriction_method)
end

function gmg_vector_laplace_driver(t,parts,mh,order,restriction_method)
  tic!(t;barrier=true)
  α    = 1.0
  u(x) = VectorValue(x[1],x[2])
  f(x) = VectorValue(2.0*x[2]*(1.0-x[1]*x[1]),2.0*x[1]*(1-x[2]*x[2]))
  biform(u,v,dΩ) = ∫(v⋅u)dΩ + ∫(α*∇(v)⊙∇(u))dΩ
  liform(v,dΩ)   = ∫(v⋅f)dΩ
  qdegree   = 2*order+1
  reffe     = ReferenceFE(lagrangian,VectorValue{2,Float64},order)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),num_levels(mh)-1)

  tests     = TestFESpace(mh,reffe,dirichlet_tags="boundary")
  trials    = TrialFESpace(tests,u)
  spaces    = tests, trials
  toc!(t,"FESpaces")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u,restriction_method)
end

function gmg_hdiv_driver(t,parts,mh,order,restriction_method)
  tic!(t;barrier=true)
  α     = 1.0
  u(x)  = VectorValue(x[1],x[2])
  f(x)  = VectorValue(2.0*x[2]*(1.0-x[1]*x[1]),2.0*x[1]*(1-x[2]*x[2]))
  biform(u,v,dΩ)  = ∫(v⋅u)dΩ + ∫(α*divergence(v)⋅divergence(u))dΩ
  liform(v,dΩ)    = ∫(v⋅f)dΩ
  qdegree   = 2*(order+1)
  reffe     = ReferenceFE(raviart_thomas,Float64,order)

  tests     = TestFESpace(mh,reffe,dirichlet_tags="boundary")
  trials    = TrialFESpace(tests,u)
  spaces    = tests, trials
  toc!(t,"FESpaces")

  tic!(t;barrier=true)
  patch_decompositions = PatchDecomposition(mh)
  patch_spaces = PatchFESpace(mh,reffe,DivConformity(),patch_decompositions,tests)
  smoothers = get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)
  toc!(t,"Patch Decomposition")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u,restriction_method)
end

function main_gmg_driver(parts,mh,order,restriction_method,pde)
  t = PTimer(parts,verbose=true)
  if     pde == :poisson
    gmg_poisson_driver(t,parts,mh,order,restriction_method)
  elseif pde == :laplace
    gmg_laplace_driver(t,parts,mh,order,restriction_method)
  elseif pde == :vector_laplace
    gmg_vector_laplace_driver(t,parts,mh,order,restriction_method)
  elseif pde == :hdiv
    gmg_hdiv_driver(t,parts,mh,order,restriction_method)
  end
end

function get_mesh_hierarchy(parts,Dc,np_per_level,num_refs_coarse)
  if Dc == 2
    domain = (0,1,0,1)
    nc = (2,2)
  else
    @assert Dc == 3
    domain = (0,1,0,1,0,1)
    nc = (2,2,2)
  end
  
  num_levels   = length(np_per_level)
  cparts       = generate_subparts(parts,np_per_level[num_levels])
  cmodel       = CartesianDiscreteModel(domain,nc)
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  return mh
end

function main(distribute,np,Dc,np_per_level)
  parts = distribute(LinearIndices((prod(np),)))

  num_refs_coarse = 2
  mh = get_mesh_hierarchy(parts,Dc,np_per_level,num_refs_coarse)

  for pde in [:poisson,:laplace,:vector_laplace,:hdiv]
    methods = (pde !== :hdiv) ? [:projection,:interpolation] : [:projection]
    for restriction_method in methods
      if i_am_main(parts)
        println(repeat("=",80))
        println("Testing GMG with Dc=$Dc, PDE=$pde and restriction_method=$restriction_method")
      end
      order = (pde !== :hdiv) ? 1 : 0
      main_gmg_driver(parts,mh,order,restriction_method,pde)
    end
  end
end

with_mpi() do distribute
  main(distribute,4,2,[4,2,1])
end

end # module GMGTests