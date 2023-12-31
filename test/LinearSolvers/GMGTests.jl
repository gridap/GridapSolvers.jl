module GMGTests

using MPI
using Test
using LinearAlgebra
using IterativeSolvers
using FillArrays

using Gridap
using Gridap.ReferenceFEs, Gridap.Algebra
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
      local_solver = LUSolver()
      patch_smoother = PatchBasedLinearSolver(biform,Ph,Vh,dΩ,local_solver)
      smoothers[lev] = RichardsonSmoother(patch_smoother,10,0.2)
    end
  end
  return smoothers
end

function gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u)
  tests, trials = spaces

  tic!(t;barrier=true)
  # Integration
  smatrices, A, b = compute_hierarchy_matrices(trials,tests,biform,liform,qdegree)

  # Preconditioner
  coarse_solver = LUSolver()
  restrictions, prolongations = setup_transfer_operators(trials,
                                                         qdegree;
                                                         mode=:residual,
                                                         solver=IS_ConjugateGradientSolver(;reltol=1.e-6))
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
  
  solver = CGSolver(gmg;maxiter=20,atol=1e-14,rtol=1.e-6,verbose=i_am_main(parts))
  ns = numerical_setup(symbolic_setup(solver,A),A)
  toc!(t,"GMG setup")

  # Solve
  tic!(t;barrier=true)
  x = pfill(0.0,partition(axes(A,2)))
  solve!(x,ns,b)
  toc!(t,"Solver")

  # Error
  if !isa(u,Nothing)
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
  end
end

function gmg_poisson_driver(t,parts,mh,order)
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

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u)
end

function gmg_laplace_driver(t,parts,mh,order)
  tic!(t;barrier=true)
  α    = 1.0
  u(x) = x[1] + x[2]
  f(x) = u(x) - α * Δ(u)(x)
  biform(u,v,dΩ) = ∫(v*u)dΩ + ∫(α*∇(v)⋅∇(u))dΩ
  liform(v,dΩ)   = ∫(v*f)dΩ
  qdegree   = 2*order+1
  reffe     = ReferenceFE(lagrangian,Float64,order)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),num_levels(mh)-1)

  tests     = TestFESpace(mh,reffe,dirichlet_tags="boundary")
  trials    = TrialFESpace(tests,u)
  spaces    = tests, trials
  toc!(t,"FESpaces")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u)
end

function gmg_vector_laplace_driver(t,parts,mh,order)
  tic!(t;barrier=true)
  Dc   = num_cell_dims(get_model(mh,1))
  α    = 1.0
  u(x) = (Dc==2) ? VectorValue(x[1],x[2]) : VectorValue(x[1],x[2],x[3])
  f(x) = (Dc==2) ? VectorValue(x[1],x[2]) : VectorValue(x[1],x[2],x[3])
  biform(u,v,dΩ) = ∫(v⋅u)dΩ + ∫(α*∇(v)⊙∇(u))dΩ
  liform(v,dΩ)   = ∫(v⋅f)dΩ
  qdegree   = 2*order+1
  reffe     = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
  smoothers = Fill(RichardsonSmoother(JacobiLinearSolver(),10,2.0/3.0),num_levels(mh)-1)

  tests     = TestFESpace(mh,reffe,dirichlet_tags="boundary")
  trials    = TrialFESpace(tests,u)
  spaces    = tests, trials
  toc!(t,"FESpaces")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u)
end

function gmg_hdiv_driver(t,parts,mh,order)
  tic!(t;barrier=true)
  Dc   = num_cell_dims(get_model(mh,1))
  α    = 1.0
  u(x) = (Dc==2) ? VectorValue(x[1],x[2]) : VectorValue(x[1],x[2],x[3])
  f(x) = (Dc==2) ? VectorValue(x[1],x[2]) : VectorValue(x[1],x[2],x[3])
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
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  smoothers = get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)
  toc!(t,"Patch Decomposition")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,u)
end

function gmg_multifield_driver(t,parts,mh,order)
  tic!(t;barrier=true)
  Dc = num_cell_dims(get_model(mh,1))
  @assert Dc == 3

  β = 1.0
  γ = 1.0
  B = VectorValue(0.0,0.0,1.0)
  f = VectorValue(fill(1.0,Dc)...)
  biform((u,j),(v_u,v_j),dΩ) = ∫(β*∇(u)⊙∇(v_u) -γ*(j×B)⋅v_u + j⋅v_j - (u×B)⋅v_j)dΩ
  liform((v_u,v_j),dΩ) = ∫(v_u⋅f)dΩ

  reffe_u  = ReferenceFE(lagrangian,VectorValue{Dc,Float64},order)
  tests_u  = FESpace(mh,reffe_u;dirichlet_tags="boundary");
  trials_u = TrialFESpace(tests_u);

  reffe_j  = ReferenceFE(raviart_thomas,Float64,order-1)
  tests_j  = FESpace(mh,reffe_j;dirichlet_tags="boundary");
  trials_j = TrialFESpace(tests_j);

  trials = MultiFieldFESpace([trials_u,trials_j]);
  tests  = MultiFieldFESpace([tests_u,tests_j]);
  spaces = tests, trials
  toc!(t,"FESpaces")

  tic!(t;barrier=true)
  qdegree = 2*(order+1)
  patch_decompositions = PatchDecomposition(mh)
  patch_spaces = PatchFESpace(tests,patch_decompositions)
  smoothers = get_patch_smoothers(tests,patch_spaces,patch_decompositions,biform,qdegree)
  toc!(t,"Patch Decomposition")

  return gmg_driver(t,parts,mh,spaces,qdegree,smoothers,biform,liform,nothing)
end

function main_gmg_driver(parts,mh,order,pde)
  t = PTimer(parts,verbose=true)
  if     pde == :poisson
    gmg_poisson_driver(t,parts,mh,order)
  elseif pde == :laplace
    gmg_laplace_driver(t,parts,mh,order)
  elseif pde == :vector_laplace
    gmg_vector_laplace_driver(t,parts,mh,order)
  elseif pde == :hdiv
    gmg_hdiv_driver(t,parts,mh,order)
  elseif pde == :multifield
    gmg_multifield_driver(t,parts,mh,order)
  else
    error("Unknown PDE")
  end
end

function get_mesh_hierarchy(parts,nc,np_per_level)
  Dc = length(nc)
  domain = (Dc == 2) ? (0,1,0,1) : (0,1,0,1,0,1)
  num_refs_coarse = (Dc == 2) ? 1 : 0
  
  num_levels   = length(np_per_level)
  cparts       = generate_subparts(parts,np_per_level[num_levels])
  cmodel       = CartesianDiscreteModel(domain,nc)
  coarse_model = OctreeDistributedDiscreteModel(cparts,cmodel,num_refs_coarse)
  mh = ModelHierarchy(parts,coarse_model,np_per_level)
  return mh
end

function main(distribute,np::Integer,nc::Tuple,np_per_level::Vector)
  parts = distribute(LinearIndices((np,)))
  mh = get_mesh_hierarchy(parts,nc,np_per_level)
  Dc = length(nc)

  for pde in [:poisson,:laplace,:vector_laplace,:hdiv,:multifield]
    if (pde != :multifield) || (Dc == 3)
      if i_am_main(parts)
        println(repeat("=",80))
        println("Testing GMG with Dc=$(length(nc)), PDE=$pde")
      end
      order = (pde !== :hdiv) ? 1 : 0
      main_gmg_driver(parts,mh,order,pde)
    end
  end
end

end # module GMGTests