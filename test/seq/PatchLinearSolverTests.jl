module PatchLinearSolverTests
  using Gridap
  using Gridap.Geometry
  using Gridap.FESpaces
  using Gridap.ReferenceFEs
  using FillArrays
  using PartitionedArrays
  using Test

  using GridapSolvers
  using GridapSolvers.PatchBasedSmoothers

  order=1

  function returns_PD_Ph_xh_Vh(model;style=GridapSolvers.PatchBasedSmoothers.PatchBoundaryExclude())
    reffe = ReferenceFE(lagrangian,Float64,order)
    # reffe=ReferenceFE(lagrangian,VectorValue{2,Float64},order) @santiagobadia: For Vector Laplacian
    Vh = TestFESpace(model,reffe)
    PD = PatchDecomposition(model;patch_boundary_style=style)
    Ph = PatchFESpace(model,reffe,H1Conformity(),PD,Vh)
    assembler = SparseMatrixAssembler(Ph,Ph)
    Ωₚ  = Triangulation(PD)
    dΩₚ = Measure(Ωₚ,2*order+1)
    a(u,v) = ∫(∇(v)⋅∇(u))*dΩₚ
    l(v) = ∫(1*v)*dΩₚ
    # α =1,0; a(u,v)=∫(v⋅u)dΩ+∫(α*∇(v)⊙∇(u))dΩ # @santiagobadia: For vector Laplacian
    # f(x) = VectorValue(1.0,0.0)
    # l(v)=∫(v⋅f)dΩ
    Ah = assemble_matrix(a,assembler,Ph,Ph)
    fh = assemble_vector(l,assembler,Ph)
    return PD, Ph, Ah\fh, Vh
  end

  function compute_matrix_vector(model,Vh)
    Ω      = Triangulation(model)
    dΩ     = Measure(Ω,2*order+1)
    a(u,v) = ∫(∇(v)⋅∇(u))*dΩ
    l(v)   = ∫(1*v)*dΩ
    # α =1,0; a(u,v)=∫(v⋅u)dΩ+∫(α*∇(v)⊙∇(u))dΩ # @santiagobadia: For vector Laplacian
    # f(x) = VectorValue(1.0,0.0)
    # l(v)=∫(v⋅f)dΩ
    assembler = SparseMatrixAssembler(Vh,Vh)
    Ah = assemble_matrix(a,assembler,Vh,Vh)
    lh = assemble_vector(l,assembler,Vh)
    return Ah,lh
  end

  function test_smoother(PD,Ph,Vh,A,b)
    Ωₚ  = Triangulation(PD)
    order = 1
    dΩₚ = Measure(Ωₚ,2*order+1)
    a(u,v) = ∫(∇(v)⋅∇(u))*dΩₚ
    # α =1,0; a(u,v)=∫(v⋅u)dΩ+∫(α*∇(v)⊙∇(u))dΩ # @santiagobadia: For vector Laplacian
    M = PatchBasedLinearSolver(a,Ph,Vh,LUSolver())
    s = RichardsonSmoother(M,10,1.0/3.0)
    x = GridapSolvers.PatchBasedSmoothers._allocate_col_vector(A)
    r = b-A*x
    solve!(x,s,A,r)
    return x
  end

  ##################################################

  domain    = (0.0,1.0,0.0,1.0)
  partition = (2,3)
  
  model  = CartesianDiscreteModel(domain,partition)
  _,Ph,xh,Vh    = returns_PD_Ph_xh_Vh(model)

  parts  = get_part_ids(sequential,(1,2))
  dmodel = CartesianDiscreteModel(parts,domain,partition)
  _,dPh,dxh,dVh = returns_PD_Ph_xh_Vh(dmodel);

  @test num_free_dofs(Ph) == num_free_dofs(dPh)
  @test all(dxh.owned_values.parts[1] .≈ xh[1:3])
  @test all(dxh.owned_values.parts[2] .≈ xh[4:end])

  #################################################
  
  model  = CartesianDiscreteModel(domain,partition)
  PD,Ph,xh,Vh = returns_PD_Ph_xh_Vh(model)
  A,b = compute_matrix_vector(model,Vh)
  x   = test_smoother(PD,Ph,Vh,A,b)

  parts  = get_part_ids(SequentialBackend(),(1,1))
  dmodel = CartesianDiscreteModel(parts,domain,partition)
  dPD,dPh,dxh,dVh = returns_PD_Ph_xh_Vh(dmodel);
  dA,db = compute_matrix_vector(dmodel,dVh);
  dx    = test_smoother(dPD,dPh,dVh,dA,db)

  @test all(dx.owned_values.parts[1] .≈ x)
end
