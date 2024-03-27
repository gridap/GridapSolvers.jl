
using Gridap
using GridapDistributed
using PartitionedArrays
using GridapPETSc
using SparseMatricesCSR
using LinearAlgebra

using Gridap.Geometry, Gridap.FESpaces, Gridap.CellData, Gridap.Arrays

function get_operators(V_H1_sc,V_H1_vec,V_Hcurl,V_Hdiv)
  G = interpolation_operator(u->∇(u),V_H1_sc,V_Hcurl)
  C = interpolation_operator(u->cross(∇,u),V_Hcurl,V_Hdiv)
  Π_div  = interpolation_operator(u->u,V_H1_vec,V_Hdiv)
  Π_curl = interpolation_operator(u->u,V_H1_vec,V_Hcurl)
  return G, C, Π_div, Π_curl
end

function interpolation_operator(op,U_in,V_out;
                                strat=SubAssembledRows(),
                                Tm=SparseMatrixCSR{0,PetscScalar,PetscInt},
                                Tv=Vector{PetscScalar})
  out_dofs = get_fe_dof_basis(V_out)
  in_basis  = get_fe_basis(U_in)
  
  cell_interp_mats = out_dofs(op(in_basis))
  local_contr = map(local_views(out_dofs),cell_interp_mats) do dofs, arr
    contr = DomainContribution()
    add_contribution!(contr,get_triangulation(dofs),arr)
    return contr
  end
  contr = GridapDistributed.DistributedDomainContribution(local_contr)
  
  matdata = collect_cell_matrix(U_in,V_out,contr)
  assem = SparseMatrixAssembler(Tm,Tv,U_in,V_out,strat)
  
  I = allocate_matrix(assem,matdata)
  takelast_matrix!(I,assem,matdata)
  return I
end

function takelast_matrix(a::SparseMatrixAssembler,matdata)
  m1 = Gridap.Algebra.nz_counter(get_matrix_builder(a),(get_rows(a),get_cols(a)))
  symbolic_loop_matrix!(m1,a,matdata)
  m2 = Gridap.Algebra.nz_allocation(m1)
  takelast_loop_matrix!(m2,a,matdata)
  m3 = Gridap.Algebra.create_from_nz(m2)
  return m3
end

function takelast_matrix!(mat,a::SparseMatrixAssembler,matdata)
  LinearAlgebra.fillstored!(mat,zero(eltype(mat)))
  takelast_matrix_add!(mat,a,matdata)
end

function takelast_matrix_add!(mat,a::SparseMatrixAssembler,matdata)
  takelast_loop_matrix!(mat,a,matdata)
  Gridap.Algebra.create_from_nz(mat)
end

function takelast_loop_matrix!(A,a::GridapDistributed.DistributedSparseMatrixAssembler,matdata)
  rows = get_rows(a)
  cols = get_cols(a)
  map(takelast_loop_matrix!,local_views(A,rows,cols),local_views(a),matdata)
end

function takelast_loop_matrix!(A,a::SparseMatrixAssembler,matdata)
  strategy = Gridap.FESpaces.get_assembly_strategy(a)
  for (cellmat,_cellidsrows,_cellidscols) in zip(matdata...)
    cellidsrows = Gridap.FESpaces.map_cell_rows(strategy,_cellidsrows)
    cellidscols = Gridap.FESpaces.map_cell_cols(strategy,_cellidscols)
    @assert length(cellidscols) == length(cellidsrows)
    @assert length(cellmat) == length(cellidsrows)
    if length(cellmat) > 0
      rows_cache = array_cache(cellidsrows)
      cols_cache = array_cache(cellidscols)
      vals_cache = array_cache(cellmat)
      mat1 = getindex!(vals_cache,cellmat,1)
      rows1 = getindex!(rows_cache,cellidsrows,1)
      cols1 = getindex!(cols_cache,cellidscols,1)
      add! = Gridap.Arrays.AddEntriesMap((a,b) -> b)
      add_cache = return_cache(add!,A,mat1,rows1,cols1)
      caches = add_cache, vals_cache, rows_cache, cols_cache
      _takelast_loop_matrix!(A,caches,cellmat,cellidsrows,cellidscols)
    end
  end
  A
end

@noinline function _takelast_loop_matrix!(mat,caches,cell_vals,cell_rows,cell_cols)
  add_cache, vals_cache, rows_cache, cols_cache = caches
  add! = Gridap.Arrays.AddEntriesMap((a,b) -> b)
  for cell in 1:length(cell_cols)
    rows = getindex!(rows_cache,cell_rows,cell)
    cols = getindex!(cols_cache,cell_cols,cell)
    vals = getindex!(vals_cache,cell_vals,cell)
    evaluate!(add_cache,add!,mat,vals,rows,cols)
  end
end

function ads_ksp_setup(ksp,rtol,maxits,dim,G,C,Π_div,Π_curl)
  rtol = PetscScalar(rtol)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = PetscInt(maxits)

  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCHYPRE)

  _G = convert(PETScMatrix,G)
  _C = convert(PETScMatrix,C)
  _Π_div = convert(PETScMatrix,Π_div)
  _Π_curl = convert(PETScMatrix,Π_curl)
  @check_error_code GridapPETSc.PETSC.PCHYPRESetDiscreteGradient(pc[],_G.mat[])
  @check_error_code GridapPETSc.PETSC.PCHYPRESetDiscreteCurl(pc[],_C.mat[])
  @check_error_code GridapPETSc.PETSC.PCHYPRESetInterpolations(pc[],dim,_Π_div.mat[],C_NULL,_Π_curl.mat[],C_NULL)

  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
end

###############################################################################

n  = 10
D  = 3
order = 1
np = (1,1,1)#Tuple(fill(1,D))
ranks = with_mpi() do distribute
  distribute(LinearIndices((prod(np),)))
end

domain = (D==2) ? (0,1,0,1) : (0,1,0,1,0,1)
ncells = (D==2) ? (n,n) : (n,n,n)
model = CartesianDiscreteModel(ranks,np,domain,ncells)
trian = Triangulation(model)

reffe_H1_sc = ReferenceFE(lagrangian,Float64,order)
V_H1_sc = FESpace(model,reffe_H1_sc)#;dirichlet_tags="boundary")
U_H1_sc = TrialFESpace(V_H1_sc)

reffe_H1 = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
V_H1 = FESpace(model,reffe_H1)#;dirichlet_tags="boundary")
U_H1 = TrialFESpace(V_H1)

reffe_Hdiv = ReferenceFE(raviart_thomas,Float64,order-1)
V_Hdiv = FESpace(model,reffe_Hdiv)#;dirichlet_tags="boundary")
U_Hdiv = TrialFESpace(V_Hdiv)

reffe_Hcurl = ReferenceFE(nedelec,Float64,order-1)
V_Hcurl = FESpace(model,reffe_Hcurl)#;dirichlet_tags="boundary")
U_Hcurl = TrialFESpace(V_Hcurl)

##############################################################################
dΩ = Measure(trian,(order+1)*2)

G, C, Π_div, Π_curl = get_operators(V_H1_sc,V_H1,V_Hcurl,V_Hdiv);

u(x) = x[1]^3 + x[2]^3
u_h1 = interpolate(u,U_H1_sc)
x_h1 = get_free_dof_values(u_h1)

#u_hcurl = interpolate(∇(u_h1),U_Hcurl)
#x_hcurl = G * x_h1
#@assert norm(x_hcurl - get_free_dof_values(u_hcurl)) < 1.e-8
#
#u_hdiv = interpolate(∇×(u_hcurl),U_Hdiv)
#x_hdiv  = C * x_hcurl
#@assert norm(x_hdiv - get_free_dof_values(u_hdiv)) < 1.e-8
#
#u_vec(x) = VectorValue(x[1]^3,x[2]^3,x[3]^3)
#u_h1_vec = interpolate(u_vec,V_H1)
#x_h1_vec = get_free_dof_values(u_h1_vec)
#
#u_hcurl_bis = interpolate(u_h1_vec,U_Hcurl)
#x_hcurl_bis = Π_curl * x_h1_vec
#@assert norm(x_hcurl_bis - get_free_dof_values(u_hcurl_bis)) < 1.e-8

#u_hdiv_bis = interpolate(u_h1_vec,U_Hcurl)
#x_hdiv_bis = Π_curl * x_h1_vec
#@assert norm(x_hdiv_bis - get_free_dof_values(u_hdiv_bis)) < 1.e-8

############################################################################################

sol(x) = (D==2) ? VectorValue(x[1],x[2]) : VectorValue(x[1],x[2],x[3])
f(x)   = (D==2) ? VectorValue(x[1],x[2]) : VectorValue(x[1],x[2],x[3])

α = 1.0
a(u,v) = ∫(u⋅v + α⋅(∇⋅u)⋅(∇⋅v)) * dΩ
l(v)   = ∫(f⋅v) * dΩ

V = FESpace(model,reffe_Hdiv)#;dirichlet_tags="boundary")
U = TrialFESpace(V,sol)
op = AffineFEOperator(a,l,V,U)
A  = get_matrix(op)
b  = get_vector(op)

options = "-ksp_converged_reason"
GridapPETSc.with(args=split(options)) do
  ksp_setup(ksp) = ads_ksp_setup(ksp,1e-8,300,D,G,C,Π_div,Π_curl)
  solver = PETScLinearSolver(ksp_setup)
  ns = numerical_setup(symbolic_setup(solver,A),A)
  x = pfill(0.0,partition(axes(A,2)))
  solve!(x,ns,b)
end
