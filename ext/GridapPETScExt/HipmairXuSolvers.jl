
function ADS_Solver(model,tags,order;rtol=1e-8,maxits=300)
  @assert (num_cell_dims(model) == 3) "Not implemented for 2D"
  @assert (order == 1) "Only works for linear order"

  V_H1_sc, V_H1, V_Hdiv, V_Hcurl = get_ads_spaces(model,order,tags)
  G, C, Π_div, Π_curl = get_ads_operators(V_H1_sc,V_H1,V_Hcurl,V_Hdiv)

  D = num_cell_dims(model)
  ksp_setup(ksp) = ads_ksp_setup(ksp,rtol,maxits,D,G,C,Π_div,Π_curl)
  return PETScLinearSolver(ksp_setup)
end

function get_ads_spaces(model,order,tags)
  reffe_H1_sc = ReferenceFE(lagrangian,Float64,order)
  V_H1_sc = FESpace(model,reffe_H1_sc;dirichlet_tags=tags)
  
  reffe_H1 = ReferenceFE(lagrangian,VectorValue{D,Float64},order)
  V_H1 = FESpace(model,reffe_H1;dirichlet_tags=tags)
  
  reffe_Hdiv = ReferenceFE(raviart_thomas,Float64,order-1)
  V_Hdiv = FESpace(model,reffe_Hdiv;dirichlet_tags=tags)
  
  reffe_Hcurl = ReferenceFE(nedelec,Float64,order-1)
  V_Hcurl = FESpace(model,reffe_Hcurl;dirichlet_tags=tags)

  return V_H1_sc, V_H1, V_Hdiv, V_Hcurl
end

function get_ads_operators(V_H1_sc,V_H1_vec,V_Hcurl,V_Hdiv)
  G = interpolation_operator(u->∇(u),V_H1_sc,V_Hcurl)
  C = interpolation_operator(u->cross(∇,u),V_Hcurl,V_Hdiv)
  Π_div  = interpolation_operator(u->u,V_H1_vec,V_Hdiv)
  Π_curl = interpolation_operator(u->u,V_H1_vec,V_Hcurl)
  return G, C, Π_div, Π_curl
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