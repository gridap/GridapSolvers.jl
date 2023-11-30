
using Gridap
using GridapDistributed
using PartitionedArrays
using GridapPETSc
using SparseMatricesCSR

using Gridap.Geometry, Gridap.FESpaces

function get_dof_coords(trian,space)
  coords = map(local_views(trian),local_views(space),partition(space.gids)) do trian, space, dof_indices
    node_coords = Gridap.Geometry.get_node_coordinates(trian)
    dof_to_node = space.metadata.free_dof_to_node
    dof_to_comp = space.metadata.free_dof_to_comp

    o2l_dofs = own_to_local(dof_indices)
    coords = Vector{PetscScalar}(undef,length(o2l_dofs))
    for (i,dof) in enumerate(o2l_dofs)
      node = dof_to_node[dof]
      comp = dof_to_comp[dof]
      coords[i] = node_coords[node][comp]
    end
    return coords
  end
  ngdofs  = length(space.gids)
  indices = map(local_views(space.gids)) do dof_indices
    owner = part_id(dof_indices)
    own_indices   = OwnIndices(ngdofs,owner,own_to_global(dof_indices))
    ghost_indices = GhostIndices(ngdofs,Int64[],Int32[]) # We only consider owned dofs
    OwnAndGhostIndices(own_indices,ghost_indices)   
  end
  return PVector(coords,indices)
end

function get_sparse_discrete_operators(model)
  topo = get_grid_topology(model)

  G_values, C_values = map(local_views(topo)) do topo
    nF = num_faces(topo,2); nE = num_faces(topo,1); nN = num_faces(topo,0)
    f2e_map = Geometry.get_faces(topo,2,1)
    e2n_map = Geometry.get_faces(topo,1,0)
  
    # Discrete Gradient
    G_rowptr = fill(2,nE+1);G_rowptr[1] = 0
    Gridap.Arrays.length_to_ptrs!(G_rowptr)
  
    G_colval  = fill(0,nE*2)
    G_nzval   = fill(0,nE*2)
    for iE in 1:nE
      for iN in 1:2
        node = e2n_map.data[e2n_map.ptrs[iE]+iN-1]
        G_colval[G_rowptr[iE]+iN-1] = node
        G_nzval[G_rowptr[iE]+iN-1] = (iN == 1 ? -1 : 1)
      end
    end
    G = SparseMatrixCSR{1}(nE,nN,G_rowptr,G_colval,G_nzval)
  
    # Discrete Curl
    C_rowptr = fill(4,nF+1); C_rowptr[1] = 0
    Gridap.Arrays.length_to_ptrs!(C_rowptr)
    C_colval = fill(0,nF*4)
    C_nzval  = fill(0,nF*4)
    for iF in 1:nF
      for iE in 1:4
        edge = f2e_map.data[f2e_map.ptrs[iF]+iE-1]
        C_colval[C_rowptr[iF]+iE-1] = edge
        C_nzval[C_rowptr[iF]+iE-1] = (iE == 1 || iE == 3 ? -1 : 1)
      end
    end
    C = SparseMatrixCSR{1}(nF,nE,C_rowptr,C_colval,C_nzval)
  
    return G, C
  end |> tuple_of_arrays
  
  node_gids = partition(get_face_gids(model,0))
  edge_gids = partition(get_face_gids(model,1))
  face_gids = partition(get_face_gids(model,2))
  G = PSparseMatrix(G_values,edge_gids,node_gids)
  C = PSparseMatrix(C_values,face_gids,edge_gids)
  return G, C
end

function interpolation_operator(biform,U_in,V_out;
                                strat=FullyAssembledRows(),
                                Tm=SparseMatrixCSR{0,PetscScalar,PetscInt},
                                Tv=Vector{PetscScalar})
  assem = SparseMatrixAssembler(Tm,Tv,U_in,V_out,strat)
  return assemble_matrix(biform,assem,U_in,V_out)
end

function get_operators(V_H1_sc,V_H1_vec,V_Hcurl,V_Hdiv,dΩ)
  biform_mass(u,v) = ∫(u⋅v) * dΩ
  biform_grad(u,v) = ∫(∇(u)⋅v) * dΩ
  biform_curl(u,v) = ∫((∇×u)⋅v) * dΩ
  
  G = interpolation_operator(biform_grad,V_H1_sc,V_Hcurl)
  C = interpolation_operator(biform_curl,V_Hcurl,V_Hdiv)
  Π_div  = interpolation_operator(biform_mass,V_H1_vec,V_Hdiv)
  Π_curl = interpolation_operator(biform_mass,V_H1_vec,V_Hcurl)
  return G, C, Π_div, Π_curl
end

###############################################################################

np = (1,1,1)
ranks = with_debug() do distribute
  distribute(LinearIndices((prod(np),)))
end

model = CartesianDiscreteModel(ranks,np,(0,1,0,1,0,1),(10,10,10))
trian = Triangulation(model)

order = 1

reffe_H1_sc = ReferenceFE(lagrangian,Float64,order)
V_H1_sc = FESpace(model,reffe_H1_sc)
U_H1_sc = TrialFESpace(V_H1_sc)

reffe_H1 = ReferenceFE(lagrangian,VectorValue{3,Float64},order)
V_H1 = FESpace(model,reffe_H1)
U_H1 = TrialFESpace(V_H1)

reffe_Hdiv = ReferenceFE(raviart_thomas,Float64,order-1)
V_Hdiv = FESpace(model,reffe_Hdiv)
U_Hdiv = TrialFESpace(V_Hdiv)

reffe_Hcurl = ReferenceFE(nedelec,Float64,order-1)
V_Hcurl = FESpace(model,reffe_Hcurl)
U_Hcurl = TrialFESpace(V_Hcurl)

##############################################################################
dΩ = Measure(trian,(order+1)*2)

coords = get_dof_coords(trian,V_H1)

G, C, Π_div, Π_curl = get_operators(V_H1_sc,V_H1,V_Hcurl,V_Hdiv,dΩ);

α  = 1.0
f(x)   = VectorValue([0.0,0.0,1.0])
a(u,v) = ∫(u⋅v + α⋅(∇⋅u)⋅(∇⋅v)) * dΩ
l(v)   = ∫(f⋅v) * dΩ

op = AffineFEOperator(a,l,V_Hdiv,U_Hdiv)
A  = get_matrix(op)
b  = get_vector(op)


function ads_ksp_setup(ksp,rtol,maxits,dim,coords,G,C,Π_div,Π_curl)
  rtol = PetscScalar(rtol)
  atol = GridapPETSc.PETSC.PETSC_DEFAULT
  dtol = GridapPETSc.PETSC.PETSC_DEFAULT
  maxits = PetscInt(maxits)

  @check_error_code GridapPETSc.PETSC.KSPView(ksp[],C_NULL)
  @check_error_code GridapPETSc.PETSC.KSPSetType(ksp[],GridapPETSc.PETSC.KSPGMRES)
  @check_error_code GridapPETSc.PETSC.KSPSetTolerances(ksp[], rtol, atol, dtol, maxits)

  pc = Ref{GridapPETSc.PETSC.PC}()
  @check_error_code GridapPETSc.PETSC.KSPGetPC(ksp[],pc)
  @check_error_code GridapPETSc.PETSC.PCSetType(pc[],GridapPETSc.PETSC.PCHYPRE)

  #map(partition(coords)) do coords
  #  nloc = length(coords)
  #  @check_error_code GridapPETSc.PETSC.PCSetCoordinates(pc[],dim,nloc,coords)
  #end

  _G = convert(PETScMatrix,G)
  _C = convert(PETScMatrix,C)
  _Π_div = convert(PETScMatrix,Π_div)
  _Π_curl = convert(PETScMatrix,Π_curl)
  @check_error_code GridapPETSc.PETSC.PCHYPRESetDiscreteGradient(pc[],_G.mat[])
  @check_error_code GridapPETSc.PETSC.PCHYPRESetDiscreteCurl(pc[],_C.mat[])
  @check_error_code GridapPETSc.PETSC.PCHYPRESetInterpolations(pc[],dim,_Π_div.mat[],C_NULL,_Π_curl.mat[],C_NULL)
end

options = "-ksp_converged_reason"
GridapPETSc.with(args=split(options)) do
  ksp_setup(ksp) = ads_ksp_setup(ksp,1e-8,100,3,coords,G,C,Π_div,Π_curl)
  solver = PETScLinearSolver(ksp_setup)
  ns = numerical_setup(symbolic_setup(solver,A),A)
  x = pfill(0.0,partition(axes(A,2)))
  solve!(x,ns,b)
end
