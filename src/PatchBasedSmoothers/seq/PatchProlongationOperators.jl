
struct PatchProlongationOperator{A,B,C,D,E}
  op :: A
  Ph :: B
  Vh :: C
  PD :: D
  lhs :: Function
  rhs :: Function
  caches :: E
  is_nonlinear :: Bool
end

function PatchProlongationOperator(lev,sh,PD,lhs,rhs;is_nonlinear=false)
  @assert has_refinement(sh,lev)

  # Default prolongation (i.e interpolation)
  op = ProlongationOperator(lev,sh,0;mode=:residual)

  # Patch-based correction fespace
  fmodel = get_model(sh,lev)
  glue = sh[lev].mh_level.ref_glue
  patches_mask = get_coarse_node_mask(fmodel,glue)

  Vh = MultilevelTools.get_fe_space(sh,lev)
  cell_conformity = sh[lev].cell_conformity
  Ph = PatchFESpace(Vh,PD,cell_conformity;patches_mask)

  # Solver caches
  u, v = get_trial_fe_basis(Vh), get_fe_basis(Vh)
  contr = is_nonlinear ? lhs(zero(Vh),u,v) : lhs(u,v)
  matdata = collect_cell_matrix(Ph,Ph,contr)
  ns, Ap = map(local_views(Ph),matdata) do Ph, matdata
    assem = SparseMatrixAssembler(Ph,Ph)
    Ap    = assemble_matrix(assem,matdata)
    ns = numerical_setup(symbolic_setup(LUSolver(),Ap),Ap)
    return ns, Ap
  end |> tuple_of_arrays
  Ap = is_nonlinear ? Ap : nothing

  xh, dxh = zero_free_values(Vh), zero_free_values(Vh)
  dxp, rp = zero_free_values(Ph), zero_free_values(Ph)
  caches = ns, xh, dxh, dxp, rp, Ap

  return PatchProlongationOperator(op,Ph,Vh,PD,lhs,rhs,caches,is_nonlinear)
end

# Please make this a standard API or something
function update_patch_operator!(op::PatchProlongationOperator,x::PVector)
  Vh, Ph = op.Vh, op.Ph
  ns, _, _, _, _, Ap = op.caches
  
  u, v = get_trial_fe_basis(Vh), get_fe_basis(Vh)
  contr = is_nonlinear ? op.lhs(FEFunction(x,Vh),u,v) : op.lhs(u,v)
  matdata = collect_cell_matrix(Ph,Ph,contr)
  map(ns,Ap,local_views(Ph),matdata) do ns,Ap,Ph, matdata
    assem = SparseMatrixAssembler(Ph,Ph)
    assemble_matrix!(Ap,assem,matdata)
    numerical_setup!(ns,Ap)
  end
end

function LinearAlgebra.mul!(x,op::PatchProlongationOperator,y)
  Ap_ns, xh, dxh, dxp, rp, _ = op.caches

  mul!(x,op.op,y) # TODO: Quite awful, but should be fixed with PA 0.4
  copy!(xh,x)
  duh = FEFunction(op.Vh,xh)
  assemble_vector!(v->op.rhs(duh,v),rp,op.Ph)
  map(solve!,partition(dxp),Ap_ns,partition(rp))
  inject!(dxh,op.Ph,dxp)

  map(own_values(x),own_values(dxh)) do x, dxh
    x .= x .- dxh
  end
  consistent!(x) |> fetch
  return x
end

function setup_patch_prolongation_operators(sh,patch_decompositions,lhs,rhs,qdegrees;is_nonlinear=false)
  map(linear_indices(patch_decompositions),patch_decompositions) do lev,PD
    qdegree = isa(qdegrees,Number) ? qdegrees : qdegrees[lev]
    Ω = Triangulation(PD)
    dΩ = Measure(Ω,qdegree)
    lhs_i = is_nonlinear ? (u,du,dv) -> lhs(u,du,dv,dΩ) : (u,v) -> lhs(u,v,dΩ)
    rhs_i = (u,v) -> rhs(u,v,dΩ)
    PatchProlongationOperator(lev,sh,PD,lhs_i,rhs_i;is_nonlinear)
  end
end

function get_coarse_node_mask(fmodel::GridapDistributed.DistributedDiscreteModel,glue)
  gids = get_face_gids(fmodel,0)
  mask = map(local_views(fmodel),glue,partition(gids)) do fmodel, glue, gids
    mask = get_coarse_node_mask(fmodel,glue)
    mask[ghost_to_local(gids)] .= false # Mask ghost nodes as well
    return mask
  end
  return mask
end

# Coarse nodes are the ones that are shared by fine cells that do not belong to the same coarse cell. 
# Conversely, fine nodes are the ones shared by fine cells that all have the same parent coarse cell.
function get_coarse_node_mask(fmodel::DiscreteModel{Dc},glue) where Dc
  ftopo = get_grid_topology(fmodel)
  n2c_map = Gridap.Geometry.get_faces(ftopo,0,Dc)
  n2c_map_cache = array_cache(n2c_map)
  f2c_cells   = glue.n2o_faces_map[Dc+1]
  is_boundary = get_isboundary_face(ftopo,0)

  is_coarse = map(1:length(n2c_map)) do n
    nbor_cells = getindex!(n2c_map_cache,n2c_map,n)
    parent = f2c_cells[first(nbor_cells)]
    return is_boundary[n] || any(c -> f2c_cells[c] != parent, nbor_cells)
  end

  return is_coarse
end
