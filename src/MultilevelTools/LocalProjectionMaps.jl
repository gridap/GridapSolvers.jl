
"""
    abstract type LocalProjectionMap{T} <: Map end
"""
abstract type LocalProjectionMap{T} <: Map end

## LocalProjectionMap API

function Arrays.evaluate!(
  cache,
  k::LocalProjectionMap,
  u::GridapDistributed.DistributedCellField,
  dΩ::GridapDistributed.DistributedMeasure
)
  fields = map(local_views(u),local_views(dΩ)) do u,dΩ
    evaluate!(nothing,k,u,dΩ)
  end
  return GridapDistributed.DistributedCellField(fields,u.trian)
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,u::MultiField.MultiFieldFEBasisComponent,dΩ::Measure
)
  nfields, fieldid = u.nfields, u.fieldid
  block_fields(fields,::TestBasis) = lazy_map(BlockMap(nfields,fieldid),fields)
  block_fields(fields,::TrialBasis) = lazy_map(BlockMap((1,nfields),fieldid),fields)

  sf = evaluate!(nothing,k,u.single_field,dΩ)
  sf_data = CellData.get_data(sf)
  mf_data = block_fields(sf_data,BasisStyle(u.single_field))
  return CellData.similar_cell_field(sf,mf_data,get_triangulation(sf),DomainStyle(sf))
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,v::SingleFieldFEBasis{<:TestBasis},dΩ::Measure
)
  cell_v = CellData.get_data(v)
  cell_u = lazy_map(transpose,cell_v)
  u = FESpaces.similar_fe_basis(v,cell_u,get_triangulation(v),TrialBasis(),DomainStyle(v))
  
  data = _compute_local_projections(k,u,dΩ)
  return GenericCellField(data,get_triangulation(u),ReferenceDomain())
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,u::SingleFieldFEBasis{<:TrialBasis},dΩ::Measure
)
  _data = _compute_local_projections(k,u,dΩ)
  data = lazy_map(transpose,_data)
  return GenericCellField(data,get_triangulation(u),ReferenceDomain())
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,u::SingleFieldFEFunction,dΩ::Measure
)
  data = _compute_local_projections(k,u,dΩ)
  return GenericCellField(data,get_triangulation(u),ReferenceDomain())
end

"""
    struct ReffeProjectionMap{T} <: LocalProjectionMap{T}
      op    :: Operation{T}
      reffe :: Tuple{<:ReferenceFEName,Any,Any}
    end

  Map that projects a field/field-basis onto another local reference space 
  given by a `ReferenceFE`.

  Example:

  ```julia
  model = CartesianDiscreteModel((0,1,0,1),(2,2))

  reffe_h1 = ReferenceFE(QUAD,lagrangian,Float64,1,space=:Q)
  reffe_l2 = ReferenceFE(QUAD,lagrangian,Float64,1,space=:P)
  U = FESpace(model,reffe_h1)
  u_h1 = interpolate(f,U)

  Ω = Triangulation(model)
  dΩ = Measure(Ω,2)
  
  Π = LocalProjectionMap(reffe_l2)
  u_l2 = Π(u_h1,dΩ)
  ```
"""
struct ReffeProjectionMap{T,A} <: LocalProjectionMap{T}
  op::Operation{T}
  reffe::A
  function ReffeProjectionMap(
    op::Function,
    reffe::Tuple{<:ReferenceFEName,Any,Any},
  )
    T = typeof(op)
    A = typeof(reffe)
    return new{T,A}(Operation(op),reffe)
  end
end

function LocalProjectionMap(op::Function,basis::ReferenceFEName,args...;kwargs...)
  LocalProjectionMap(op,(basis,args,kwargs))
end

function LocalProjectionMap(basis::ReferenceFEName,args...;kwargs...)
  LocalProjectionMap(identity,basis,args...;kwargs...)
end

function LocalProjectionMap(op::Function,reffe::Tuple{<:ReferenceFEName,Any,Any},)
  ReffeProjectionMap(op,reffe)
end

# We expect the input to be in `TrialBasis` style.
function _compute_local_projections(
  k::ReffeProjectionMap,u::CellField,dΩ::Measure
)
  Ω = get_triangulation(u)
  basis, args, kwargs = k.reffe
  cell_polytopes = lazy_map(get_polytope,get_cell_reffe(Ω))
  cell_reffe = lazy_map(p -> ReferenceFE(p,basis,args...;kwargs...),cell_polytopes)
  test_shapefuns =  lazy_map(get_shapefuns,cell_reffe)
  trial_shapefuns = lazy_map(transpose,test_shapefuns)
  p = SingleFieldFEBasis(trial_shapefuns,Ω,TrialBasis(),ReferenceDomain())
  q = SingleFieldFEBasis(test_shapefuns,Ω,TestBasis(),ReferenceDomain())

  op = k.op.op
  lhs_data = get_array(∫(p⋅q)dΩ)
  rhs_data = get_array(∫(q⋅op(u))dΩ)
  basis_data = CellData.get_data(q)
  return lazy_map(k,lhs_data,rhs_data,basis_data)
end

# Note on the caches: 
#  - We CANNOT overwrite `lhs`: In the case of constant cell_maps and `u` being a `FEFunction`,
#    the lhs will be a Fill, but the rhs will not be optimized (regular LazyArray). 
#    In that case, we will see multiple times the same `lhs` being used for different `rhs`.
#  - The converse never happens (I think), so we can overwrite `rhs` since it will always be recomputed.
function Arrays.return_cache(::LocalProjectionMap,lhs::Matrix{T},rhs::A,basis) where {T,A<:Union{Matrix{T},Vector{T}}}
  return CachedArray(copy(lhs)), CachedArray(copy(rhs))
end

function Arrays.evaluate!(cache,::LocalProjectionMap,lhs::Matrix{T},rhs::A,basis) where {T,A<:Union{Matrix{T},Vector{T}}}
  cmat, cvec = cache

  setsize!(cmat,size(lhs))
  mat = cmat.array
  copyto!(mat,lhs)

  setsize!(cvec,size(rhs))
  vec = cvec.array
  copyto!(vec,rhs)

  f = cholesky!(mat,NoPivot();check=false)
  @check issuccess(f) "Factorization failed"
  ldiv!(f,vec)

  return linear_combination(vec,basis)
end

"""
    SpaceProjectionMap{T} <: LocalProjectionMap{T}
      op    :: Operation{T}
      space :: A
    end
"""
struct SpaceProjectionMap{T,A} <: LocalProjectionMap{T}
  op::Operation{T}
  space::A

  function SpaceProjectionMap(
    op::Function,
    space::FESpace
  )
    T = typeof(op)
    A = typeof(space)
    return new{T,A}(Operation(op),space)
  end
end

function LocalProjectionMap(op::Function,space::FESpace)
  SpaceProjectionMap(op,space)
end

function LocalProjectionMap(space::FESpace)
  LocalProjectionMap(identity,space)
end

function GridapDistributed.local_views(k::SpaceProjectionMap)
  @check isa(k.space,GridapDistributed.DistributedFESpace)
  map(local_views(k.space)) do space
    SpaceProjectionMap(k.op.op,space)
  end
end

function Arrays.evaluate!(
  cache,
  k::SpaceProjectionMap,
  u::GridapDistributed.DistributedCellField,
  dΩ::GridapDistributed.DistributedMeasure
)
  @check isa(k.space,GridapDistributed.DistributedFESpace)
  fields = map(local_views(k),local_views(u),local_views(dΩ)) do k,u,dΩ
    evaluate!(nothing,k,u,dΩ)
  end
  return GridapDistributed.DistributedCellField(fields,u.trian)
end

function _compute_local_projections(
  k::SpaceProjectionMap,u::CellField,dΩ::Measure
)
  space = k.space
  p = get_trial_fe_basis(space)
  q = get_fe_basis(space)

  Ωi = get_triangulation(dΩ.quad)
  ids = lazy_map(ids -> findall(id -> id > 0, ids), get_cell_dof_ids(space,Ωi))

  op = k.op.op
  lhs_data = get_array(∫(p⋅q)dΩ)
  rhs_data = get_array(∫(q⋅op(u))dΩ)
  basis_data = CellData.get_data(q)
  return lazy_map(k,lhs_data,rhs_data,basis_data,ids)
end

function Arrays.return_cache(::LocalProjectionMap,lhs::Matrix{T},rhs::A,basis,ids) where {T,A<:Union{Matrix{T},Vector{T}}}
  return CachedArray(copy(lhs)), CachedArray(copy(rhs)), CachedArray(zeros(eltype(rhs),(length(ids),size(rhs,2))))
end

function Arrays.evaluate!(cache,::LocalProjectionMap,lhs::Matrix{T},rhs::A,basis,ids) where {T,A<:Union{Matrix{T},Vector{T}}}
  cmat, cvec, cvec2 = cache
  n = length(ids)

  if iszero(n)
    setsize!(cvec,size(rhs))
    vec = cvec.array
    fill!(vec,zero(eltype(rhs)))
  else
    setsize!(cmat,(n,n))
    mat = cmat.array
    copyto!(mat,lhs[ids,ids])

    setsize!(cvec2,(n,size(rhs,2)))
    vec2 = cvec2.array
    copyto!(vec2,rhs[ids,:])

    setsize!(cvec,size(rhs))
    vec = cvec.array
    fill!(vec,zero(eltype(rhs)))

    f = cholesky!(mat,NoPivot();check=false)
    @check issuccess(f) "Factorization failed"
    ldiv!(f,vec2)
    vec[ids,:] .= vec2
  end

  return linear_combination(vec,basis)
end
