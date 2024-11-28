
"""
    abstract type LocalProjectionMap{T} <: Map end
"""
abstract type LocalProjectionMap{T} <: Map end

## LocalProjectionMap API

function Arrays.evaluate!(
  cache,
  k::LocalProjectionMap,
  u::GridapDistributed.DistributedCellField
)
  fields = map(local_views(u)) do u
    evaluate!(nothing,k,u)
  end
  return GridapDistributed.DistributedCellField(fields,u.trian)
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,u::MultiField.MultiFieldFEBasisComponent
)
  nfields, fieldid = u.nfields, u.fieldid
  block_fields(fields,::TestBasis) = lazy_map(BlockMap(nfields,fieldid),fields)
  block_fields(fields,::TrialBasis) = lazy_map(BlockMap((1,nfields),fieldid),fields)

  sf = evaluate!(nothing,k,u.single_field)
  sf_data = CellData.get_data(sf)
  mf_data = block_fields(sf_data,BasisStyle(u.single_field))
  return CellData.similar_cell_field(sf,mf_data,get_triangulation(sf),DomainStyle(sf))
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,v::SingleFieldFEBasis{<:TestBasis}
)
  cell_v = CellData.get_data(v)
  cell_u = lazy_map(transpose,cell_v)
  u = FESpaces.similar_fe_basis(v,cell_u,get_triangulation(v),TrialBasis(),DomainStyle(v))
  
  data = _compute_local_projections(k,u)
  return GenericCellField(data,get_triangulation(u),ReferenceDomain())
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,u::SingleFieldFEBasis{<:TrialBasis}
)
  _data = _compute_local_projections(k,u)
  data = lazy_map(transpose,_data)
  return GenericCellField(data,get_triangulation(u),ReferenceDomain())
end

function Arrays.evaluate!(
  cache,k::LocalProjectionMap,u::SingleFieldFEFunction
)
  data = _compute_local_projections(k,u)
  return GenericCellField(data,get_triangulation(u),ReferenceDomain())
end

"""
    struct ReffeProjectionMap{T} <: LocalProjectionMap{T}
      op      :: Operation{T}
      reffe   :: Tuple{<:ReferenceFEName,Any,Any}
      qdegree :: Int
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
  qdegree::Int
  function ReffeProjectionMap(
    op::Function,
    reffe::Union{<:Tuple{<:ReferenceFEName,Any,Any},<:ReferenceFE},
    qdegree::Integer
  )
    T = typeof(op)
    A = typeof(reffe)
    return new{T,A}(Operation(op),reffe,Int(qdegree))
  end
end

function LocalProjectionMap(
  op::Function,
  reffe::Union{<:Tuple{<:ReferenceFEName,Any,Any},<:ReferenceFE},
  qdegree::Integer=2*maximum(reffe[2][2])
)
  ReffeProjectionMap(op,reffe,qdegree)
end

function LocalProjectionMap(op::Function,basis::ReferenceFEName,args...;kwargs...)
  LocalProjectionMap(op,(basis,args,kwargs))
end

function LocalProjectionMap(basis::ReferenceFEName,args...;kwargs...)
  LocalProjectionMap(identity,basis,args...;kwargs...)
end

# We expect the input to be in `TrialBasis` style.
function _compute_local_projections(
  k::ReffeProjectionMap,u::CellField
)
  function _cell_reffe(reffe::Tuple,Ω)
    basis, args, kwargs = reffe
    cell_polytopes = lazy_map(get_polytope,get_cell_reffe(Ω))
    return lazy_map(p -> ReferenceFE(p,basis,args...;kwargs...),cell_polytopes)
  end
  _cell_reffe(reffe::ReferenceFE,Ω) = Fill(reffe,num_cells(Ω))
  
  Ω = get_triangulation(u)
  dΩ = Measure(Ω,k.qdegree)
  cell_reffe = _cell_reffe(k.reffe,Ω)
  test_shapefuns =  lazy_map(get_shapefuns,cell_reffe)
  trial_shapefuns = lazy_map(transpose,test_shapefuns)
  p = SingleFieldFEBasis(trial_shapefuns,Ω,TrialBasis(),ReferenceDomain())
  q = SingleFieldFEBasis(test_shapefuns,Ω,TestBasis(),ReferenceDomain())

  op = k.op.op
  lhs_data = get_array(∫(q⋅p)dΩ)
  rhs_data = get_array(∫(q⋅op(u))dΩ)
  basis_data = CellData.get_data(q)
  return lazy_map(k,lhs_data,rhs_data,basis_data)
end

function Arrays.return_value(::LocalProjectionMap,lhs::Matrix{T},rhs::A,basis) where {T,A<:Union{Matrix{T},Vector{T}}}
  vec = zeros(T,size(rhs))
  return linear_combination(vec,basis)
end

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
    struct SpaceProjectionMap{T} <: LocalProjectionMap{T}
      op      :: Operation{T}
      space   :: A
      qdegree :: Int
    end

Map that projects a CellField onto another `FESpace`. Necessary when the arrival space 
has constraints (e.g. boundary conditions) that need to be taken into account.

"""
struct SpaceProjectionMap{T,A} <: LocalProjectionMap{T}
  op::Operation{T}
  space::A
  qdegree::Int
  function SpaceProjectionMap(
    op::Function,
    space::FESpace,
    qdegree::Integer
  )
    T = typeof(op)
    A = typeof(space)
    return new{T,A}(Operation(op),space,Int(qdegree))
  end
end

function LocalProjectionMap(op::Function,space::FESpace,qdegree::Integer)
  SpaceProjectionMap(op,space,qdegree)
end

function LocalProjectionMap(space::FESpace,qdegree::Integer)
  LocalProjectionMap(identity,space,qdegree)
end

function GridapDistributed.local_views(k::SpaceProjectionMap)
  @check isa(k.space,GridapDistributed.DistributedFESpace)
  map(local_views(k.space)) do space
    SpaceProjectionMap(k.op.op,space,k.qdegree)
  end
end

function Arrays.evaluate!(
  cache,
  k::SpaceProjectionMap,
  u::GridapDistributed.DistributedCellField,
)
  @check isa(k.space,GridapDistributed.DistributedFESpace)
  fields = map(local_views(k),local_views(u)) do k,u
    evaluate!(nothing,k,u)
  end
  return GridapDistributed.DistributedCellField(fields,u.trian)
end

function _compute_local_projections(
  k::SpaceProjectionMap,u::CellField
)
  space = k.space
  p = get_trial_fe_basis(space)
  q = get_fe_basis(space)

  Ω = get_triangulation(space)
  dΩ = Measure(Ω,k.qdegree)
  ids = lazy_map(ids -> findall(id -> id > 0, ids), get_cell_dof_ids(space))

  op = k.op.op
  lhs_data = get_array(∫(p⋅q)dΩ)
  rhs_data = get_array(∫(q⋅op(u))dΩ)
  basis_data = CellData.get_data(q)
  return lazy_map(k,lhs_data,rhs_data,basis_data,ids)
end

function Arrays.return_value(::LocalProjectionMap,lhs::Matrix{T},rhs::A,basis,ids) where {T,A<:Union{Matrix{T},Vector{T}}}
  vec = zeros(T,size(rhs))
  return linear_combination(vec,basis)
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
