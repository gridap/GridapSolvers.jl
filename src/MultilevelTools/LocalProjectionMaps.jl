"""
    struct LocalProjectionMap <: Map

  Map that projects a field/field-basis onto another local reference space 
  given by a `ReferenceFE`.

  Example:

  ```julia
  model = CartesianDiscreteModel((0,1,0,1),(2,2))

  reffe_h1 = ReferenceFE(QUAD,lagrangian,Float64,1,space=:Q)
  reffe_l2 = ReferenceFE(QUAD,lagrangian,Float64,1,space=:P)
  U = FESpace(model,reffe_h1)
  u_h1 = interpolate(f,U)

  q_degree = 2
  Π = LocalProjectionMap(reffe_l2,q_degree)
  u_l2 = Π(u_h1)
  ```
"""
struct LocalProjectionMap{A,B,C} <: Map
  reffe :: A
  quad  :: B
  Mq    :: C
end

# Constructors

function LocalProjectionMap(reffe::ReferenceFE,quad::Quadrature)
  q = get_shapefuns(reffe)
  pq = get_coordinates(quad)
  wq = get_weights(quad)

  aq = Fields.BroadcastOpFieldArray(⋅,q,transpose(q))
  Mq = evaluate(IntegrationMap(),evaluate(aq,pq),wq)
  Mq_factorized = cholesky(Mq)
  return LocalProjectionMap(reffe,quad,Mq_factorized)
end

function LocalProjectionMap(reffe::ReferenceFE,qorder)
  quad = Quadrature(get_polytope(reffe),qorder)
  return LocalProjectionMap(reffe,quad)
end

function LocalProjectionMap(poly::Polytope,name::ReferenceFEName,args...;quad_order=-1,kwargs...)
  reffe = ReferenceFE(poly,name,args...;kwargs...)
  if quad_order == -1
    quad_order = 2*(get_order(reffe)+1)
  end
  return LocalProjectionMap(reffe,quad_order)
end

# Action on Field / Array{<:Field}

Arrays.return_cache(k::LocalProjectionMap,f::Field) = _return_cache(k,f)
Arrays.evaluate!(cache,k::LocalProjectionMap,f::Field) = _evaluate!(cache,k,f)

function Arrays.return_cache(k::LocalProjectionMap,f::AbstractVector{<:Field})
  _return_cache(k,transpose(f))
end
function Arrays.evaluate!(cache,k::LocalProjectionMap,f::AbstractVector{<:Field})
  _evaluate!(cache,k,transpose(f))
end

function Arrays.return_cache(k::LocalProjectionMap,f::AbstractMatrix{<:Field})
  @check size(f,1) == 1
  _return_cache(k,f)
end
function Arrays.evaluate!(cache,k::LocalProjectionMap,f::AbstractMatrix{<:Field})
  @check size(f,1) == 1
  ff = _evaluate!(cache,k,f)
  return transpose(ff)
end

function Arrays.return_cache(k::LocalProjectionMap,f::ArrayBlock{A,N}) where {A,N}
  fi = testitem(f)
  ci = return_cache(k,fi)
  fix = evaluate!(ci,k,fi)
  c = Array{typeof(ci),N}(undef,size(f.array))
  g = Array{typeof(fix),N}(undef,size(f.array))
  for i in eachindex(f.array)
    if f.touched[i]
      c[i] = return_cache(k,f.array[i])
    end
  end
  ArrayBlock(g,f.touched),c
end
function Arrays.evaluate!(cache,k::LocalProjectionMap,f::ArrayBlock{A,N}) where {A,N}
  g, c = cache
  @check g.touched == f.touched
  for i in eachindex(f.array)
    if f.touched[i]
      g.array[i] = evaluate!(c[i],k,f.array[i])
    end
  end
  return g
end

function _return_cache(k::LocalProjectionMap,f)
  q = get_shapefuns(k.reffe)
  pq = get_coordinates(k.quad)
  wq = get_weights(k.quad)

  lq = Fields.BroadcastOpFieldArray(⋅,q,f)
  eval_cache = return_cache(lq,pq)
  lqx = evaluate!(eval_cache,lq,pq)
  integration_cache = return_cache(IntegrationMap(),lqx,wq)
  return eval_cache, integration_cache
end

function _evaluate!(cache,k::LocalProjectionMap,f)
  eval_cache, integration_cache = cache
  q = get_shapefuns(k.reffe)

  lq = Fields.BroadcastOpFieldArray(⋅,q,f)
  lqx = evaluate!(eval_cache,lq,get_coordinates(k.quad))
  bq = evaluate!(integration_cache,IntegrationMap(),lqx,get_weights(k.quad))

  λ = ldiv!(k.Mq,bq)
  return linear_combination(λ,q)
end

# Action on CellField / DistributedCellField

function Arrays.evaluate!(cache,k::LocalProjectionMap,f::CellField)
  @assert isa(DomainStyle(f),ReferenceDomain)
  f_data = CellData.get_data(f)
  fk_data = lazy_map(k,f_data)
  return GenericCellField(fk_data,get_triangulation(f),ReferenceDomain())
end

function Arrays.evaluate!(cache,k::LocalProjectionMap,f::GridapDistributed.DistributedCellField)
  fields = map(k,local_views(f))
  return GridapDistributed.DistributedCellField(fields,f.trian)
end

# Optimization for MultiField
function Arrays.lazy_map(k::LocalProjectionMap,a::LazyArray{<:Fill{<:BlockMap}})
  args = map(i->lazy_map(k,i),a.args)
  bm = a.maps.value
  lazy_map(bm,args...)
end