
abstract type TransferOperatorType end
struct Prolongation <: TransferOperatorType end
struct Restriction <: TransferOperatorType end

abstract type TransferRedistributionType end
struct SerialTransfer <: TransferRedistributionType end
struct DistributedTransfer <: TransferRedistributionType end
struct RedistributedTransfer <: TransferRedistributionType end

abstract type TransferOperator{T<:TransferOperatorType,R<:TransferRedistributionType} <: AbstractMatrix end
const ProlongationOperator = TransferOperator{<:Prolongation}
const RestrictionOperator = TransferOperator{<:Restriction}

function Base.size(op::TransferOperator)
  @abstractmethod
end

function LinearAlgebra.mul!(y::AbstractVector, op::TransferOperator, x::AbstractVector)
  @abstractmethod
end

function LinearAlgebra.adjoint(op::TransferOperator)
  @abstractmethod
end

function update_transfer_operator!(op::TransferOperator, A::AbstractMatrix, x::AbstractVector)
  return op
end

struct TransferRedistributionCache
  components_in
  components_out
  glue
  exchange_cache
  reversed::Bool
  model
  model_red
  space
  space_red
  dv
  dv_red
  glue
  cache_exchange
end

struct InterpolationOperator <: TransferOperator

end

struct RestrictionOperator <: TransferOperator

end

function InterpolationOperator()

end

function RestrictionOperator()

end
