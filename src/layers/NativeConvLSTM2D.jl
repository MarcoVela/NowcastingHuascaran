using Flux

# Utils

function _catn(x::AbstractArray{T, N}...) where {T, N}
  cat(x...; dims=Val(N))
end

function RepeatVector(n)
  Parallel(_catn, ntuple(Returns(identity), n))
end

# layer

# This is a LSTM Cell that receive pre-calculated gates and only calculates new states
# Receives FxNxT tensors (Features, Batches, Time)
struct PassThroughLSTMCell{S}
  state0::S
  function PassThroughLSTMCell(out;
                               init_state = Flux.zeros32)
    state = init_state(out,1)
    return new{typeof(state)}(state)
  end
end

function (m::PassThroughLSTMCell{<:NTuple{2,AbstractMatrix{T}}})((h, c), g::AbstractMatrix{T}) where {T}
  o = size(h, 1)
  input, forget, cell, output = Flux.multigate(g, o, Val(4))
  c′ = @. sigmoid_fast(forget) * c + sigmoid_fast(input) * tanh_fast(cell)
  h′ = @. sigmoid_fast(output) * tanh_fast(c′)
  return (h′, c′), reshape_cell_output(h′, x)
end


