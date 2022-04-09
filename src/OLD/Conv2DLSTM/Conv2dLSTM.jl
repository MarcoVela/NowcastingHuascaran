using Flux

struct Conv2dLSTMCell{A,S,M}
  Wxh::A
  Whh::A
  Wc::M
  state0::S
end


function Conv2dLSTMCell(input_shape, k::Tuple{Integer,Integer}, ch; init=Flux.glorot_uniform, init_state=Flux.zeros32, batch_size=1)
  chin, chhid = ch
  wi, he = input_shape
  chxh = chin => chhid * 4
  chhh = chhid => chhid * 4

  Wxh = Conv(k, chxh; init, pad=SamePad())

  Whh = Conv(k, chhh; init, pad=SamePad())

  Wc = init_state(wi, he, chhid * 3, 1)

  h = init_state(wi, he, chhid, batch_size)

  c = h
  state0 = (h, c)
  Conv2dLSTMCell(Wxh, Whh, Wc, state0)
end

function (m::Conv2dLSTMCell)((h, c), x::M) where {M}
  gates = m.Wxh(x)::M .+ m.Whh(h)::M
  wi,he,ch,N = size(gates)
  origch = ch ÷ 4
  input_gate, forget_gate, cell_gate, output_gate = map(chunk -> reshape(chunk, wi, he, origch, N), Flux.chunk(gates, 4))
  
  Wci, Wcf, Wco = map(chunk -> reshape(chunk, wi, he, origch, 1), Flux.chunk(m.Wc, 3))

  i_t = @. σ(input_gate + Wci * c)
  f_t = @. σ(forget_gate + Wcf * c)
  # g_t = @. tanh(cell_gate)
  # c = @. f_t * c + i_t * tanh(g_t)
  c = @. f_t * c + i_t * tanh(cell_gate)
  o_t = @. σ(output_gate + Wco * c)
  h′ = @. o_t * tanh(c)
  return (h′, c), h′
end


Flux.Recur(m::Conv2dLSTMCell) = Flux.Recur(m, m.state0)
Conv2dLSTM(a...; ka...) = Flux.Recur(Conv2dLSTMCell(a...; ka...))
function Base.show(io::IO, l::Conv2dLSTMCell)
  print(io, "Conv2dLSTMCell(", size(l.Wc)[1:2])
  print(io, ", ", size(l.Wxh.weight)[1:ndims(l.Wxh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wxh), " => ", Flux._channels_out(l.Wxh) ÷ 4)
  print(io, ")")
end

Flux._show_leaflike(::Tuple{AbstractArray, AbstractArray}) = true

Flux.@functor Conv2dLSTMCell



