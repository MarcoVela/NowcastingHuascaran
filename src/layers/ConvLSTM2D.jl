using Flux
using Flux: chunk

struct ConvLSTM2DCell{A,B,S,M}
  Wxh::A
  Whh::B
  Wc::M
  state0::S
end

mutable struct ConvRecur{T,S,P}
  cell::T
  state::S
  return_sequences::P
end
Flux.@functor ConvRecur

trainable(a::ConvRecur) = (; cell = a.cell)
Base.show(io::IO, m::ConvRecur) = print(io, "ConvRecur(", m.cell, ")")
reset!(m::ConvRecur) = (m.state = m.cell.state0)

# Requirements:
# The input has to be in the format WHCTN (time and batch in the last two dims)
# WHCT has to be passed to the conv layers (last dim time dimension or merge last two dims and pass them to the conv layer)
# padding has to be a property
# we have to decide whether to return either all or none of the outputs of the convlstm2d layer (maybe a utility function, check performance)

function ConvLSTM2DCell((width, height)::Tuple{<:Integer, <:Integer}, filter::Tuple{<:Integer, <:Integer}, ch::Pair{<:Integer, <:Integer}; 
  init=Flux.glorot_uniform, 
  init_state=Flux.zeros32, 
  pad=0,
  stride=1,
  dilation=1,
  bias=true
  )

  chin, chhid = ch
  chxh = chin => chhid * 4
  chhh = chhid => chhid * 4

  Wxh = Conv(filter, chxh; init, pad=pad, stride=stride, bias=bias)
  Whh = Conv(filter, chhh; init, pad=pad, stride=stride, bias=false)

  padding = Flux.calc_padding(Flux.Conv, pad, filter, dilation, stride)
  hidden_width = width
  hidden_height = height
  filter_w, filter_h = filter
  expand_size(p::Number) = ntuple(_ -> Int(p), 2)
  expand_size(p) = tuple(p...)
  stride_w, stride_h = expand_size(stride)
  if length(padding) == 2
    pad_w, pad_h = padding
    hidden_width = ((width + 2*pad_w - filter_w) ÷ stride_w) + 1
    hidden_height = ((height + 2*pad_h - filter_h) ÷ stride_h) + 1
  elseif length(padding) == 4
    pad_w_top, pad_w_bot, pad_h_top, pad_h_bot = padding
    hidden_width = ((width + pad_w_top + pad_w_bot - filter_w) ÷ stride_w) + 1
    hidden_height = ((height + pad_h_top + pad_h_bot - filter_h) ÷ stride_h) + 1
  end
  Wc = init_state(hidden_width, hidden_height, chhid * 3, 1)

  h = init_state(hidden_width, hidden_height, chhid, 1)

  c = h
  state0 = (h, c)
  ConvLSTM2DCell(Wxh, Whh, Wc, state0)
end

function eachslice_split(A, n; dim)
  N = size(A, dim)
  ranges = chunk(1:N, n)
  inds_before = ntuple(Returns(:), dim-1)
  inds_after = ntuple(Returns(:), ndims(A)-dim)
  (view(A, inds_before..., r, inds_after...) for r in ranges)
end

function _convlstm2d_output(gates, Wc, c)
  input_gate, forget_gate, cell_gate, output_gate = eachslice_split(gates, 4; dim=3)
  Wci, Wcf, Wco = eachslice_split(Wc, 3; dim=3)
  i_t = @. sigmoid_fast(input_gate + Wci * c)
  f_t = @. sigmoid_fast(forget_gate + Wcf * c)
  c = @. f_t * c + i_t * tanh_fast(cell_gate)
  o_t = @. sigmoid_fast(output_gate + Wco * c)
  h′ = @. o_t * tanh_fast(c)
  return (h′, c), h′
end

function (m::ConvLSTM2DCell)((h, c), x_t::Q) where {Q <: AbstractArray{<:Number, 3}}
  gates = m.Wxh(reshape(x_t, size(x_t)..., 1)) .+ m.Whh(h)
  _convlstm2d_output(gates, m.Wc, c)
end

function (m::ConvLSTM2DCell)((h, c), x_t::Q) where {Q <: AbstractArray{<:Number, 4}}
  gates = m.Wxh(x_t) .+ m.Whh(h)
  _convlstm2d_output(gates, m.Wc, c)
end

function (m::ConvRecur{T})(x::Q) where {T <: ConvLSTM2DCell, Q <: AbstractArray{<:Number, 4}}
  Wxh = m.cell.Wxh(x)
  if m.return_sequences
    h = map(t -> begin
      x_t = view(Wxh, :, :, :, t:t)
      h_t, c = m.state
      gates = x_t .+ m.cell.Whh(h_t)
      m.state, y = _convlstm2d_output(gates, m.cell.Wc, c)
      y
    end, axes(Wxh, 4))
    reduce((a,b) -> cat(a,b; dims=4), h)
  else
    for t = axes(Wxh, 4)
      x_t = view(Wxh, :, :, :, t:t)
      h_t, c = m.state
      gates = x_t .+ m.cell.Whh(h_t)
      m.state, _ = _convlstm2d_output(gates, m.cell.Wc, c)
    end
    m.state[1]
  end
end


# function (m::ConvLSTM2DCell)((h, c), x::Q) where {Q <: }
#   gates = m.Wxh(x)::Q .+ m.Whh(h)::Q
#   input_gate, forget_gate, cell_gate, output_gate = view_into_channels(gates, 4)
#   Wci, Wcf, Wco = view_into_channels(m.Wc, 3)
#   i_t = @. sigmoid_fast(input_gate + Wci * c)
#   f_t = @. sigmoid_fast(forget_gate + Wcf * c)
#   c = @. f_t * c + i_t * tanh_fast(cell_gate)
#   o_t = @. sigmoid_fast(output_gate + Wco * c)
#   h′ = @. o_t * tanh_fast(c)
#   return (h′, c), h′
# end

ConvRecur(m::ConvLSTM2DCell; return_sequences) = ConvRecur(m, m.state0, return_sequences)


ConvLSTM2D(a...; return_sequences=false, ka...) = ConvRecur(ConvLSTM2DCell(a...; ka...); return_sequences)
function Base.show(io::IO, l::ConvLSTM2DCell)
  print(io, "ConvLSTM2DCell(", size(l.Wc)[1:2])
  print(io, ", ", size(l.Wxh.weight)[1:ndims(l.Wxh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wxh), " => ", Flux._channels_out(l.Wxh) ÷ 4)
  print(io, ")")
end


Flux.@functor ConvLSTM2DCell
Flux.trainable(c::ConvLSTM2DCell) = (; Wxh=c.Wxh, Whh=c.Whh, Wc=c.Wc)
#Base.show(io::IO, m::ConvLSTM2DCell) = print(io, "ConvLSTM2DCell(", m.Whh, ", ", m.Wxh, ")")


