using Flux

struct ConvLSTM2DCell{A,S,M}
  Wxh::A
  Whh::A
  Wc::M
  state0::S
end

# Requirements:
# The input has to be in the format WHCTN (time and batch in the last two dims)
# WHCT has to be passed to the conv layers (last dim time dimension or merge last two dims and pass them to the conv layer)
# padding has to be a property
# we have to decide whether to return either all or none of the outputs of the convlstm2d layer (maybe a utility function, check performance)

function ConvLSTM2DCell((width, height)::Tuple{<:Integer, <:Integer}, filter::Tuple{<:Integer, <:Integer}, ch::Pair{<:Integer, <:Integer}; 
  init=Flux.glorot_uniform, 
  init_state=Flux.zeros32, 
  batch_size=1,
  pad=0,
  stride=1,
  dilation=1,
  )

  chin, chhid = ch
  chxh = chin => chhid * 4
  chhh = chhid => chhid * 4

  Wxh = Conv(filter, chxh; init, pad=pad, stride=stride)
  Whh = Conv(filter, chhh; init, pad=pad, stride=stride)

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
  hidden_width = width + 2*pad
  Wc = init_state(hidden_width, hidden_height, chhid * 3, 1)

  h = init_state(hidden_width, hidden_height, chhid, batch_size)

  c = h
  state0 = (h, c)
  ConvLSTM2DCell(Wxh, Whh, Wc, state0)
end


function (m::ConvLSTM2DCell)((h, c), x::Q) where {Q <: AbstractArray{<: Number, 4}}
  gates = m.Wxh(x)::Q .+ m.Whh(h)::Q
  W,H,C,T = size(gates)
  ch_original = C ÷ 4
  input_gate, forget_gate, cell_gate, output_gate = map(chunk -> reshape(chunk, (W,H,ch_original,T)), Flux.chunk(gates, 4))
  Wci, Wcf, Wco = map(chunk -> reshape(chunk, (W,H,ch_original, 1)), Flux.chunk(m.Wc, 3))
  i_t = @. sigmoid_fast(input_gate + Wci * c)
  f_t = @. sigmoid_fast(forget_gate + Wcf * c)
  c = @. f_t * c + i_t * tanh_fast(cell_gate)
  o_t = @. sigmoid_fast(output_gate + Wco * c)
  h′ = @. o_t * tanh_fast(c)
  return (h′, c), h′
end

Flux.Recur(m::ConvLSTM2DCell) = Flux.Recur(m, m.state0)
ConvLSTM2D(a...; ka...) = Flux.Recur(ConvLSTM2DCell(a...; ka...))
function Base.show(io::IO, l::ConvLSTM2DCell)
  print(io, "ConvLSTM2DCell(", size(l.Wc)[1:2])
  print(io, ", ", size(l.Wxh.weight)[1:ndims(l.Wxh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wxh), " => ", Flux._channels_out(l.Wxh) ÷ 4)
  print(io, ")")
end

Flux._show_leaflike(::Tuple{AbstractArray, AbstractArray}) = true

Flux.@functor ConvLSTM2DCell



