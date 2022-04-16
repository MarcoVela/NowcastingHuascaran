using Flux

struct ConvLSTM2DCell{A,B,S,M}
  Wxh::A
  Whh::B
  Wc::M
  state0::S
end

mutable struct ConvRecur{T,S,P,N}
  cell::T
  state::Tuple{S, S}
  return_sequences::P
  repeat_input::N
end
Flux.@functor ConvRecur

Flux.trainable(a::ConvRecur) = (; cell = a.cell)

function Base.show(io::IO, m::MIME"text/plain", x::ConvRecur)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    Flux._big_show(io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    Flux._layer_show(io, x)
  else
    show(io, x)
  end
end

function _print_convrecur_options(io::IO, l)
  l.return_sequences === false || print(io, ", return_sequences=", l.return_sequences)
  l.repeat_input === 1 || print(io, ", repeat_input=", l.repeat_input)
end

function Base.show(io::IO, m::ConvRecur) 
  print(io, "ConvRecur(", m.cell)
  _print_convrecur_options(io, m)
  print(io, ")")
end
#Base.show(io::IO, m::ConvRecur) = Flux._big_show(io, m)

Flux.reset!(m::ConvRecur) = (m.state = m.cell.state0)

# Requirements:
# The input has to be in the format WHCTN (time and batch in the last two dims)
# WHCT has to be passed to the conv layers (last dim time dimension or merge last two dims and pass them to the conv layer)
# padding has to be a property
# we have to decide whether to return either all or none of the outputs of the convlstm2d layer (maybe a utility function, check performance)

function ConvLSTM2DCell((width, height)::Tuple{<:Integer, <:Integer}, filter::Tuple{<:Integer, <:Integer}, ch::Pair{<:Integer, <:Integer}; 
  init=Flux.glorot_uniform, 
  init_state=Flux.glorot_uniform, 
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
  ranges = Flux.chunk(1:N, n)
  inds_before = ntuple(Returns(:), dim-1)
  inds_after = ntuple(Returns(:), ndims(A)-dim)
  (view(A, inds_before..., r, inds_after...) for r in ranges)
end

function _convlstm2d_output(gates, Wc, c)
  ch = size(gates, 3) ÷ 4
  input_gate = view(gates, :, :, 1:ch, :)
  forget_gate = view(gates, :, :, ch+1:2*ch, :)
  cell_gate = view(gates, :, :, 2*ch+1:3*ch, :)
  output_gate = view(gates, :, :, 3*ch+1:4*ch, :)
  #input_gate, forget_gate, cell_gate, output_gate = eachslice_split(gates, 4; dim=3)
  ch = size(Wc, 3) ÷ 3
  Wci = view(Wc, :, :, 1:ch, :)
  Wcf = view(Wc, :, :, ch+1:2*ch, :)
  Wco = view(Wc, :, :, 2*ch+1:3*ch, :)
  #Wci, Wcf, Wco = eachslice_split(Wc, 3; dim=3)
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

@noinline function _conv_recur_helper(x_t, m)
  h_t, c = m.state
  gates = x_t .+ m.cell.Whh(h_t)
  m.state, y = _convlstm2d_output(gates, m.cell.Wc, c)
  y
end

function (m::ConvRecur{T, S})(x)::S where {T <: ConvLSTM2DCell, S}
  m.state, y = m.cell(m.state, x)
  return y
end

function (m::ConvRecur{T, S})(x::AbstractArray{Q, 4})::S where {T <: ConvLSTM2DCell, Q <: Number, S}
  h = [m(x) for x = eachslice(x; dims=4)]
  sze = size(h[1])
  return reshape(reduce(hcat, h), sze[1], sze[2], sze[3], length(h))
  

  Wxh = m.cell.Wxh(x)
  if m.return_sequences
    h = [
      _conv_recur_helper(Wxh[:,:,:,t:t], m) 
      for _ = 1:m.repeat_input for t in axes(Wxh, 4)
      ]
    sze = size(h[1])
    reshape(reduce(hcat, h), sze[1], sze[2], sze[3], length(h))
    #reduce((a,b) -> cat(a,b; dims=4), h)
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


ConvRecur(m::ConvLSTM2DCell; return_sequences, repeat_input) = ConvRecur(m, m.state0, return_sequences, repeat_input)


ConvLSTM2D(a...; return_sequences=false, repeat_input=1, ka...) = ConvRecur(ConvLSTM2DCell(a...; ka...); return_sequences, repeat_input)



Flux.@functor ConvLSTM2DCell
Flux.trainable(c::ConvLSTM2DCell) = (; Wxh=c.Wxh, Whh=c.Whh, Wc=c.Wc)
Flux._show_children(::ConvLSTM2DCell) = (; )
Flux._show_leaflike(::ConvLSTM2DCell) = true

function _print_convlstm2d_options(io::IO, l)
  all(==(0), l.Wxh.pad) || print(io, ", pad=", Flux._maybetuple_string(l.Wxh.pad))
  all(==(1), l.Wxh.stride) || print(io, ", stride=", Flux._maybetuple_string(l.Wxh.stride))
  all(==(1), l.Wxh.dilation) || print(io, ", dilation=", Flux._maybetuple_string(l.Wxh.dilation))
  (l.Wxh.bias === false) && print(io, ", bias=false")
end


function Base.show(io::IO, l::ConvLSTM2DCell)
  print(io, "ConvLSTM2DCell(")
  print(io, size(l.Wxh.weight)[1:ndims(l.Wxh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wxh), " => ", Flux._channels_out(l.Wxh) ÷ 4)
  _print_convlstm2d_options(io, l)
  print(io, ")")
end
#Base.show(io::IO, m::ConvLSTM2DCell) = print(io, "ConvLSTM2DCell(", m.Whh, ", ", m.Wxh, ")")


