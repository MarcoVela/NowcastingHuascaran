using Flux

# Helpers




function _calc_out_dims((width, height), pad, filter, dilation, stride)
  padding = Flux.calc_padding(Flux.Conv, pad, filter, dilation, stride)
  out_width = width
  out_height = height
  filter_w, filter_h = filter
  expand_size(p::Number) = ntuple(_ -> Int(p), 2)
  expand_size(p) = tuple(p...)
  stride_w, stride_h = expand_size(stride)
  if length(padding) == 2
    pad_w, pad_h = padding
    out_width = ((width + 2*pad_w - filter_w) ÷ stride_w) + 1
    out_height = ((height + 2*pad_h - filter_h) ÷ stride_h) + 1
  elseif length(padding) == 4
    pad_w_top, pad_w_bot, pad_h_top, pad_h_bot = padding
    out_width = ((width + pad_w_top + pad_w_bot - filter_w) ÷ stride_w) + 1
    out_height = ((height + pad_h_top + pad_h_bot - filter_h) ÷ stride_h) + 1
  end
  out_width, out_height
end


function _catn(x::AbstractArray{T, N}...) where {T, N}
  cat(x...; dims=Val(N))
end
struct TimeDistributed{M}
  m::M
end

function (t::TimeDistributed)(x::AbstractArray{T, 5}) where {T}
  h = [t.m(x_t) for x_t in eachslice(x; dims=5)]
  sze = size(h[1])
  reshape(cat(h...; dims=ndims(h[1])), sze..., length(h))
end

Flux.@functor TimeDistributed

Flux.trainable(l::TimeDistributed) = (; m=l.m)

Base.show(io::IO, m::TimeDistributed) = print(io,"TimeDistributed(", m.m, ")")# Flux._big_show(io, m)


struct KeepLast{N, M<:Flux.Recur}
  n::N
  m::M
end

function KeepLast(m)
  KeepLast(1, m)
end

function (k::KeepLast)(x::AbstractArray{T, N}) where {T, N}
  before_dims = ntuple(Returns(:), N-1)
  n2 = size(x, N)
  discarted = ifelse(n2 - k.n < 0, 0, n2 - k.n)
  k.m(view(x, before_dims..., 1:discarted))
  k.m(view(x, before_dims..., discarted+1:n2))
end

Flux.@functor KeepLast

Flux.trainable(k::KeepLast) = (; m=k.m)

function Base.show(io::IO, m::MIME"text/plain", x::KeepLast)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    Flux._big_show(io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    Flux._layer_show(io, x)
  else
    show(io, x)
  end
end

Base.show(io::IO, k::KeepLast) = print(io,"KeepLast(", k.m, ")")# Flux._big_show(io, m)



function Base.show(io::IO, m::MIME"text/plain", x::TimeDistributed)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    Flux._big_show(io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    Flux._layer_show(io, x)
  else
    show(io, x)
  end
end


struct RepeatInput{N, M}
  n::N
  m::M
end

function (r::RepeatInput)(x::AbstractArray{T, N}) where {T, N}
  h = [r.m(x) for _ in 1:r.n]
  cat(h...; dims=ndims(h[1]))
end

function Base.show(io::IO, m::MIME"text/plain", x::RepeatInput)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    Flux._big_show(io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    Flux._layer_show(io, x)
  else
    show(io, x)
  end
end

Base.show(io::IO, r::RepeatInput) = print(io, "RepeatInput(", r.n, ", ", r.m, ")")


Flux.@functor RepeatInput


# Layers:

struct SimpleConvLSTM2DCell{W1, W2, B, F, F2, S, SH}
  Wxh::W1
  Whh::W2
  b::B
  activation::F
  σ::F2
  state0::S
  state_shape::SH
end

function _create_bias(b::Bool, n, initb)
  b ? initb(n) : b
end

function SimpleConvLSTM2DCell((w,h)::Tuple{<:Integer, <:Integer}, filter::Tuple{<:Integer, <:Integer}, ch::Pair{<:Integer, <:Integer};
  init = Flux.glorot_uniform, 
  init_state = Flux.zeros32,
  initb = Flux.zeros32,
  activation = tanh,
  σ = σ,
  pad = 0,
  stride = 1,
  dilation = 1,
  bias::Bool = true)

  chin, chhid = ch
  chxh = chin => chhid * 4
  chhh = chhid => chhid * 4

  Wxh = Conv(filter, chxh; init, pad=pad, stride=stride, bias=true)
  Whh = Conv(filter, chhh; init, pad=SamePad(), stride=stride, bias=true)

  out_width, out_height = _calc_out_dims((w, h), pad, filter, dilation, stride)

  _h = init_state(out_width*out_height*chhid, 1)
  _c = init_state(out_width*out_height*chhid, 1)
  _b = _create_bias(bias, out_width*out_height*chhid*4, initb)
  _state0 = (_h, _c)

  _state_shape = (out_width, out_height, chhid)
  SimpleConvLSTM2DCell(Wxh, Whh, _b, activation, σ, _state0, _state_shape)
  # new{typeof(m), typeof(_b), typeof(_state0), typeof(_state_shape)}(m, _b, _state0, _state_shape)
end





# Receives WxHxCxNxT (Width, Height, Channels, Batch, Time)

# Receives

function _lstm_output((h, c), g, activ, σ)
  _activ = NNlib.fast_act(activ, g)
  _σ = NNlib.fast_act(σ, g)
  o = size(h, 1)
  input, forget, cell, output = Flux.multigate(g, o, Val(4))
  c′ = @. _σ(forget) * c + _σ(input) * _activ(cell)
  h′ = @. _σ(output) * _activ(c′)
  return (h′, c′), h′ # Removing Flux.reshape_cell_output 
end


function (m::SimpleConvLSTM2DCell)((h, c), x::AbstractArray{T, 4}) where {T}
  # h_size = Flux.outputsize(m.m[1].layers[1], size(x))[begin:4-1]
  h_mat = reshape(h, m.state_shape..., size(h, 2)) # size of hidden state is initialized to 1 and can change if the batch size increases
  gates = Flux.flatten(m.Wxh(x) .+ m.Whh(h_mat)) .+ m.b
  (h′, c′), y = _lstm_output((h, c), gates, m.activation, m.σ) # m.lstm((h, c), Flux.flatten(m.conv(x, h_mat))) # all are matrices and second dimension is batch size
  (h′, c′), reshape(y, m.state_shape..., size(x, 4))
end

Flux.@functor SimpleConvLSTM2DCell

Flux.trainable(m::SimpleConvLSTM2DCell{T,W,<:Bool}) where {T,W} = (; Wxh=m.Wxh, Whh=m.Whh)
Flux.trainable(m::SimpleConvLSTM2DCell{T,W,B}) where {T,W,B} = (; Wxh=m.Wxh, Whh=m.Whh, b=m.b)



function (m::Flux.Recur{<:SimpleConvLSTM2DCell})(x)
  m.state, y = m.cell(m.state, x)
  return y
end


function (m::Flux.Recur{<:SimpleConvLSTM2DCell})(x::AbstractArray{T, 5}) where {T}
  h = [m(x_t) for x_t in eachslice(x; dims=5)]
  sze = size(h[1])
  reshape(reduce(_catn, h), sze..., length(h))
end


Flux.Recur(m::SimpleConvLSTM2DCell) = Flux.Recur(m, m.state0)
SimpleConvLSTM2D(a...; kw...) = Flux.Recur(SimpleConvLSTM2DCell(a...; kw...))
Flux._show_leaflike(::SimpleConvLSTM2DCell) = true


function _print_convlstm2d_options(io::IO, l)
  all(==(0), l.Wxh.pad) || print(io, ", pad=", Flux._maybetuple_string(l.Wxh.pad))
  all(==(1), l.Wxh.stride) || print(io, ", stride=", Flux._maybetuple_string(l.Wxh.stride))
  all(==(1), l.Wxh.dilation) || print(io, ", dilation=", Flux._maybetuple_string(l.Wxh.dilation))
  (l.b === false) && print(io, ", bias=false")
end

function Base.show(io::IO, l::SimpleConvLSTM2DCell)
  print(io, "SimpleConvLSTM2DCell(")
  print(io, size(l.Wxh.weight)[1:ndims(l.Wxh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wxh), " => ", Flux._channels_in(l.Whh))
  _print_convlstm2d_options(io, l)
  print(io, ")")
end