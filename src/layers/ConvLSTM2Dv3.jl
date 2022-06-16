using Flux
using Flux: gate


include("utils.jl")

struct ConvLSTM2Dv3Cell{W1, W2, W3, S}
  Wi::W1
  Wh::W2
  Wc::W3
  state0::S
end

function Flux.gate(x::AbstractArray{T, N}, h, n) where {T, N}
  before_dims = ntuple(_ -> :, N-2)
  view(x, before_dims..., gate(h,n), :)
end

function ConvLSTM2Dv3Cell(
  input_shape::NTuple{2, <:Integer},
  kxh::NTuple{2, <:Integer},
  khh::NTuple{2, <:Integer},
  ch::Pair{<:Integer, <:Integer};
  init = Flux.glorot_uniform,
  init_state = Flux.zeros32,
  pad = 0,
  stride = 1,
  dilation = 1,
  bias = true)

  ch_in, ch_hid = ch
  ch_in_to_hid = ch_in => ch_hid * 4
  ch_hid_to_hid = ch_hid => ch_hid * 4

  Wi = Conv(kxh, ch_in_to_hid; init, pad, stride, bias, dilation)
  Wh = Conv(khh, ch_hid_to_hid; init, pad=SamePad(), bias=false)

  output_shape = calc_out_dims(input_shape, Wi.pad, kxh, Wi.stride)
  h = init_state(output_shape..., ch_hid, 1)
  c = init_state(output_shape..., ch_hid, 1)
  Wc = init_state(output_shape..., ch_hid * 3, 1)
  state0 = (h, c)
  ConvLSTM2Dv3Cell(Wi, Wh, Wc, state0)
end

function (m::ConvLSTM2Dv3Cell)((h, c), x::AbstractArray{T, 4}) where {T}
  gates = m.Wi(x) .+ m.Wh(h)
  input, forget, cell, output = Flux.multigate(gates, size(c, ndims(c) - 1), Val(4))
  Wci, Wcf, Wco = Flux.multigate(m.Wc, size(c, ndims(c) - 1), Val(3))
  input = @. sigmoid_fast(input + Wci * c)
  forget = @. sigmoid_fast(forget + Wcf * c)
  c′ = @. forget * c + input * tanh_fast(cell)
  output = @. sigmoid_fast(output + Wco * c′)
  h′ = @. output * tanh_fast(c′)
  (h′, c′), h′
end

Flux.@functor ConvLSTM2Dv3Cell
Flux.trainable(c::ConvLSTM2Dv3Cell) = (Wi=c.Wi, Wh=c.Wh, Wc=c.Wc)


function (m::Flux.Recur{<:ConvLSTM2Dv3Cell})(x::AbstractArray{T, 5}) where {T}
  h = [m(x_t) for x_t in Flux.eachlastdim(x)]
  reshape(reduce(ncat, h), size(h[1])..., :)
end

Flux.Recur(m::ConvLSTM2Dv3Cell) = Flux.Recur(m, m.state0)
ConvLSTM2Dv3(a...; kw...) = Flux.Recur(ConvLSTM2Dv3Cell(a...; kw...))
Flux._show_leaflike(::ConvLSTM2Dv3Cell) = true

function _print_convlstm2d_options(io::IO, l)
  all(==(0), l.Wi.pad) || print(io, ", pad=", Flux._maybetuple_string(l.Wi.pad))
  all(==(1), l.Wi.stride) || print(io, ", stride=", Flux._maybetuple_string(l.Wi.stride))
  all(==(1), l.Wi.dilation) || print(io, ", dilation=", Flux._maybetuple_string(l.Wi.dilation))
  (l.Wi.bias === false) && print(io, ", bias=false")
end

function Base.show(io::IO, l::ConvLSTM2Dv3Cell)
  print(io, "ConvLSTM2Dv3Cell(")
  print(io, size(l.Wi.weight)[1:ndims(l.Wi.weight)-2])
  print(io, ", ", size(l.Wh.weight)[1:ndims(l.Wh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wi), " => ", Flux._channels_in(l.Wh))
  _print_convlstm2d_options(io, l)
  print(io, ")")
end
