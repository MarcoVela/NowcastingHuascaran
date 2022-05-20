using Flux

include("utils.jl")

struct ConvLSTM2Dv2Cell{W1, W2, S}
  Wi::W1
  Wh::W2
  state0::S
end

function Flux.gate(x::AbstractArray{T, N}, h, n) where {T, N}
  selectdim(x, N-1, Flux.gate(h,n))
end

function ConvLSTM2Dv2Cell(
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
  state0 = (h, c)
  ConvLSTM2Dv2Cell(Wi, Wh, state0)
end

function (m::ConvLSTM2Dv2Cell)((h, c), x::AbstractArray{T, 4}) where {T}
  gates = m.Wi(x) .+ m.Wh(h)
  input, forget, cell, output = Flux.multigate(gates, size(c, 3), Val(4))
  c′ = @. sigmoid_fast(forget) * c + sigmoid_fast(input) * tanh_fast(cell)
  h′ = @. sigmoid_fast(output) * tanh_fast(c′)
  (h′, c′), h′
end

Flux.@functor ConvLSTM2Dv2Cell (Wi, Wh)


function (m::Flux.Recur{<:ConvLSTM2Dv2Cell})(x::AbstractArray{T, 5}) where {T}
  h = [m(x_t) for x_t in Flux.eachlastdim(x)]
  reshape(reduce(ncat, h), size(h[1])..., :)
end

Flux.Recur(m::ConvLSTM2Dv2Cell) = Flux.Recur(m, m.state0)
ConvLSTM2Dv2(a...; kw...) = Flux.Recur(ConvLSTM2Dv2Cell(a...; kw...))

function _print_convlstm2d_options(io::IO, l)
  all(==(0), l.Wi.pad) || print(io, ", pad=", Flux._maybetuple_string(l.Wi.pad))
  all(==(1), l.Wi.stride) || print(io, ", stride=", Flux._maybetuple_string(l.Wi.stride))
  all(==(1), l.Wi.dilation) || print(io, ", dilation=", Flux._maybetuple_string(l.Wi.dilation))
  (l.Wi.bias === false) && print(io, ", bias=false")
end

function Base.show(io::IO, l::ConvLSTM2Dv2Cell)
  print(io, "ConvLSTM2Dv2Cell(")
  print(io, size(l.Wi.weight)[1:ndims(l.Wi.weight)-2])
  print(io, ", ", size(l.Wh.weight)[1:ndims(l.Wh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wi), " => ", Flux._channels_in(l.Wh))
  _print_convlstm2d_options(io, l)
  print(io, ")")
end
