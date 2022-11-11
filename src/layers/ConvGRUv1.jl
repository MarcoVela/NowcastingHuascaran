using Flux

struct ConvGRUv1Cell{A1,A2,S}
  Wi::A1
  Wh::A2
  state0::S
end
include("utils.jl")

function ConvGRUv1Cell(
  input_shape::NTuple{2, <:Integer},
  filter_xh::NTuple{2, <:Integer},
  filter_hh::NTuple{2, <:Integer},
  (in, out)::Pair;
  init = Flux.glorot_uniform,
  init_state = Flux.zeros32,
  pad = 0,
  stride = 1,
  dilation = 1,
  bias = true,
)
  Wi = Conv(filter_xh, in => out * 3; init, pad, stride, bias, dilation)
  Wh = Conv(filter_hh, out => out * 3; init, pad=SamePad(), bias=false)

  output_shape = calc_out_dims(input_shape, Wi.pad, filter_xh, Wi.stride)
  state0 = init_state(output_shape..., out, 1)

  ConvGRUv1Cell(Wi, Wh, state0)
end

function (m::ConvGRUv1Cell)(h, x::AbstractArray{T, 4}) where {T}
  o = size(h, ndims(h) - 1)
  gxs = Flux.multigate(m.Wi(x), o, Val(3))
  ghs = Flux.multigate(m.Wh(h), o, Val(3))
  r = @. sigmoid_fast(gxs[1] + ghs[1])
  z = @. sigmoid_fast(gxs[2] + ghs[2])
  h̃ = @. tanh_fast(gxs[3] + r * ghs[3])
  h′ = @. (1 - z) * h̃ + z * h
  return h′, h′
end

Flux.@functor ConvGRUv1Cell
Flux.trainable(c::ConvGRUv1Cell) = (Wi=c.Wi, Wh=c.Wh)

function (m::Flux.Recur{<:ConvGRUv1Cell})(x::AbstractArray{T, 5}) where {T}
  h = [m(x_t) for x_t in Flux.eachlastdim(x)]
  reshape(reduce(ncat, h), size(h[1])..., :)
end

Flux.Recur(m::ConvGRUv1Cell) = Flux.Recur(m, m.state0)
ConvGRUv1(a...; kw...) = Flux.Recur(ConvGRUv1Cell(a...; kw...))

function _print_convgru2d_options(io::IO, l)
  all(==(0), l.Wi.pad) || print(io, ", pad=", Flux._maybetuple_string(l.Wi.pad))
  all(==(1), l.Wi.stride) || print(io, ", stride=", Flux._maybetuple_string(l.Wi.stride))
  all(==(1), l.Wi.dilation) || print(io, ", dilation=", Flux._maybetuple_string(l.Wi.dilation))
  (l.Wi.bias === false) && print(io, ", bias=false")
end

function Base.show(io::IO, l::ConvGRUv1Cell)
  print(io, "ConvGRUv1Cell(")
  print(io, size(l.Wi.weight)[1:ndims(l.Wi.weight)-2])
  print(io, ", ", size(l.Wh.weight)[1:ndims(l.Wh.weight)-2])
  print(io, ", ", Flux._channels_in(l.Wi), " => ", Flux._channels_in(l.Wh))
  _print_convgru2d_options(io, l)
  print(io, ")")
end


