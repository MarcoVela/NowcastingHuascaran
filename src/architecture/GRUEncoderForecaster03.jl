using Flux
using Flux: params
using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvGRUv1.jl"))

struct Encoder{T}
  m::T
end

struct Decoder{T}
  m::T
end

function (e::Encoder)(x::AbstractArray)
  e.m(x)
  e.m.state
end

function _get_hidden_state(m, input)
  m(input)
  m.state[2]
end

mutable struct RecurDecoder{T, T2, N}
  m::T
  x::AbstractArray{T2, N}
end

function (r::RecurDecoder{T, T2, N})() where {T, T2, N}
  inds_before = ntuple(_ -> :, N-1)
  r.x = view(r.m(r.x), inds_before..., 1:1)
end

function (d::Decoder)(state, x_start, steps)
  d.m.state = state
  r = RecurDecoder(d.m, x_start)
  out = [r() for _ in 1:steps]
  reshape(reduce(ncat, out), size(out[1])[1:end-1]..., :)
end

Flux.@functor Encoder
Flux.@functor Decoder
Flux.@functor RecurDecoder

struct Seq2Seq{D,E}
  e::E
  d::D
  t::Int
end

function (s::Seq2Seq)(x::AbstractArray{T,N}) where {T,N}
  state = s.e(x)
  n = size(x, N)
  inds_before = ntuple(_ -> :, N-1)
  s.d(state, view(x, inds_before..., n:n), s.t)
end

Flux.@functor Seq2Seq
function build_model(; out, device)
  _model = Chain(
    TimeDistributed(
      Chain(
        Conv((3, 3), 1 => 64, leakyrelu, pad=SamePad(), stride=2),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, leakyrelu, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, leakyrelu, pad=SamePad()),
        BatchNorm(64),
      ),
    ),
    Seq2Seq(
      Encoder(
        ConvGRUv1((32, 32), (3, 3), (3, 3), 64=>64, pad=SamePad()),
      ),
      Decoder(
        ConvGRUv1((32, 32), (3, 3), (3, 3), 64=>64, pad=SamePad()),
      ),
      out,
    ),
    TimeDistributed(
      Chain(
        BatchNorm(64),
        ConvTranspose((5, 5), 64 => 64, pad=SamePad(), stride=2),
        BatchNorm(64),
        Conv((3, 3), 64 => 64, pad=SamePad()),
        BatchNorm(64),
        Conv((3, 3), 64 => 1, pad=SamePad(), sigmoid, bias=false),
      ),
    ),
  )
  model = device(_model)
  model, params(model)
end

