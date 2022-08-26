using Flux
using Flux: params

using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvLSTM2Dv2.jl"))
include(srcdir("layers", "C3v2.jl"))



struct Encoder{T}
  m::T
end

struct Decoder{T}
  m::T
end

function (e::Encoder)(x::AbstractArray)
  out = e.m(x)
  e.m.state, out
end

mutable struct RecurDecoder{T, T2, N}
  m::T
  x::AbstractArray{T2, N}
end

function (r::RecurDecoder{T, T2, N})() where {T, T2, N}
  inds_before = ntuple(_ -> :, N-1)
  r.x = view(r.m(r.x), inds_before..., 1:1)
end

function (d::Decoder)(x_start, steps)
  r = RecurDecoder(d.m, x_start)
  out = [r() for _ in 1:steps]
  reshape(reduce(ncat, out), size(out[1])[1:end-1]..., :)
end

Flux.@functor Encoder
Flux.@functor Decoder
Flux.@functor RecurDecoder

struct Seq2Seq{S1}
  s::S1
  t::Int
end

function (s::Seq2Seq)(x::AbstractArray{T,N}) where {T,N}
  s.s(x)
  n = size(x, N)
  inds_before = ntuple(_ -> :, N-1)
  Decoder(s.s)(view(x, inds_before..., n:n), s.t)
end

Flux.@functor Seq2Seq

struct ShortCircuit{S1,M}
  s1::S1
  m::M
end

function (s::ShortCircuit)(x::AbstractArray{T, N}) where {T,N}
  out = s.s1(x)
  out2 = s.m(out)
  s.s1(out2)
end

Flux.@functor ShortCircuit

function build_model(; out, device, dropout=.25)
  _model = Chain(
    TimeDistributed(
      Chain(
        Conv((3, 3), 1 => 64, pad=SamePad(), stride=2, bias=false),
        BatchNorm(64, Flux.swish),
        C3(64 => 64, n=3),
      ),
    ),
    Dropout(dropout; dims=3),
    ShortCircuit(
      ConvLSTM2Dv2((32, 32), (3, 3), (3, 3), 64=>64, pad=SamePad(), bias=false),
      Chain(
        TimeDistributed(
          Chain(
            Conv((3, 3), 64 => 128, pad=SamePad(), stride=2, bias=false),
            BatchNorm(128, Flux.swish),
            C3(128 => 128, n=3),
            Dropout(dropout; dims=3),
          ),
        ),
        Seq2Seq(
          ConvLSTM2Dv2((16, 16), (3, 3), (3, 3), 128=>128, pad=SamePad(), bias=false),
          out,
        ),
        TimeDistributed(
          Chain(
            Upsample(2, :bilinear),
            C3(128 => 64, n=3),
            Conv((1, 1), 64 => 64, pad=SamePad(), bias=false),
            BatchNorm(64, Flux.swish),
            Dropout(dropout; dims=3),
          ),
        ),
      ),
    ),
    TimeDistributed(
      Chain(
        Upsample(2, :bilinear),
        C3(64 => 32, n=3),
        Conv((1, 1), 32 => 32, pad=SamePad(), bias=false),
        BatchNorm(32, Flux.swish),
        Conv((1, 1), 32 => 1, sigmoid, pad=SamePad(), bias=false),
      ),
    ),
  )
  model = device(_model)
  model, params(model)
end

