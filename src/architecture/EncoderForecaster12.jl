using Flux
using Flux: params

using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvLSTM2Dv2.jl"))
include(srcdir("layers", "C3v3.jl"))



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
  state, _ = s.e(x)
  n = size(x, N)
  inds_before = ntuple(_ -> :, N-1)
  s.d(state, view(x, inds_before..., n:n), s.t)
end

Flux.@functor Seq2Seq

struct SharedState{S1,M,S2}
  s1::S1
  m::M
  s2::S2
end

function (s::SharedState)(x::AbstractArray{T, N}) where {T,N}
  state, out = s.s1(x)
  out2 = s.m(out)
  s.s2.state = state
  s.s2(out2)
end

Flux.@functor SharedState

function build_model(; out, device)
  _model = Chain(
    TimeDistributed(
      Chain(
        C3(1=>16, n=3, activation=Flux.leakyrelu),
        Conv((3, 3), 16=>16, stride=2, pad=SamePad(), bias=false),
        BatchNorm(16, Flux.leakyrelu),
      )
    ),
    SharedState(
      Encoder(ConvLSTM2Dv2((32, 32), (3, 3), (3, 3), 16=>16, pad=SamePad(), bias=false)),
      Chain(
        TimeDistributed(
          Chain(
            C3(16=>32, n=3, activation=Flux.leakyrelu),
            Conv((3, 3), 32=>32, stride=2, pad=SamePad(), bias=false),
            BatchNorm(32, Flux.leakyrelu),
          )
        ),
        SharedState(
          Encoder(ConvLSTM2Dv2((16, 16), (3, 3), (3, 3), 32=>32, pad=SamePad(), bias=false)),
          Chain(
            TimeDistributed(
              Chain(
                C3(32=>64, n=3, activation=Flux.leakyrelu),
                Conv((3, 3), 64=>64, stride=2, pad=SamePad(), bias=false),
                BatchNorm(64, Flux.leakyrelu),
              )
            ),
            Seq2Seq(
              Encoder(ConvLSTM2Dv2((8, 8), (3, 3), (3, 3), 64=>64, pad=SamePad(), bias=false)),
              Decoder(ConvLSTM2Dv2((8, 8), (3, 3), (3, 3), 64=>64, pad=SamePad(), bias=false)),
              out,
            ),
            TimeDistributed(
              Chain(
                ConvTranspose((3, 3), 64=>64, stride=2, pad=SamePad(), bias=false, groups=64),
                BatchNorm(64, Flux.leakyrelu),                
                C3(64=>64, n=3, activation=Flux.leakyrelu),
              )
            ),
          ),
          ConvLSTM2Dv2((16, 16), (3, 3), (3, 3), 64=>32, pad=SamePad(), bias=false),
        ),
        TimeDistributed(
          Chain(
            ConvTranspose((3, 3), 32=>32, stride=2, pad=SamePad(), bias=false, groups=32),
            BatchNorm(32, Flux.leakyrelu),
            C3(32=>32, n=3, activation=Flux.leakyrelu),
          )
        ),
      ),
      ConvLSTM2Dv2((32, 32), (3, 3), (3, 3), 32=>16, pad=SamePad(), bias=false),
    ),
    TimeDistributed(
      Chain(
        ConvTranspose((3, 3), 16=>16, stride=2, pad=SamePad(), bias=false, groups=16),
        BatchNorm(16, Flux.leakyrelu),
        Conv((1, 1), 16=>1, sigmoid, pad=SamePad(), bias=false),
      )
    ),
  )
  model = device(_model)
  model, params(model)
end

