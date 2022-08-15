using Flux
using Flux: params

using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))


struct Encoder{T}
  m::T
end

struct Decoder{T}
  m::T
end

mutable struct StateRecur{M, S}
  m::M
  state::S
end

function (r::StateRecur)()
  r.state, h = r.m(r.state, r.state[1])
  h
end

function (e::Encoder)(x::AbstractArray)
  e.m(x)
  e.m.state
end

function _get_hidden_state(m, input)
  m(input)
  m.state[2]
end

function (d::Decoder)(state, steps)
  r = StateRecur(d.m, state)
  out = [r() for _ in 1:steps]
  reshape(reduce(ncat, out), size(out[1])..., :)
end

Flux.@functor StateRecur
Flux.@functor Encoder
Flux.@functor Decoder

struct Seq2Seq{D,E}
  e::E
  d::D
  t::Int
end

function (s::Seq2Seq)(x)
  state = s.e(x)
  s.d(state, s.t)
end

Flux.@functor Seq2Seq

function build_model(; out, device)
  _model = Chain(
    TimeDistributed(
      Chain(
        Conv((3, 3), 1  => 64, leakyrelu, stride=2),
        Conv((3, 3), 64 => 64, leakyrelu, stride=2),
        Conv((3, 3), 64 => 64, leakyrelu, stride=2),
        Flux.flatten,
      )
    ),
    Seq2Seq(
      Encoder(
        LSTM(64*29*29),
      ),
      Decoder(
        ConvLSTM2Dv3Cell((32, 32), (5, 5), (5, 5), 64=>64, pad=SamePad())
      ),
      out,
    ),
    TimeDistributed(
      Chain(
        ConvTranspose((5, 5), 64 => 64, leakyrelu, pad=SamePad(), stride=2),
        ConvTranspose((1, 1), 64 => 1, sigmoid, bias=false, pad=SamePad()),
      )
    ),
  )
  model = device(_model)
  model, params(model)
end

