using Flux
using Flux: params

using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvLSTM2Dv3.jl"))


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
    Seq2Seq(
      Encoder(
        ConvLSTM2Dv3((64, 64), (5, 5), (5, 5), 1=>32, pad=SamePad()),
      ),
      Decoder(
        ConvLSTM2Dv3Cell((64, 64), (5, 5), (5, 5), 32=>32, pad=SamePad())
      ),
      out,
    ),
    TimeDistributed(
      Conv((1, 1), 32 => 1, sigmoid, bias=false),
    ),
  )
  model = device(_model)
  model, params(model)
end

