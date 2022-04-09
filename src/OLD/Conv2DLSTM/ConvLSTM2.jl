using Flux
using ProtoStructs

struct ConvLSTMCell{A,X,M}
  Wxi::A
  Whi::A
  bi::X

  Wxf::A
  Whf::A
  bf::X

  Wxc::A
  Whc::A
  bc::X

  Wxo::A
  Who::A
  bo::X

  Wci::A
  Wcf::A
  Wco::A

  memory::X
  hidden::X

  pad::NTuple{M,Int}
end

function (m::ConvLSTMCell)((h, c), x::AbstractArray)
  ihdim = Flux.DenseConvDims(x, m.Wxi; padding = m.pad)
  hhdim = Flux.DenseConvDims(h, m.Whi; padding = m.pad)
  i = σ.(conv(x, m.Wxi, ihdim) .+ conv(h, m.Whi, hhdim) .+ m.Wci .* c .+ m.bi)
  f = σ.(conv(x, m.Wxf, ihdim) .+ conv(h, m.Whf, hhdim) .+ m.Wcf .* c .+ m.bf)
  c = f .* c .+ i .* tanh.(conv(x, m.Wxc, ihdim) .+ conv(h, m.Whc, hhdim) .+ m.bc)
  o = σ.(conv(x, m.Wxo, ihdim) .+ conv(h, m.Who, hhdim) .+ m.Wco .* c .+ m.bo)
  h = o .* tanh.(c)
  return (h, c), h
end

function ConvLSTMCell(k, ch, shape; init = Flux.glorot_uniform, initb = Flux.zeros32, init_state = Flux.zeros32)
  cin, chid = ch
  Wxi = init(k..., cin, chid)
  Whi = init(k..., chid, chid)
  bi = initb(shape..., cin, chid)

  Wxf = init(k..., cin, chid)
  Whf = init(k..., chid, chid)
  bf = initb(shape..., cin, chid)

  Wxc = init(k..., cin, chid)
  Whc = init(k..., chid, chid)
  bc = init(shape..., cin, chid)

  Wxo = init(k..., cin, chid)
  Who = init(k..., chid, chid)
  bo = init(shape..., cin, chid)

  Wci = init(shape..., cin, chid)
  Wcf = init(shape..., cin, chid)
  Wco = init(shape..., cin, chid)

  pad = Flux.calc_padding(Conv, SamePad(), k, 1, 1)

  memory = init_state(shape..., chid, chid)
  hidden = init_state(shape..., chid, chid)
  ConvLSTMCell(
    Wxi,
    Whi,
    bi,
    Wxf,
    Whf,
    bf,
    Wxc,
    Whc,
    bc,
    Wxo,
    Who,
    bo,
    Wci,
    Wcf,
    Wco,
    memory,
    hidden,
    pad,
  )
end


Flux.@functor ConvLSTMCell
Flux.Recur(m::ConvLSTMCell) = Flux.Recur(m, (m.hidden, m.memory))
ConvLSTM(a...; ka...) = Flux.Recur(ConvLSTMCell(a...; ka...))
