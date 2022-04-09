using Flux

# ConvLSTM



struct ConvLSTMCell{N,M,F,A,V,S, H}
  Wxi::Conv{N,M,F,A,V}
  Whi::Conv{N,M,F,A,Flux.Zeros}
  Wxf::Conv{N,M,F,A,V}
  Whf::Conv{N,M,F,A,Flux.Zeros}
  Wxc::Conv{N,M,F,A,V}
  Whc::Conv{N,M,F,A,Flux.Zeros}
  Wxo::Conv{N,M,F,A,V}
  Who::Conv{N,M,F,A,Flux.Zeros}

  Wci::S
  Wcf::S
  Wco::S
  state0::Tuple{H, H}
end

Flux.@functor ConvLSTMCell

function ConvLSTMCell(k::NTuple{N,Integer}, ch::Pair{<:Integer,<:Integer}, σ = sigmoid;
        init = Flux.glorot_uniform, pad = 0, init_state = Flux.zeros32) where N
  hidden = last(ch)
  ch1 = ch
  ch2 = hidden => hidden
  Wxi = Flux.Conv(k, ch1, σ; init=init, stride=1, pad=pad, bias=true)
  Whi = Flux.Conv(k, ch2, σ; init=init, stride=1, pad=pad, bias=false)
  Wxf = Flux.Conv(k, ch1, σ; init=init, stride=1, pad=pad, bias=true)
  Whf = Flux.Conv(k, ch2, σ; init=init, stride=1, pad=pad, bias=false)
  Wxc = Flux.Conv(k, ch1, σ; init=init, stride=1, pad=pad, bias=true)
  Whc = Flux.Conv(k, ch2, σ; init=init, stride=1, pad=pad, bias=false)
  Wxo = Flux.Conv(k, ch1, σ; init=init, stride=1, pad=pad, bias=true)
  Who = Flux.Conv(k, ch2, σ; init=init, stride=1, pad=pad, bias=false)

  w,h = k
  Wci = init_state(w, h, hidden, 1)
  Wcf = init_state(w, h, hidden, 1)
  Wco = init_state(w, h, hidden, 1)

  state0 = init_state(w, h, hidden, 1)
  state1 = init_state(w, h, hidden, 1)
  ConvLSTMCell(Wxi, Whi, Wxf, Whf, Wxc, Whc, Wxo, Who, Wci, Wcf, Wco, (state0, state1))
end


Flux.Recur(m::ConvLSTMCell) = Flux.Recur(m, m.state0)
ConvLSTM(a...; ka...) = Flux.Recur(ConvLSTMCell(a...; ka...))

function (m::ConvLSTMCell)((h, c), x::Array{N, T}) where {N, T}
  input = σ.(m.Wxi(x) .+ m.Whi(h) .+ m.Wci .* c)
  forget = σ.(m.Wxf(x) .+ m.Whf(h) .+ m.Wcf .* c)
  c = forget .* c .+ input .* tanh.(m.Wxc(x) .+ m.Whc(h))
  output = σ.(m.Wxo(x) .+ m.Who(h) .+ m.Wco .* c)
  h′ = output .* tanh.(c)
  sz = size(x)
  return (h′, c), reshape(h′, :, sz[2:end]...)
end

