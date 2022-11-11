using Flux

include("Bottleneckv3.jl")

# https://arxiv.org/abs/1911.11929
struct C3{D,S,T}
  dense::D
  shortcut::S
  transition::T
end

function C3((cin, cout)::Pair{<:Integer, <:Integer}; n=1, shortcut=true, groups=1, e=1//2, activation=Flux.swish)
  c_ = Int(cout * e)
  c1 = Conv((1, 1), cin=>c_, bias=false, pad=SamePad())
  bn1 = BatchNorm(c_, activation)
  c2 = Conv((1, 1), cin=>c_, bias=false, pad=SamePad())
  bn2 = BatchNorm(c_, activation)
  c3 = Conv((1, 1), c_*2=>cout, bias=false, pad=SamePad())
  bn3 = BatchNorm(cout, activation)
  dense = Chain(c1, bn1, [Bottleneck(c_ => c_, shortcut=shortcut, groups=groups, e=1//1, activation=activation) for _ in 1:n]...)
  shortcut = Chain(c2, bn2)
  transition = Chain(c3, bn3)
  C3(dense, shortcut, transition)
end

function (c::C3)(x::AbstractArray)
  c1 = c.dense(x)
  c2 = c.shortcut(x)
  out1 = cat(c1, c2; dims=Val(3))
  c.transition(out1)
end

Flux.@functor C3
Flux.trainable(c::C3) = (dense=c.dense, shortcut=c.shortcut, transition=c.transition)
