using Flux

include("Bottleneck.jl")

struct C3{F,C,B,M}
  σ::F
  c1::C
  bn1::B
  c2::C
  bn2::B
  c3::C
  bn3::B
  m::M
end

function C3((cin, cout)::Pair{<:Integer, <:Integer}; n=1, shortcut=true, groups=1, e=1//2, activation=Flux.swish)
  c_ = Int(cout * e)
  c1 = Conv((1, 1), cin=>c_, bias=false, pad=SamePad())
  bn1 = BatchNorm(c_)
  c2 = Conv((1, 1), cin=>c_, bias=false, pad=SamePad())
  bn2 = BatchNorm(c_)
  c3 = Conv((1, 1), c_*2=>cout, bias=false, pad=SamePad())
  bn3 = BatchNorm(cout)
  m = Chain([Bottleneck(c_ => c_, shortcut=shortcut, groups=groups, e=1//1, activation=activation) for _ in 1:n]...)
  C3(activation, c1, bn1, c2, bn2, c3, bn3, m)
end

function (c::C3)(x::AbstractArray)
  c1 = c.m(c.σ.(c.bn1(c.c1(x))))
  c2 = c.σ.(c.bn2(c.c2(x)))
  out1 = cat(c1, c2; dims=Val(3))
  c.σ.(c.bn3(c.c3(out1)))
end

Flux.@functor C3
Flux.trainable(c::C3) = (c1=c.c1, bn1=c.bn1, c2=c.c2, bn2=c.bn2, c3=c.c3, bn3=c.bn3, m=c.m)
