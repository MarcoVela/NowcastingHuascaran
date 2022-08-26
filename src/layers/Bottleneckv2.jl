using Flux

struct Bottleneck{S, C, B}
  shortcut::Val{S}
  c1::C
  bn1::B
  c2::C
  bn2::B
end

function Bottleneck((cin, cout)::Pair{<:Integer, <:Integer}; shortcut::Bool=true, groups=1, e::Rational=1//2, activation=Flux.swish)
  c_ = Int(cout*e)
  c1 = Conv((1, 1), cin => c_, pad=SamePad(), groups=groups, bias=false)
  bn1 = BatchNorm(c_, activation)
  c2 = Conv((3, 3), c_ => cout, pad=SamePad(), bias=false)
  bn2 = BatchNorm(cout, activation)
  Bottleneck(Val(shortcut & (cin==cout)), c1, bn1, c2, bn2)
end

function (m::Bottleneck{S})(x::AbstractArray) where {S}
  if S
    x .+ m.bn2(m.c2(m.bn1(m.c1(x))))
  else
    m.bn2(m.c2(m.bn1(m.c1(x))))
  end
end

Flux.@functor Bottleneck
Flux.trainable(b::Bottleneck) = (c1=b.c1, bn1=b.bn1, c2=b.c2, bn2=b.bn2)

