using Flux

struct Bottleneck{S, C}
  shortcut::Val{S}
  expansion::C
  projection::C
end

function Bottleneck((cin, cout)::Pair{<:Integer, <:Integer}; shortcut::Bool=true, groups=1, e::Union{Rational,Integer}=1//2, activation=Flux.swish)
  c_ = Int(cout*e)
  c1 = Conv((1, 1), cin => c_, pad=SamePad(), groups=groups, bias=false)
  bn1 = BatchNorm(c_, activation)
  c2 = Conv((3, 3), c_ => cout, pad=SamePad(), bias=false)
  bn2 = BatchNorm(cout, activation)
  Bottleneck(Val(shortcut & (cin==cout)), Chain(c1, bn1), Chain(c2, bn2))
end

function (m::Bottleneck{S})(x::AbstractArray) where {S}
  if S
    x .+ m.projection(m.expansion(x))
  else
    m.projection(m.expansion(x))
  end
end

Flux.@functor Bottleneck
Flux.trainable(b::Bottleneck) = (projection=b.projection, expansion=b.expansion)

