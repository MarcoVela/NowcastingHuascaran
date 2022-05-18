module MyLosses
  import Statistics
  using Flux.Losses
  function csi(ŷ, y; agg = sum)
    _one = one(eltype(y))
    intersection = agg(ŷ .* y)
    union = agg(ŷ) + agg(y) - intersection
    (intersection .+ _one) ./ (union .+ _one)
  end

  function negcsi(args...; kwargs...) 
    c = csi(args...; kwargs...)
    one(eltype(c)) - c
  end
end

function get_metric(s::Symbol)
  getproperty(MyLosses, s)
end