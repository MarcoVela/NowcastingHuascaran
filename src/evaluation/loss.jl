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

  function logcsi(args...; kwargs...) 
    -log(csi(args...; kwargs...))
  end

  function csit(ŷ, y)
    ndim = ndims(y)
    n = size(y, ndim)
    scores = collect(csi(a,b) for (a,b) in zip(eachslice(ŷ; dims=ndim), eachslice(y; dims=ndim)))
    Dict(1:n .=> scores)
  end
end

function get_metric(s::Symbol)
  getproperty(MyLosses, s)
end