module MyLosses
  import Statistics
  using Flux.Losses
  using MLJBase: f1score
  using Suppressor
  using OrderedCollections

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
    -log1p(csi(args...; kwargs...))
  end

  function csit(ŷ, y)
    ndim = ndims(y)
    n = size(y, ndim)
    scores = collect(csi(a,b) for (a,b) in zip(eachslice(ŷ; dims=ndim), eachslice(y; dims=ndim)))
    OrderedDict(1:n .=> scores)
  end

  function f1(ŷ, y, thresholds=.05:.05:.95)
    @suppress begin
      y = y .> 0.9
      val, threshold = findmax(t -> f1score(ŷ .> t, y), thresholds)
      return (; score=val, threshold=threshold)
    end
  end

  function f1_threshold(ŷ, y, thresholds=.05:.05:.95)
    @suppress begin
      y = y .> 0.9
      Dict(thresholds .=> [f1score(ŷ .> t, y) for t in thresholds])
    end
  end
end

function get_metric(s::Symbol)
  getproperty(MyLosses, s)
end