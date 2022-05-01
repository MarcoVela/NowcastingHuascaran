function keep_last_samples(n = 1)
  return function keep(x)
    N = length(size(x))
    inds_before = ntuple(Returns(:), N - 1)
    view(x, inds_before..., size(x, N)-n+1:size(x, N))
  end
end

function keep_last_sample(x)
  N = length(size(x))
  inds_before = ntuple(Returns(:), N - 1)
  view(x, inds_before..., size(x, N))
end

function repeat_samples(n)
  return function r(x::Q)::AbstractArray where {Q <: AbstractArray}
    h = map(_ -> copy(x), Base.OneTo(n))
    reduce((a,b) -> cat(a,b; dims=4), h)
  end
end

