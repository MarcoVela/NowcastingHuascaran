using Flux
struct KeepLast{N, M<:Flux.Recur}
  n::N
  m::M
end

function KeepLast(m)
  KeepLast(1, m)
end

function (k::KeepLast)(x::AbstractArray{T, N}) where {T, N}
  before_dims = ntuple(_ -> :, N-1)
  n2 = size(x, N)
  discarted = ifelse(n2 - k.n < 0, 0, n2 - k.n)
  discarted > 0 && k.m(view(x, before_dims..., 1:discarted))
  k.m(view(x, before_dims..., discarted+1:n2))
end

Flux.@functor KeepLast (m,)

Base.show(io::IO, k::KeepLast) = print(io,"KeepLast(", k.m, ")")# Flux._big_show(io, m)

function Base.show(io::IO, ::MIME"text/plain", x::KeepLast)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    Flux._big_show(io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    Flux._layer_show(io, x)
  else
    show(io, x)
  end
end

