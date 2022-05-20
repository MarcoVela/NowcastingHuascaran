using Flux
include("utils.jl")
struct RepeatInput{N, M}
  n::N
  m::M
end

function (r::RepeatInput)(x::AbstractArray{T, N}) where {T, N}
  h = [r.m(x) for _ in 1:r.n]
  reshape(reduce(ncat, h), size(h[1])..., :)
end

function Base.show(io::IO, ::MIME"text/plain", x::RepeatInput)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    Flux._big_show(io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    Flux._layer_show(io, x)
  else
    show(io, x)
  end
end

Base.show(io::IO, r::RepeatInput) = print(io, "RepeatInput(", r.n, ", ", r.m, ")")


Flux.@functor RepeatInput (m,)
