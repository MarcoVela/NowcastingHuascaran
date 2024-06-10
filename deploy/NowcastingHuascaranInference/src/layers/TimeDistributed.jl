using Flux

include("utils.jl")

struct TimeDistributed{M}
  m::M
end

function (t::TimeDistributed)(x::AbstractArray{T, N}) where {T,N}
  h = [t.m(x_t) for x_t in Flux.eachlastdim(x)]
  reshape(reduce(ncat, h), size(h[1])..., :)
end

Flux.@functor TimeDistributed

Base.show(io::IO, m::TimeDistributed) = print(io,"TimeDistributed(", m.m, ")")

function Base.show(io::IO, ::MIME"text/plain", x::TimeDistributed)
  if get(io, :typeinfo, nothing) === nothing  # e.g. top level in REPL
    Flux._big_show(io, x)
  elseif !get(io, :compact, false)  # e.g. printed inside a Vector, but not a Matrix
    Flux._layer_show(io, x)
  else
    show(io, x)
  end
end