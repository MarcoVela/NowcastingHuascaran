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

function _calc_out_dims((width, height), pad, filter, dilation, stride)
  padding = Flux.calc_padding(Flux.Conv, pad, filter, dilation, stride)
  out_width = width
  out_height = height
  filter_w, filter_h = filter
  expand_size(p::Number) = ntuple(_ -> Int(p), 2)
  expand_size(p) = tuple(p...)
  stride_w, stride_h = expand_size(stride)
  if length(padding) == 2
    pad_w, pad_h = padding
    out_width = ((width + 2*pad_w - filter_w) ÷ stride_w) + 1
    out_height = ((height + 2*pad_h - filter_h) ÷ stride_h) + 1
  elseif length(padding) == 4
    pad_w_top, pad_w_bot, pad_h_top, pad_h_bot = padding
    out_width = ((width + pad_w_top + pad_w_bot - filter_w) ÷ stride_w) + 1
    out_height = ((height + pad_h_top + pad_h_bot - filter_h) ÷ stride_h) + 1
  end
  out_width, out_height
end

function calc_out_dims(W::NTuple{M}, padding::NTuple{N}, K::NTuple{M}, stride) where {N,M}
  if N == M 
    pad = padding .* 2
  elseif N==M*2
    pad = ntuple(i -> padding[2i-1] + padding[2i], M)
  end
  ((W .+ pad .- K) .÷ stride) .+ 1
end


ncat(A::AbstractArray{T, N}...) where {T, N} = cat(A...; dims=Val(N))

using Flux

function Flux.gate(x::AbstractArray{T, N}, h, n) where {T, N}
  before_dims = ntuple(_ -> :, N-2)
  view(x, before_dims..., Flux.gate(h, n), :)
end
