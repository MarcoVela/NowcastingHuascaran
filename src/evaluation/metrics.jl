using Statistics

function csi_framewise(y_pred, y; threshold = 0.5)
  cortados = y_pred .> threshold
  cort_y = y .> .9
  a = sum(cortados .== cort_y .== 1; dims=(1,2))[:]
  b = sum((cortados .== 1) .& (cort_y .== 0); dims=(1,2))[:]
  c = sum((cortados .== 0) .& (cort_y .== 1); dims=(1,2))[:]
  # d = sum(cortados .== cort_y .== 0; dims=(1,2))[:]
  @. a / (a + b + c)
end

function accuracy_framewise(y_pred, y; threshold = 0.5)
  cortados = y_pred .> threshold
  cort_y = y .> .9
  a = sum(cortados .== cort_y .== 1; dims=(1,2))[:]
  b = sum((cortados .== 1) .& (cort_y .== 0); dims=(1,2))[:]
  c = sum((cortados .== 0) .& (cort_y .== 1); dims=(1,2))[:]
  d = sum(cortados .== cort_y .== 0; dims=(1,2))[:]
  @. (a + d) / (a + b + c + d)
end

function metric_batch(metric, y_pred_batch, y_batch, threshold)
  n = size(y_batch, 4)
  res = zeros(n)
  for (y_pred, y) in zip(eachslice(y_pred_batch; dims=3), eachslice(y_batch; dims=3))
    res .+= metric(y_pred, y; threshold)
  end
  res ./ n
end

csi_batch(y_pred_batch, y_batch, threshold) = metric_batch(csi_framewise, y_pred_batch, y_batch, threshold)

accuracy_batch(y_pred_batch, y_batch, threshold) = metric_batch(accuracy_framewise, y_pred_batch, y_batch, threshold)

function csi(y_pred::AbstractArray{T}, y::AbstractArray{T}) where {T}
  a = sum(y_pred .== y .== one(T))
  b = sum((y_pred .== one(T)) .& (y .== zero(T)))
  c = sum((y_pred .== zero(T)) .& (y .== one(T)))
  a / (a + b + c)
end