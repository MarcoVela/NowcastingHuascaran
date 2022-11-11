using Statistics
using OrderedCollections


function csi(y_pred, y)
  intersection = sum(y_pred .* y)
  (intersection + one(eltype(intersection)))/(sum(y_pred) + sum(y) - intersection + one(eltype(intersection)))
end

function true_positive(y_pred, y)
  return sum(y_pred .* y)
end

function true_negative(y_pred, y)
  return sum((1 .- y_pred) .* (1 .- y))
end

function false_positive(y_pred, y)
  sum((y_pred) .* (1 .- y))
end

function false_negative(y_pred, y)
  sum((1 .- y_pred) .* (y))
end

function confmatrix(y_pred, y, thresholds=.05:.05:.95)
  OrderedDict(
    :thresholds => thresholds,
    :TN => [true_negative(y_pred .> t, y) for t in thresholds],
    :TP => [true_positive(y_pred .> t, y) for t in thresholds],
    :FN => [false_negative(y_pred .> t, y) for t in thresholds],
    :FP => [false_positive(y_pred .> t, y) for t in thresholds],
  )
end