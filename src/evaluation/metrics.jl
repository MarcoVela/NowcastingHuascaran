using Statistics


function csi(y_pred, y)
  intersection = sum(y_pred .* y)
  (intersection + one(eltype(intersection)))/(sum(y_pred) + sum(y) - intersection + one(eltype(intersection)))
end