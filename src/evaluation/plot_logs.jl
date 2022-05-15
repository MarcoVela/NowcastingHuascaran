using Plots


function plot_logs!(log_records, metrics; prefix)
  (; first_log, test_logs, train_logs) = log_records
  initial_time = first(train_logs).date
  train_x = @. getfield(getfield(train_logs, :date) - initial_time, :value) / (60*1000)
  t_train_loss = [x.payload.train_loss for x in train_logs]
  t_test_loss = [x.payload.test_loss for x in train_logs]
  y_max = -Inf
  y_min = Inf
  train_metrics = Dict([
    (metric, [getfield(x.payload, metric) for x in train_logs])
    for metric in metrics
    if hasfield(payloadtype(eltype(train_logs)), metric)
  ])
  test_metrics = Dict([
    (metric, [getfield(x.payload, metric) for x in test_logs])
    for metric in metrics
    if hasfield(payloadtype(eltype(test_logs)), metric)
  ])
  prefix = ismissing(prefix) ? first_log.payload[:architecture][:type] : string(prefix)
  prefix = isempty(prefix) ? "" : "$(prefix)_"
  for (metric, values) in train_metrics
    y_max = max(y_max, maximum(values))
    y_min = min(y_min, minimum(values))
    plot!(train_x, values; label="$(prefix)$(metric)")
  end
  # plot!(train_x, t_train_loss; label="$(prefix)_train_loss")
  # plot!(train_x, t_test_loss; label="$(prefix)_test_loss")
  test_x = @. getfield(getfield(test_logs, :date) - initial_time, :value) / (60*1000)
  for (metric, values) in test_metrics
    y_max = max(y_max, maximum(values))
    y_min = min(y_min, minimum(values))
    plot!(test_x, values; marker=:circle, label="$(prefix)$(metric)")
  end
  y_min = y_min * (1 + -sign(y_min)*.25)
  y_max = y_max * (1 +  sign(y_max)*.25)
  ylims!((y_min, y_max))
  if !iszero(length(test_x))
    vline!(test_x; linestyle=:dash, label="$(prefix)epochs")
  end
end

function plot_logs(log_records, metrics = Symbol[]; prefix = "")
  plot(; size=(900, 500))
  plot_logs!(log_records, metrics; prefix)
end
