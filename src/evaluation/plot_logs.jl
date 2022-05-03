using Plots


function plot_logs!(log_records, metrics; prefix)
  (; first_log, test_logs, train_logs) = log_records
  initial_time = first(train_logs).date
  train_x = @. getfield(getfield(train_logs, :date) - initial_time, :value) / (60*1000)
  t_train_loss = [x.payload.train_loss for x in train_logs]
  t_test_loss = [x.payload.test_loss for x in train_logs]
  test_metrics = Dict([
    (metric, [getfield(x.payload, metric) for x in test_logs])
    for metric in metrics
  ])
  prefix = isempty(string(prefix)) ? first_log.payload.architecture : string(prefix)
  plot!(train_x, t_train_loss; label="$(prefix)_train_loss")
  plot!(train_x, t_test_loss; label="$(prefix)_test_loss")
  test_x = @. getfield(getfield(test_logs, :date) - initial_time, :value) / (60*1000)
  for (metric, values) in test_metrics
    plot!(test_x, values; marker=:circle, label="$(prefix)_test_$(metric)")
  end
  vline!(test_x; linestyle=:dash, label="$(prefix)_epochs")
end

function plot_logs(log_records, metrics = Symbol[]; prefix = "")
  plot(; size=(900, 500))
  plot_logs!(log_records, metrics; prefix)
end
