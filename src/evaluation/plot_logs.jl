function plot_logs(log_records)
  (; first_log, test_logs, train_logs) = log_records
  train_x = [x.date for x in train_logs]
  train_loss_during_train = [x.payload.train_loss for x in train_logs]
  test_loss_during_train = [x.payload.test_loss for x in train_logs]
  epochs_end = [x.date for x in test_logs]
  plot(train_x, train_loss_during_train; label="train")
  plot!(train_x, test_loss_during_train; label="test")
  vline!(epochs_end; size=(900, 500), label=nothing)
  xticks!(Dates.value.(epochs_end), string.(1:length(epochs_end)))
  # annotate!([(x, 0, text("$i", 8, rotation=45)) for (i,x) in enumerate(epochs_end)])
  # ylims!(-.01, 1.1maximum(vcat(train_loss_during_train, test_loss_during_train)))
end