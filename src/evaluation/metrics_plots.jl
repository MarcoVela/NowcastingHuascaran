using Plots

include("metrics.jl")

function plot_loss_framewise(loss_function, y_pred, y)
  loss = loss_function(y_pred, y; agg=identity)
  framewise = mean(loss; dims=(1,2))[:]
  plot(framewise, ylims=(0, maximum(framewise) * 1.5), legend=false)
end

function plot_csi_framewise(y_pred, y; thresholds=0.1:.15:.9)
  t = size(y, 3)
  ps = map(x -> begin
    d = csi_framewise(y_pred, y; threshold=x)
    label = "umbral: $x - maximo: $(round(maximum(d), digits=2))"
    plot(d, label=label, xticks=1:t, ylims=(0,1))
  end, thresholds)
  plot(ps...; layout=(length(thresholds), 1), size=(800, 900))
end


function plot_acc_framewise(y_pred, y; thresholds=0.1:.15:.9)
  t = size(y, 3)
  ps = map(x -> begin
    d = accuracy_framewise(y_pred, y; threshold=x)
    label = "umbral: $x - máximo: $(round(maximum(d), digits=2))"
    plot(d, label=label, xticks=1:t, ylims=(0,1))
  end, thresholds)
  plot(ps...; layout=(length(thresholds), 1), size=(900, 900))
end
