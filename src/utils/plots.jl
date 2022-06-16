using Plots

function plot_results(x, y_pred, y)
  ps = []
  for i = axes(x, 3)
    p = heatmap(x[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    push!(ps, (p, p))
  end
  for i = axes(y_pred, 3)
    p1 = heatmap(y_pred[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    p2 = heatmap(y[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    push!(ps, (p1, p2))
  end
  g = @animate for (p1, p2) in ps
    plot(p1, p2, size=(800, 400))
  end
  gif(g, fps=2)
end


function plot_many(samples, labels=1:length(samples), truth=0; kws...)
  Ti = axes(samples[1], 3)
  maybetitle(l, t) = (t <= truth) ? "Truth" : string(l)
  ps_list = [
    [
      heatmap(sample[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing, title=maybetitle(l, i))
      for (sample, l) in zip(samples, labels)
    ]
    for i in Ti
  ]
  kws = Dict(kws)
  fps = pop!(kws, :fps, 2)
  g = @animate for ps in ps_list
    plot(ps...; kws...)
  end
  gif(g; fps)
end