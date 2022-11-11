using Flux
using DrWatson

function build_model(; device, kwargs...)
  _model = Chain(
    Conv((3, 3, 3), 1=>64, leakyrelu, pad=SamePad()),
    Flux.BatchNorm(64, track_stats=true),
    Conv((3, 3, 3), 64=>64, leakyrelu, pad=SamePad()),
    Flux.BatchNorm(64, track_stats=true),
    Conv((3, 3, 3), 64=>1 , sigmoid, pad=SamePad()),
  )
  model = device(_model)
  model, params(model)
end