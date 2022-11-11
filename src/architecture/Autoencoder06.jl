using Flux
function build_model(; device, kwargs...)
  _model = Chain(
    Chain(
      Conv((5, 5), 1=>16, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(16, track_stats=true),
      Flux.Dropout(.1; dims=3),
      Conv((5, 5), 16=>32, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(32, track_stats=true),
      Flux.Dropout(.1; dims=3),
      Conv((5, 5), 32=>64, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(64, track_stats=true),
      Flux.Dropout(.1; dims=3),
    ),
    Chain(
      ConvTranspose((5, 5), 64=>32, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(32, track_stats=true),
      Flux.Dropout(.1; dims=3),
      ConvTranspose((5, 5), 32=>16, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(16, track_stats=true),
      Flux.Dropout(.1; dims=3),
      ConvTranspose((5, 5), 16=>1 , σ, pad=SamePad(), stride=2),
    )
  )
  model = device(_model)
  model, params(model)
end