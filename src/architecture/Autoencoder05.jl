using Flux
function build_model(; device, kwargs...)
  _model = Chain(
    Chain(
      Conv((3,3), 1=>4, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(4, track_stats=true),
      Conv((3,3), 4=>8, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(8, track_stats=true),
      Conv((3,3), 8=>16, leakyrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(16, track_stats=true),
    ),
    Chain(
      ConvTranspose((7,7), 16=>8, leakyrelu, pad=SamePad(), stride=2),
      ConvTranspose((7,7), 8=> 4, leakyrelu, pad=SamePad(), stride=2),
      ConvTranspose((7,7), 4=>1, σ, pad=SamePad(), stride=2),
    )
  )
  model = device(_model)
  model, params(model)
end