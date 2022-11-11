using Flux
function build_model(; device, kwargs...)
  lrelu = Base.Fix2(leakyrelu, 0.2f0)
  _model = Chain(
    Chain(
      Conv((3,3),  1=>64, lrelu, pad=SamePad()),
      Conv((3,3), 64=>64, lrelu, pad=SamePad()),
      Conv((3,3), 64=>64, lrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(64, track_stats=true),
    
      Conv((3,3),  64=>128, lrelu, pad=SamePad()),
      Conv((3,3), 128=>128, lrelu, pad=SamePad()),
      Conv((3,3), 128=>128, lrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(128, track_stats=true),
  
      Conv((3,3), 128=>256, lrelu, pad=SamePad()),
      Conv((3,3), 256=>256, lrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(256, track_stats=true),
    ),
    Chain(
      ConvTranspose((7,7), 256=>256, lrelu, pad=SamePad()),
      ConvTranspose((7,7), 256=>128, lrelu, pad=SamePad(), stride=2),
  
      ConvTranspose((7,7), 128=>128, lrelu, pad=SamePad()),
      ConvTranspose((7,7), 128=> 64, lrelu, pad=SamePad(), stride=2),
    
      ConvTranspose((7,7), 64=>64, lrelu, pad=SamePad()),
      ConvTranspose((7,7), 64=> 1, σ, pad=SamePad(), stride=2),
    )
  )
  model = device(_model)
  model, params(model)
end