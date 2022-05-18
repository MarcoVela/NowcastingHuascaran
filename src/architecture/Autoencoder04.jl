using Flux
function build_model(; device, kwargs...)
  lrelu = Base.Fix2(leakyrelu, 0.2f0)
  _model = Chain(
    Chain(
      Conv((3,3), 1=>8, lrelu, pad=SamePad()),
      Conv((3,3), 8=>8, lrelu, pad=SamePad()),
      Conv((3,3), 8=>8, lrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(8, track_stats=true),
    
      Conv((3,3),  8=>16, lrelu, pad=SamePad()),
      Conv((3,3), 16=>16, lrelu, pad=SamePad()),
      Conv((3,3), 16=>16, lrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(16, track_stats=true),
  
      Conv((3,3), 16=>32, lrelu, pad=SamePad()),
      Conv((3,3), 32=>32, lrelu, pad=SamePad(), stride=2),
      Flux.BatchNorm(32, track_stats=true),
    ),
    Chain(
      ConvTranspose((7,7), 32=>32, lrelu, pad=SamePad()),
      ConvTranspose((7,7), 32=>16, lrelu, pad=SamePad(), stride=2),
  
      ConvTranspose((7,7), 16=>16, lrelu, pad=SamePad()),
      ConvTranspose((7,7), 16=> 8, lrelu, pad=SamePad(), stride=2),
    
      ConvTranspose((7,7), 8=>8, lrelu, pad=SamePad()),
      ConvTranspose((7,7), 8=>1, σ, pad=SamePad(), stride=2),
    )
  )
  model = device(_model)
  model, params(model)
end