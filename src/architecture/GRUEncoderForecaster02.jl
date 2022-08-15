using Flux
using Flux: params
using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvGRUv1.jl"))

function build_model(; out, device)
  _model = Chain(
    TimeDistributed(
      Chain(
        Conv((3, 3), 1 => 64, leakyrelu, pad=SamePad()),
        Conv((3, 3), 64 => 64, leakyrelu, pad=SamePad()),
        Conv((3, 3), 64 => 64, leakyrelu, pad=SamePad(), stride=2),
      )
    ),
    KeepLast(
      ConvGRUv1((32, 32), (3, 3), (3, 3), 64=>64, pad=SamePad()),
    ),
    RepeatInput(
      out,
      ConvGRUv1((32, 32), (3, 3), (3, 3), 64=>64, pad=SamePad()),
    ),
    TimeDistributed(
      Chain(
        ConvTranspose((3, 3), 64 => 64, pad=SamePad(), stride=2),
        Conv((3, 3), 64 => 64, pad=SamePad()),
        Conv((3, 3), 64 => 1, pad=SamePad(), sigmoid, bias=false),
      ),
    ),
  )
  model = device(_model)
  model, params(model)
end

