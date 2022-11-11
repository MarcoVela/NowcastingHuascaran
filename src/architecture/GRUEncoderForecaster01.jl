using Flux
using Flux: params
using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvGRUv1.jl"))

function build_model(; out, device)
  _model = Chain(
    KeepLast(
      ConvGRUv1((64, 64), (5, 5), (5, 5), 1=>32, pad=SamePad()),
    ),
    RepeatInput(
      out,
      ConvGRUv1((64, 64), (5, 5), (5, 5), 32=>32, pad=SamePad()),
    ),
    TimeDistributed(
      Chain(
        Conv((1, 1), 32 => 1, pad=SamePad(), sigmoid),
      ),
    ),
  )
  model = device(_model)
  model, params(model)
end

