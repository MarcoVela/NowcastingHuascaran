using Flux
using Flux: params
using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvLSTM2Dv3.jl"))

function build_model(; out, device)
  _model = Chain(
    TimeDistributed(
      Chain(
        Conv((5, 5), 1  =>64, leakyrelu, pad=SamePad(), stride=2),
        Conv((5, 5), 64=>128, leakyrelu, pad=SamePad(), stride=2),
      ),
    ),
    KeepLast(
      ConvLSTM2Dv3((16, 16), (5, 5), (5, 5), 128=>256, pad=SamePad(), stride=2),
    ),
    RepeatInput(
      out,
      ConvLSTM2Dv3((8, 8), (5, 5), (5, 5), 256=>128, pad=SamePad()),
    ),
    TimeDistributed(
      Chain(
        ConvTranspose((5, 5), 128=> 64, pad=SamePad(), stride=2),
        ConvTranspose((5, 5), 64 => 64, pad=SamePad(), stride=2),
        ConvTranspose((5, 5), 64 => 1, pad=SamePad(), leakyrelu, stride=2),
      ),
    ),
  )
  model = device(_model)
  model, params(model)
end

