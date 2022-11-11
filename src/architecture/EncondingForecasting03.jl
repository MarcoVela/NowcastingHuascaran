using Flux
using Flux: params
using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvLSTM2Dv2.jl"))

function build_model(; out, device)
  _model = Chain(
    TimeDistributed(
      Chain(
        Conv((5, 5), 1  =>128, leakyrelu, pad=SamePad(), stride=2),
        Conv((5, 5), 128=>128, leakyrelu, pad=SamePad(), stride=2),
      ),
    ),
    ConvLSTM2Dv2((16, 16), (5, 5), (5, 5),  128=>128, pad=SamePad()),
    KeepLast(
      ConvLSTM2Dv2((16, 16), (5, 5), (5, 5), 128=>64, pad=SamePad()),
    ),
    RepeatInput(
      out,
      ConvLSTM2Dv2((16, 16), (5, 5), (5, 5), 64=>64, pad=SamePad()),
    ),
    TimeDistributed(
      Chain(
        ConvTranspose((5, 5), 64 => 8, pad=SamePad(), stride=2),
        ConvTranspose((5, 5), 8 => 8, pad=SamePad(), stride=2),
        Conv((1,1), 8=>1, pad=SamePad(), sigmoid_fast, bias=false),
      ),
    ),
  )
  model = device(_model)
  model, params(model)
end

