using Flux
using Flux: params
using DrWatson

include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
include(srcdir("layers", "ConvLSTM2Dv2.jl"))

function build_model(; out, device)
  _model = Chain(
    ConvLSTM2Dv2((64, 64), (5, 5), (5, 5), 1=>128, pad=SamePad()),
    ConvLSTM2Dv2((64, 64), (5, 5), (5, 5), 128=>64, pad=SamePad()),
    KeepLast(
      ConvLSTM2Dv2((64, 64), (5, 5), (5, 5), 64=>64, pad=SamePad()),
    ),
    RepeatInput(
      out,
      ConvLSTM2Dv2((64, 64), (5, 5), (5, 5), 64=>64, pad=SamePad()),
    ),
    TimeDistributed(
      Conv((1,1), 64=>1, pad=SamePad(), sigmoid_fast,bias=false)
    )
  )
  model = device(_model)
  model, params(model)
end

