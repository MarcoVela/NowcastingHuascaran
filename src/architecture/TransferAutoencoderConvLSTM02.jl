using Flux
include(srcdir("layers", "ConvLSTM2Dv3.jl"))
include(srcdir("layers", "TimeDistributed.jl"))
include(srcdir("layers", "KeepLast.jl"))
include(srcdir("layers", "RepeatInput.jl"))
using BSON

function determine_channels(c::BatchNorm)
  c.chs
end
function determine_channels(c::Conv)
  size(c.weight, ndims(c))
end

function build_model(; out, device, base_model, ch=nothing)
  @assert (length(base_model) == 2) "Base model should only have 2 elements"
  testmode!(base_model)
  encoder, decoder = base_model
  _ch = determine_channels(encoder[end]) # Asumiendo una capa conv o BatchNorm
  hidden_channels = isnothing(ch) ? _ch : ch
  temporal_encoder_decoder = Chain(
    ConvLSTM2Dv3((8, 8), (5, 5), (5, 5),  _ch => hidden_channels, pad=SamePad()),
    KeepLast(
      ConvLSTM2Dv3((8, 8), (5, 5), (5, 5),  hidden_channels => hidden_channels, pad=SamePad()),
    ),
    RepeatInput(
      out,
      ConvLSTM2Dv3((8, 8), (5, 5), (5, 5), hidden_channels => _ch, pad=SamePad()),
    ),
  )
  _model = Chain(
    TimeDistributed(encoder),
    temporal_encoder_decoder,
    TimeDistributed(decoder),
  )
  model = device(_model)
  model, params(model[2])
end
