using Flux
include(srcdir("layers", "ConvLSTM2D.jl"))
using BSON

function determine_channels(c::BatchNorm)
  c.chs
end
function determine_channels(c::Conv)
  size(c.weight, ndims(c))
end

function build_model(; out, device, base_model, ch="same")
  @assert (length(base_model) == 2) "Base model should only have 2 elements"
  testmode!(base_model)
  encoder, decoder = base_model
  _ch = determine_channels(encoder[end]) # Asumiendo una capa conv o BatchNorm
  hidden_channels = ch == "same" ? _ch : ch
  temporal_encoder_decoder = Chain(
    KeepLast(
      ConvLSTM2D((8, 8), (3, 3),  _ch => hidden_channels, pad=SamePad()),
    ),
    RepeatInput(
      out,
      ConvLSTM2D((8, 8), (3, 3), hidden_channels => _ch, pad=SamePad()),
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
