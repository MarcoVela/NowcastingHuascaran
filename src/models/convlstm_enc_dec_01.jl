function build_model(; N_out, device, dropout)
  model = Chain(
    TimeDistributed(
      Chain(
        Conv((3,3),  1=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
        Conv((3,3), 64=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
        Conv((3,3), 64=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad(), stride=2),
      
        Conv((3,3),  64=>128, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
        Conv((3,3), 128=>128, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad(), stride=2),
      ),
    ),
    KeepLast(
      ConvLSTM2D((16, 16), (3, 3),  128 => 128, pad=SamePad()),
    ),
    Dropout(dropout),
    RepeatInput(
      N_out,
      ConvLSTM2D((16, 16), (3, 3), 128 => 128, pad=SamePad()),
    ),
    TimeDistributed(
      Chain(
        ConvTranspose((7,7), 128=>128, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
        ConvTranspose((7,7), 128=> 64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad(), stride=2),
      
        ConvTranspose((7,7), 64=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
        ConvTranspose((7,7), 64=> 1, σ, pad=SamePad(), stride=2),
      )
    ),
  )
  device(model)
end