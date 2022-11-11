push!(LOAD_PATH,"../src/")

using Documenter
using Revise
Revise.revise()
using NowcastingHuascaran

makedocs(sitename="Nowcasting Huascaran", modules=[NowcastingHuascaran])
