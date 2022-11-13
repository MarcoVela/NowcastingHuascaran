push!(LOAD_PATH,"../src/")

using Documenter
using Revise
Revise.revise()
using NowcastingHuascaran

makedocs(modules = [NowcastingHuascaran],
    sitename="Nowcasting Huascaran",
    pages = [
      "Introducción" => "index.md",
      "Dataset" => "man/basics.md",
      "API" => [
        "Tipos" => "lib/types.md",
        "Funciones" => "lib/functions.md",
      ]
    ]
)

deploydocs(
    repo = "github.com/characat0/NowcastingHuascaran.git",
)