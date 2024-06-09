push!(LOAD_PATH,"../src/")

using Documenter
using Revise
Revise.revise()
using NowcastingHuascaran

makedocs(modules = [NowcastingHuascaran],
    sitename="Nowcasting Huascaran",
    pages = [
      "Introducción" => "index.md",
      "Descarga" => "man/download.md",
      "Generación de Conjuntos de Datos" => "man/dataset.md",
      "Definición de Modelos" => "man/model_definition.md",
      "Entrenamiento" => "man/training.md",
      "Evaluación" => "man/evaluation.md",
      "API" => [
        "Tipos" => "lib/types.md",
        "Funciones" => "lib/functions.md",
      ],
      "Configuración Avanzada" => "man/advanced.md",
    ]
)

deploydocs(
    repo = "github.com/characat0/NowcastingHuascaran.git",
)