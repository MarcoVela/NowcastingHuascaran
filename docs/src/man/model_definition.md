# Definición de Modelos

## Carpetas relevantes

```
src/
  architecture/
  layers/
```
La carpeta architecture sirve como repositorio de arquitecturas de modelo, cada archivo 
contiene la definición de una nueva arquitectura. 
La carpeta layers contiene utilitarios como capas que se reutilizarán en varias 
arquitecturas.

IMPORTANTE: No modificar el archivo de arquitectura una vez que ha sido usados para entrenar 
un modelo, hacerlo puede afectar la evaluación, reentrenamiento y despliegue de los modelos 
ya entrenados. Si se realizan modificaciones, deberán eliminarse los experimentos asociados.


## Introducción

Para definir un modelo nuevo a entrenar se deberá crear un archivo en el repositorio de 
arquitecturas.

Un archivo de arquitectura debe definir la función [`build_model`](@ref).

El framework para el desarrollo de modelos es [Flux.jl](http://fluxml.ai/Flux.jl/stable/), 
los modelos son definidos en forma de functores, con parámetros que pueden actualizarse 
usando backpropagation.

## Ejemplos

```jl
# layers/TimeDistributed.jl
using Flux

struct TimeDistributed{M}
  m::M
end

function (t::TimeDistributed)(x::AbstractArray{T, N}) where {T,N}
  h = [t.m(x_t) for x_t in Flux.eachlastdim(x)]
  reshape(reduce(ncat, h), size(h[1])..., :)
end

Flux.@functor TimeDistributed

```

```jl
# architecture/Arquitectura01.jl
using Flux
using Flux: params

include("../layers/TimeDistributed.jl")
include("../layers/KeepLast.jl")
include("../layers/RepeatInput.jl")
include("../layers/ConvLSTM2Dv2.jl")

function build_model(; out, device)
  _model = Chain(
    KeepLast(
      ConvLSTM2Dv2((64, 64), (5, 5), (5, 5), 1=>256, pad=SamePad()),
    ),
    RepeatInput(
      out,
      ConvLSTM2Dv2((64, 64), (5, 5), (5, 5), 256=>256, pad=SamePad()),
    ),
    TimeDistributed(
      Conv((1,1), 256=>1, pad=SamePad(), sigmoid_fast, bias=false)
    )
  )
  model = device(_model)
  model, params(model)
end

```


