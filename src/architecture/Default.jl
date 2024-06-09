"""
    build_model(; out::Int, device, kwargs...)::Tuple{Function, Zygote.Params}
Crea un modelo y devuelve una tupla con la función que realiza el forward pass y los parámetros
para realizar back propagation.
# Argumentos
- `out`: Se utiliza para indicar la cantidad de pasos a futuro a predecir.
- `device`: Parámetro comunmente utilizado para indicar el dispositivo de aceleración.
"""
function build_model(; kwargs...)
  error("Modelo no implementado.")
end
