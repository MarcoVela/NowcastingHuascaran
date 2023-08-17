using Clustering
using StaticArrays
using NearestNeighbors
using Distances
using Base.Iterators: product
using Dates


function climarr_cluster(arr::AbstractArray{T, N}; 
  radius, min_neighbors, min_cluster_size, t_scale = 1, threshold=zero(T)) where {T,N}
  found_idx = findall(>=(threshold), arr)

  idx_mat = reinterpret(reshape, Int, found_idx)
  points = Float32.(idx_mat)
  points[3,:] .*= t_scale
  dbscan_results = dbscan(points, radius; min_cluster_size, min_neighbors)
  [
    found_idx[append!(x.boundary_indices, x.core_indices)]
    for x in dbscan_results.clusters
  ]
end

function bbox(cluster::AbstractVector{CartesianIndex{N}}, padding::NTuple{N} = ntuple(_ -> 0, N)) where {N}
  cluster_arr = reinterpret(reshape, Int, cluster)
  ntuple(i -> extrema(@view(cluster_arr[i, :])) .+ (-padding[i], padding[i]), N)
end

function expand_bbox(box::NTuple{N}, min_dims::NTuple{N}, limits::NTuple{N}) where {N}
  ntuple(i -> begin
    lower, upper = box[i]
    lower = max(lower, 1)
    upper = min(upper, limits[i])
    range = upper - lower
    if lower - (min_dims[i] - range) ÷ 2 < 1
      lower = 1
      upper = lower + min_dims[i]
    elseif upper + (min_dims[i] - range) ÷ 2 > limits[i]
      upper = limits[i]
      lower = upper - min_dims[i]
    end
    range = upper - lower
    if range < min_dims[i]
      lower -= (min_dims[i]-range) ÷ 2
      if lower < 1
        lower = 1
      end
      upper += (min_dims[i]-upper + lower)
      if upper > limits[i]
        upper = limits[i]
        lower = upper - min_dims[i]
      end
    end
    (lower, upper)
  end, N)
end


function moving_window(climarr, clusters, dimensions, windows, padding)
  for i in 1:length(dimensions)
    @assert length(climarr.dims[i]) >= dimensions[i]
  end
  limits = tuple([length(x) for x in dims(climarr)]...)
  bboxes = bbox.(clusters, Ref(padding))
  boxes = expand_bbox.(bboxes, Ref(dimensions), Ref(limits))
  res = Vector{typeof(climarr[ntuple(_->:, length(dims(climarr)))])}()
  N = length(dimensions)
  for box in boxes
    ranges = ntuple(i -> begin
      lower, upper = box[i]
      windows[i] == 0 && return lower:1:lower
      lower:windows[i]:(upper-dimensions[i])
    end, N)
    for start_indices in product(ranges...)
      indices = ntuple(i -> start_indices[i]:(start_indices[i] + dimensions[i]-1), N)
      push!(res, climarr[indices...])
    end
  end
  res
end

function subarrs_to_plain(climarrs)
  lon, lat, tim = size(climarrs[1])
  N = length(climarrs)
  lon_dim, lat_dim, tim_dim = dims(climarrs[1])
  out_arr = Array{eltype(climarrs[1]), 5}(undef, lon, lat, 1, N, tim)
  out_lon = Array{eltype(lon_dim), 2}(undef, lon, N)
  out_lat = Array{eltype(lat_dim), 2}(undef, lat, N)
  out_tim = Array{eltype(tim_dim), 2}(undef, tim, N)
  for (i, arr) in enumerate(climarrs)
    out_arr[:, :, 1, i, :] = arr.data
    out_lon[:, i] = lon_dim.val
    out_lat[:, i] = lat_dim.val
    out_tim[:, i] = tim_dim.val
  end
  Dict(
    "FED" => out_arr,
    "lon" => out_lon,
    "lat" => out_lat,
    "time" => datetime2unix.(out_tim),
  )
end


# Parece que 2 es el radio ideal para obtener parches con tamaño medio de 24 frames y 10 de ancho,largo
# Tambien parece que 4 de radio y 3 de escala temporal producen resultados similares en media de frames pero con 12 de ancho,largo

"""
    generate_dataset(climarr::AbstractArray, dimensions::NTuple{3}; 
                     radius::AbstractFloat, 
                     min_neighbors::Int, 
                     min_cluster_size::Int, 
                     t_scale::AbstractFloat, 
                     windows::NTuple{3}=(0,0,0), 
                     threshold::Int=0, 
                     padding::NTuple{3}=(0,0,0))
Crea un diccionario con los elementos componentes del dataset.

Los argumentos `radius`, `min_neighbors` y `min_cluster_size` son pasados directamente a [`Clustering.dbscan`](https://juliastats.org/Clustering.jl/stable/dbscan.html#Clustering.dbscan).
# Argumentos
- `climarr` : `ClimArray` que contiene la data con dimensiones longitud, latitud y tiempo
- `dimensions` : tupla que representa las dimensiones de los parches de salida
- `t_scale::AbstractFloat` : factor para multiplicar la dimensión temporal antes de clusterizar
- `windows::NTuple{3} = (0, 0, 0)` : dimensiones de las ventanas moviles que se aplicarán
- `threshold::Int = 0` : umbral para la binarización
- `padding::NTuple{3} = (0, 0, 0)` : tamaño del padding en cada dimensión
# Detalles
El argumento `climarr` debe ser un `ClimArray` con 3 dimensiones ordenadas: longitud, latitud y tiempo.

Se aplica clusterización al array `climarr`, usando `dbscan` y tratando a la dimensión `T` como una tercera dimensión espacial escalada con el factor `t_scale`. 
Se calculan bounding boxes de tamaño `dimensions` para cada cluster obtenido.

Para cada dimensión si el tamaño del cluster es mayor al del bounding box propuesto aplicará una ventana movil en esa dimensión.
La cantidad de elementos a desplazarse en la dimensión está indicado por `windows`, si este valor es 0, se tomará el límite inferior del cluster. 
En caso contrario se aplicará `padding` en la dimensión y se obtendrán nuevas bounding boxes de tamaño `dimensions`.

La función `generate_dataset` devolverá un diccionario con 4 entradas:
- FED : un arreglo de 5 dimensiones (Lon×Lat×1×N×T) donde la 3ra dimensión está vacía por conveniencia
- lon : una matriz (Lon×N) que almacena la longitud de cada elemento en la grilla
- lat : una matriz (Lat×N) que almacena la latitud de cada elemento en la grilla
- time : una matriz (T×N) que almacena el tiempo en segundos desde el unix epoch como `Float64` de cada elemento en la grilla
"""
function generate_dataset(climarr::AbstractArray{T, N}, dimensions; 
  radius, min_neighbors, min_cluster_size, t_scale = 1, windows=ntuple(_->0, length(dimensions)), threshold=zero(T), padding=ntuple(_->0, length(dimensions))) where {T, N}

  clusters = climarr_cluster(climarr; radius, min_neighbors, min_cluster_size, t_scale, threshold)
  subarrs = moving_window(climarr, clusters, dimensions, windows, padding)
  filter!(arr -> sum(arr) > min_cluster_size, subarrs)
  subarrs_to_plain(subarrs)
end

