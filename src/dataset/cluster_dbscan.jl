using Clustering
using StaticArrays
using NearestNeighbors
using Distances
using Base.Iterators: product
using Dates

function climarr_cluster(arr::AbstractArray{T, N}; 
  radius, min_neighbors, min_cluster_size, t_scale = 1, threshold=zero(T)) where {T,N}
  found_idx = findall(>(threshold), arr)
  idx_mat = reinterpret(reshape, Int, found_idx)
  points = Float32.(idx_mat)
  points[3,:] .*= t_scale
  clusters = dbscan(points, radius; min_cluster_size, min_neighbors)
  [
    found_idx[append!(x.boundary_indices, x.core_indices)]
    for x in clusters
  ]
end

function bbox(cluster::AbstractVector{CartesianIndex{N}}) where {N}
  cluster_arr = reinterpret(reshape, Int, cluster)
  ntuple(i -> extrema(@view(cluster_arr[i, :])), N)
end

function expand_bbox(box::NTuple{N}, min_dims::NTuple{N}, limits::NTuple{N}) where {N}
  ntuple(i -> begin
    lower, upper = box[i]
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


function moving_window(climarr, clusters, dimensions, windows)
  for i in 1:length(dimensions)
    @assert length(climarr.dims[i]) >= dimensions[i]
  end
  limits = tuple([length(x) for x in dims(climarr)]...)
  boxes = expand_bbox.(bbox.(clusters), Ref(dimensions), Ref(limits))
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
    "lat" => out_lat,
    "lon" => out_lon,
    "time" => datetime2unix.(out_tim),
  )
end


# Parece que 2 es el radio ideal para obtener parches con tamaño medio de 24 frames y 10 de ancho,largo
# Tambien parece que 4 de radio y 3 de escala temporal producen resultados similares en media de frames pero con 12 de ancho,largo

function generate_dataset(climarr::AbstractArray{T, N}, dimensions; 
  radius, min_neighbors, min_cluster_size, t_scale = 1, windows=ntuple(_->0, length(dimensions)), threshold=zero(T)) where {T, N}
  clusters = climarr_cluster(climarr; radius, min_neighbors, min_cluster_size, t_scale, threshold)
  subarrs = moving_window(climarr, clusters, dimensions, windows)
  subarrs_to_plain(subarrs)
end

