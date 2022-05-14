using Clustering
using StaticArrays
using NearestNeighbors

function climarr_cluster(arr::AbstractArray{T, N}, metric; 
  radius, min_neighbors, min_cluster_size, kwargs...) where {T,N}
  found_idx = findall(>(zero(T)), arr)
  points = Array(reinterpret(reshape, Int, found_idx))
  return dbscan(points, radius; min_cluster_size, min_neighbors)
  idx = reinterpret(SVector{N, T}, found_idx)
  tree = KDTree(idx, metric)
  Clustering._dbscan(tree, points, radius; min_neighbors, min_cluster_size)
end

