using Random: MersenneTwister, shuffle!
using ClimateBase: ncread
using HDF5: h5open, create_group, close
using JLD2: jldsave
using Clustering: dbscan
using Images: dilate!, erode!
using Serialization: serialize


WIDTH = 64
HEIGHT = 64
TIME_SIZE = 32

# transforms an array of cartesian indexes to a matrix
as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))

as_floats(a::AbstractArray{CartesianIndex{L}}) where L = Float64.(as_ints(a))

# creates a bounding box for the indices matrix (y,x,t)
function cluster_to_bounding_box(clusinds)::Tuple{Int,Int,Int,Int}
    y,x = minimum(clusinds; dims=2)
    y2,x2 = maximum(clusinds; dims=2)
    w = x2 - x
    h = y2 - y
    map(Int64, (w,h,x,y))
end

# finds clusters of pixels (higher or equal than threshold) in an array and returns the bounding boxes (spatial and temporal) containing them
# note: the implementation of dbscan used here treats the time dimension as another spatial dimension
function boxes_from_array(array; threshold=1.0, radius=24, min_cluster_size=128, time_factor=radius / 2)
    inds = findall(>=(threshold), array)
    length(inds) == 0 && return []
    indsmat = as_floats(inds)
    indsmat[3, :] *= time_factor
    clusters = dbscan(indsmat, radius; min_cluster_size)
    [cluster_to_bounding_box(indsmat[:, x.core_indices]) for x in clusters]
end

# determines the bounding box of an array of cartesian indices
function index_box(inds)
    y2 = maximum([x.I[1] for x in inds])
    x2 = maximum([x.I[2] for x in inds])
    y = minimum([x.I[1] for x in inds])
    x = minimum([x.I[2] for x in inds])
    w = x2 - x
    h = y2 - y
    (w,h,x,y)
end

# reshapes a bounding box to be of a defined shape
@inline function reshape_box((w,h,x,y), (dx, dy), (max_x, max_y))::Tuple{Int,Int,Int,Int}
    diff = dx - w
    delta = floor(Int, diff / 2)
    x = min(max(1, x - delta), max_x - dx)
    w = dx
    diff = dx - h
    delta = floor(Int, diff / 2)
    y = min(max(1, y - delta), max_y - dx)
    h = dy
    (w,h,x,y)
end


# crops the dataset provided using the cluster indices
function crop_dataset(ds, indsmat)
    box = cluster_to_bounding_box(indsmat)
    x_n = size(ds, 2)
    y_n = size(ds, 1)
    box = reshape_box(box, (WIDTH, HEIGHT), (x_n, y_n))
    t_ini = floor(Int, indsmat[3, begin])
    t_fin = floor(Int, indsmat[3, end])
    (w,h,x,y) = box
    ds[y:(y+h-1), x:(x+w-1), t_ini:t_fin]
end

# returns an array of boxes from an array
# note: the implementation of dbscan used here treats the time dimension as another spatial dimension
function arr_boxes(array; 
    threshold=Float32(1.0),
    radius=24, 
    min_cluster_size=128, 
    time_factor=radius/2,
)
    ds = array |> copy |> dilate! |> dilate! |> erode! |> erode!
    inds = findall(>(threshold), ds)
    length(inds) == 0 && return []
    indsmat = as_floats(inds)
    indsmat[3, :] *= time_factor
    clusters = dbscan(indsmat, radius; min_cluster_size)
    indsmat[3, :] /= time_factor
    [crop_dataset(array, indsmat[:, c.core_indices]) for c in clusters]
end


# create observations using a sliding window
function sliding_window(boxes; step=2, window_size=TIME_SIZE)
    ds = Vector{Array{Float32, 3}}()
    for box in boxes
        n = size(box, 3)
        for i = 1:step:(n - window_size + 1)
            push!(ds, box[:, :, i:(i+window_size-1)])
        end
    end
    ds
end

# Transforms a vector of 3d tensors into a 5d tensor
function list_to_tensor(list)
    w,h,t = size(first(list))
    tensor = Array{Float32, 5}(undef, (1, w, h, t, length(list)))
    for i = 1:length(list)
        tensor[1, :, :, :, i] = list[i]
    end
    tensor
end



abstract type DataStore{T} end;

function folder_name(::DataStore) end

function save!(::DataStore, chunk; kwargs...) end


struct JLSStore{T} <: DataStore{T}
    JLSStore(Q::DataType) = new{Q}()
end

folder_name(::JLSStore) = "jls"

function save!(::JLSStore, chunk; filename, prefix)
    serialize(joinpath(prefix, filename * ".jls"), chunk)
end

struct JLD2Store{T} <: DataStore{T}
    JLD2Store(Q::DataType) = new{Q}()
end

folder_name(::JLD2Store) = "jld2"

function save!(::JLD2Store, chunk; filename, prefix)
    jldsave(joinpath(prefix, filename * ".jld2"), true; data=chunk)
end

mutable struct H5Store{T} <: DataStore{T}
    batch::Vector{T}
    batchsize::Int
    state::Int
    H5Store(Q::DataType) = new{Q}(Vector{Q}(), 1024, 0)
    H5Store(Q::DataType, batchsize) = new{Q}(Vector{Q}(), batchsize, 0)
end

folder_name(::H5Store) = "h5"

function save!(store::H5Store, chunk; filename, prefix)
    windows = sliding_window(chunk)
    append!(store.batch, windows)
    while length(store.batch) > store.batchsize
        tensor = list_to_tensor(splice!(store.batch, 1:store.batchsize))
        fname = joinpath(prefix, "$(store.state)_$(join(size(tensor), 'x')).h5")
        fid = h5open(fname, "w")
        g = create_group(fid, "data")
        g["data", shuffle=(), deflate=3] = tensor
        close(fid)
        store.state += 1
    end
end

function binarize!(x, t)
    for i in 1:length(x)
        x[i] = x[i] > t
    end
    x
end


function generate_datasets(indir, outdir; 
    stores = (JLSStore, JLD2Store, H5Store), 
    binary_threshold = zero(Float32),
)
    savers = map(x -> x(Array{Float32, 3}), stores)
    folders = map(folder_name, savers)
    outdirs = joinpath.(outdir, folders)
    mkpath.(outdirs)
    @showprogress for f in readdir(indir; join=true)
        rng = MersenneTwister(42)
        A = ncread(f, "flash_extent_density")
        boxed_events = binarize!.(arr_boxes(A.data), binary_threshold)
        shuffle!(rng, boxed_events)
        base_filename,_ = splitext(basename(f))
        @sync for (saver, folder) in zip(savers, outdirs)
            @async save!(saver, boxed_events; filename=base_filename, prefix=folder)
        end
    end
end



