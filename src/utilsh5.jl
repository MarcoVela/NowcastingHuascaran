using Random: MersenneTwister, shuffle!
using ClimateBase: ncread
using HDF5: h5open, create_group, close
using JLD2: jldsave
using Clustering: dbscan
using Images: dilate
using Serialization: serialize
using DrWatson



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
function crop_dataset(ds, indsmat; WIDTH, HEIGHT)
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
    threshold=Float32(0.0),
    radius=24, 
    min_cluster_size=128, 
    time_factor=radius/2;
    width=WIDTH,
    height=HEIGHT,
)
    inds = findall(>(threshold), dilate(array; dims=[1,2]))
    length(inds) == 0 && return []
    indsmat = as_floats(inds)
    indsmat[3, :] *= time_factor
    clusters = dbscan(indsmat, radius; min_cluster_size)
    indsmat[3, :] /= time_factor
    [crop_dataset(array, indsmat[:, c.core_indices]; width, height) for c in clusters]
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

mutable struct H5Store{T, R}
    X_batch::Vector{T}
    y_batch::Vector{R}
    batchsize::Int
    state::Int
    H5Store(Q::DataType, Q2::DataType, batchsize) = new{Q}(Vector{Q}(), Vector{Q2}(), batchsize, 0)
    H5Store(Q::DataType, batchsize) = new{Q}(Q, Q, batchsize, 0)
    H5Store(Q::DataType) = new{Q}(Q, Q, 1024, 0)
end

folder_name(::H5Store) = "h5"

function save!(store::H5Store, X, y, metadata; filename, prefix)
    append!(store.X_batch, X)
    append!(store.y_batch, y)
    @assert length(store.X_batch) == length(store.y_batch)
    while length(store.X_batch) > store.batchsize
        X_tensor = list_to_tensor(splice!(store.X_batch, 1:store.batchsize))
        y_tensor = list_to_tensor(splice!(store.y_batch, 1:store.batchsize))
        fname = joinpath(prefix, "$(store.state)_$(join(size(X_tensor), 'x'))_$(join(size(y_tensor), 'x')).h5")
        fid = h5open(fname, "w")
        data_group = create_group(fid, "data")
        data_group["X", shuffle=(), deflate=3] = X_tensor
        data_group["y", shuffle=(), deflate=3] = y_tensor
        metadata_group = create_group(fid, "metadata")
        metadata_group["params"] = metadata
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


function generate_dataset(indir, outdir;
    params=Dict(
        :WIDTH => 64,
        :HEIGHT => 64,
        :TIME_IN => 32,
        :TIME_OUT => 16,
        :RNG_SEED => 42,
        :CLUSTERING_THRESHOLD => zero(Float32),
        :CLUSTERING_RADIUS => 24,
        :CLUSTERING_MIN_CLUSTER_SIZE => 128,
        :CLUSTERING_TIME_FACTOR => 6,
        :SLIDING_WINDOW_STEPS => 2,
        :BATCH_SIZE => 2048,
    )
)
    h5store = H5Store(Tuple{rray{Float32, 3}, Array{Float32, 3}})
    h5folder = folder_name(h5store)
    savefolder = savename(params)
    outdir = joinpath(outdir, savefolder, h5folder)
    @tag!(params)
    @unpack RNG_SEED = params
    @unpack WIDTH, HEIGHT, TIME_IN, TIME_OUT = params
    @unpack CLUSTERING_THRESHOLD, CLUSTERING_RADIUS, CLUSTERING_MIN_CLUSTER_SIZE, CLUSTERING_TIME_FACTOR = params
    @unpack SLIDING_WINDOW_STEPS = params
    @showprogress for f in readdir(indir; join=true)
        rng = MersenneTwister(RNG_SEED)
        A = ncread(f, "flash_extent_density")
        boxed_events = arr_boxes(A.data; 
            threshold=CLUSTERING_THRESHOLD,
            radius=CLUSTERING_RADIUS, 
            min_cluster_size=CLUSTERING_MIN_CLUSTER_SIZE,
            time_factor=CLUSTERING_TIME_FACTOR;
            width=WIDTH,
            height=HEIGHT,
        )
        shuffle!(rng, boxed_events)
        boxes = sliding_window(boxed_events; 
            step=SLIDING_WINDOW_STEPS,
            window_size=TIME_IN+TIME_OUT,
        )
        X_boxes = [box[:,:,begin:TIME_IN] for box in boxes]
        y_boxes = [box[:,:,TIME_IN+1:end] for box in boxes]
        save!(h5store, X_boxes, y_boxes, params; prefix=outdir)
    end

end
