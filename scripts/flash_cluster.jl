using Clustering
using ClimateBase
using ThreadTools
using ProgressMeter
using Serialization
using JLD2
using CodecZlib
using Plots
using NPZ
using HDF5

const WIDTH = 64
const HEIGHT = 64


function cluster_to_bounding_box(cluster, indsmat)
    clusinds = indsmat[:, cluster.core_indices]
    y,x = minimum(clusinds; dims=2)
    y2,x2 = maximum(clusinds; dims=2)
    w = x2 - x
    h = y2 - y
    (w,h,x,y)
end

as_ints(a::AbstractArray{CartesianIndex{L}}) where L = reshape(reinterpret(Int, a), (L, size(a)...))

as_floats(a::AbstractArray{CartesianIndex{L}}) where L = Float64.(reshape(reinterpret(Int, a), (L, size(a)...)))

function boxes_from_image(image; threshold=1.0, radius=24, min_cluster_size=128, time_factor=10)
    inds = findall(>=(threshold), image)
    length(inds) == 0 && return []
    indsmat = as_floats(inds)
    clusters = dbscan(indsmat, radius; min_cluster_size)
    [cluster_to_bounding_box(x, indsmat) for x in clusters]
end

rect(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
#plot!(bbox, fillalpha=0, linecolor=:white)

function clim_arr_clusters(climarr; threshold=1.0, radius=24, min_cluster_size=128, time_factor=10)
    inds = findall(>=(threshold), climarr.data)
    indsmat = as_floats(inds)
    indsmat[3, :] *= time_factor
    dbscan(indsmat, radius; min_cluster_size), inds
end

function clim_arr_boxes(climarr)
    clusters, _ = clim_arr_clusters(climarr)
    [cluster_to_bounding_box(x, indsmat) for x in clusters]
end


function run_create_boxes(indir, outdir)
    fs = readdir(indir; join=true)
    @showprogress for f in fs
        ds = ncread(f, "flash_extent_density")
        boxes = clim_arr_boxes(ds)
        fname = basename(f)[begin:end-3] * "_boxes.jls"
        serialize(joinpath(outdir, fname), boxes)
    end
end





function index_box(inds)
    y2 = maximum([x.I[1] for x in inds])
    x2 = maximum([x.I[2] for x in inds])
    y = minimum([x.I[1] for x in inds])
    x = minimum([x.I[2] for x in inds])
    w = x2 - x
    h = y2 - y
    (w,h,x,y)
end

function reshape_box(box, (dx, dy), (max_x, max_y))
    (w,h,x,y) = box

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

function crop_boxes(indexes_groups, (dx, dy), (max_x, max_y))
    boxes = index_box.(indexes_groups)
    for i = 1:length(boxes)
        boxes[i] = reshape_box(boxes[i], (dx, dy), (max_x, max_y))
    end
    boxes
end


function visualize(ds, clusters, inds, indsmat)
    @time @gif for i = 1:size(ds, 3)
        x_n = size(ds, 2)
        y_n = size(ds, 1)
        p = heatmap(ds.data[:,:,i], clim=(0,10), c=cgrad([:black, :white]), legend=false, xlim=(0, x_n), ylim=(0, y_n), size=(480,480))
        for c in clusters
            ind = inds[c.core_indices]
            if !(ind[begin][3] <= i <= ind[end][3])
                continue
            end
            bbox = rect(reshape_box(cluster_to_bounding_box(c, indsmat), (WIDTH, HEIGHT), (x_n, y_n))...)
            plot!(bbox, fillalpha=0, linecolor=:white)
        end
        p
    end
end

function crop_event(ds, c, inds)
    ind = inds[c.core_indices]
    box = index_box(ind)
    x_n = size(ds, 2)
    y_n = size(ds, 1)
    box = reshape_box(box, (WIDTH, HEIGHT), (x_n, y_n))
    t_ini = ind[begin][3]
    t_fin = ind[end][3]
    (w,h,x,y) = box
    ds[y:(y+h-1), x:(x+w-1), t_ini:t_fin]
end

function isolate_events(ds, clusters, inds)
    [crop_event(ds, c, inds) for c in clusters]
end

const TIME_SIZE = 16

function generate_dataset_python(rng, boxes)
    ds = Vector{Array{Float32, 3}}()
    for box in boxes
        n = size(box, 3)
        for i = 1:(n - TIME_SIZE + 1)
            push!(ds, box[:, :, i:(i+TIME_SIZE-1)])
        end
    end
    shuffle!(rng, ds)
end

function consolidate_in_array(ds)
    outds = Array{Float32, 5}(undef, (1, WIDTH, HEIGHT, TIME_SIZE, length(ds)))
    for i = 1:length(ds)
        outds[1, :, :, :, i] = ds[i]
    end
    outds
end

using Random
generate_dataset_python(boxes) = generate_dataset_python(Random.GLOBAL_RNG, boxes)

function main_isolate_events(indir, outdir, batchsize=1024)
    outdir_jls = joinpath(outdir, "jls")
    outdir_jld2 = joinpath(outdir, "jld2")
    outdir_h5 = joinpath(outdir, "h5")
    #outdir_npy = joinpath(outdir, "npy")
    mkpath(outdir_jls)
    mkpath(outdir_jld2)
    mkpath(outdir_h5)
    #mkpath(outdir_npy)
    outds = Vector{Array{Float32, 3}}()
    i = 1
    @showprogress for f in readdir(indir; join=true)
        A = ncread(f, "flash_extent_density")
        clusters, inds = clim_arr_clusters(A)
        boxed_events = isolate_events(A, clusters, inds)
        fname = basename(f)[begin:end-3] * "_boxes-$(WIDTH)x$(HEIGHT).jls"
        fname_jld2 = basename(f)[begin:end-3] * "_boxes-$(WIDTH)x$(HEIGHT).jld2"
        boxes = [x.data for x in boxed_events]
        serialize(joinpath(outdir_jls, fname), boxed_events)
        jldsave(joinpath(outdir_jld2, fname_jld2), true; boxes=boxes)
        boxes_serializable = generate_dataset_python(MersenneTwister(42), boxes)
        append!(outds, boxes_serializable)
        while length(outds) > batchsize
            batch = consolidate_in_array(first(outds, batchsize))
            outds = outds[batchsize+1:end]
            fname_hdf5 = "$(i)_boxes-$(WIDTH)x$(HEIGHT).h5"
            fid = h5open(joinpath(outdir_h5, fname_hdf5), "w")
            g = create_group(fid, "flash_extent_density")
            g["train", shuffle=(), deflate=3] = batch
            close(fid)
            i += 1
        end
        #npzwrite(joinpath(outdir_npy, fname_npy), boxes_serializable)
    end
    while length(outds) > batchsize
        batch = consolidate_in_array(first(outds, batchsize))
        outds = outds[batchsize+1:end]
        fname_hdf5 = "$(i)_boxes-$(WIDTH)x$(HEIGHT).h5"
        fid = h5open(joinpath(outdir_h5, fname_hdf5), "w")
        g = create_group(fid, "flash_extent_density")
        g["train", shuffle=(), deflate=3] = batch
        close(fid)
        i += 1
    end
end

