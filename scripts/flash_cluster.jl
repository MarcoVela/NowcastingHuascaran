using Clustering
using ClimateBase
using ThreadTools
using ProgressMeter
using Serialization

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

#rectangle(w, h, x, y) = Shape(x .+ [0,w,w,0], y .+ [0,0,h,h])
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
        p = heatmap(ds.data[:,:,i], clim=(0,10), c=cgrad([:black, :white]), legend=false, xlim=(0, x_n), ylim=(0, y_n))
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

function main_isolate_events(indir, outdir)
    mkpath(outdir)
    @showprogress for f in readdir(indir; join=true)
        A = ncread(f, "flash_extent_density")
        clusters, inds = clim_arr_clusters(A)
        boxed_events = isolate_events(A, clusters, inds)
        fname = basename(f)[begin:end-3] * "_boxes.jls"
        serialize(joinpath(outdir, fname), boxed_events)
    end
end