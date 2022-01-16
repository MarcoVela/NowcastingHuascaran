using Serialization
using ClimateBase
using StatsBase
using Dates
using ProgressMeter
using Base.Threads
using Suppressor
using Statistics
using DataStructures
using ThreadTools

const ENVIRONMENT = "OR"
const PLATFORM = "G16"

function create_glm_time(datetime)
    y = lpad(year(datetime), 4, '0')
    d = lpad(dayofyear(datetime), 3, '0')
    h = lpad(hour(datetime), 2, '0')
    m = lpad(minute(datetime), 2, '0')
    s = lpad(second(datetime), 2, '0')
    t = millisecond(datetime) ÷ 100
    return "$y$d$h$m$s$t"
  end

function get_GOESR_edges(resolution_km = 2.0)
    km_to_degree_lat = 1/110.574
    km_to_degree_lon = 1/111.32
    step_lat = Float32(resolution_km * km_to_degree_lat)
    step_lon = Float32(resolution_km * km_to_degree_lon)
    lat = -51:step_lat:13
    lon = -84:step_lon:-31
    (lon, lat)
end

struct Flash
    latitude::Vector{Float32}
    longitude::Vector{Float32}
    quality::Vector{Int16}
    energy::Vector{Int16}
    area::Vector{Int16}
    time_start::DateTime
    time_end::DateTime
    fname::String
end

function group_by_interval(coords, dt)
    day = coords[begin].time_start |> Date
    n_groups = Second(60 * 60 * 24) ÷ Second(dt)
    groups = Vector{Vector{eltype(coords)}}()
    sizehint!(groups, n_groups)
    ti = DateTime(day)

    while ti < day + Day(1)
        next_ti = ti + dt
        group = filter(c -> ti <= c.time_start < next_ti, coords)
        push!(groups, group)
        ti = next_ti
    end
    groups
end


function grid(structs, (lon_edge, lat_edge), (lon_prop, lat_prop))
    lon = vcat((getproperty(x, lon_prop) for x in structs)...)
    lat = vcat((getproperty(x, lat_prop) for x in structs)...)
    hist = fit(Histogram{Float32}, (lon, lat), (lon_edge, lat_edge))
    hist.weights
end

function grid_flashes(flashes, edges)
    grid(flashes, edges, (:longitude, :latitude))
end

function create_glm_filename(
    environment,
    platform,
    start_time::DateTime,
    end_time::DateTime,
    creation_time::DateTime = now()
  )
    start_time_str = create_glm_time(start_time)
    end_time_str = create_glm_time(end_time)
    creation_time_str = create_glm_time(creation_time)
    create_glm_filename(
        environment,
        platform,
        start_time_str,
        end_time_str,
        creation_time_str
    )
end
  
function create_glm_filename(
    environment,
    platform,
    start_time::String,
    end_time::String,
    creation_time::String
)
    "$(environment)_GLM-L2-GLMF-M3_$(platform)_s$(start_time)_e$(end_time)_c$(creation_time).nc"
end

function create_FED_clim_array(groups, (lon_edge, lat_edge), dt, resolution_km)
    day = groups[begin][begin].time_start |> Date |> DateTime
    time_interval = day:Second(dt):(day + Day(1))
    lon_dim = midpoints(lon_edge)
    lat_dim = midpoints(lat_edge)
    tim_dim = midpoints(time_interval)
    n_lon, n_lat, n_tim = length(lon_dim), length(lat_dim), length(tim_dim)
    A = Array{Float32, 3}(undef, (n_lon, n_lat, n_tim))
    @assert n_tim == length(groups) "Mismatch between groups and time intervals"
    for i = 1:n_tim
        A[:,:,i] = grid_flashes(groups[i], (lon_edge, lat_edge))
    end
    start_time = day
    end_time = day + Day(1)
    filename = create_glm_filename(ENVIRONMENT, PLATFORM, start_time, end_time)
    dims = (Lon(lon_dim), Lat(lat_dim), Ti(tim_dim))

    attrib = OrderedDict(
        "cdm_data_type"             => "Image",
        "keywords"                  => "ATMOSPHERE > ATMOSPHERIC ELECTRICITY > LIGHTNING, ATMOSPHERE > ATMOSPHERIC PHENOMENA > LIGHTNING",
        "keywords_vocabulary"       => "NASA Global Change Master Directory (GCMD) Earth Science Keywords, Version 7.0.0.0.0",
        "summary"                   => "The Lightning Detection Gridded product generates fields starting from the GLM Lightning Detection Events, Groups, Flashes product.  It consists of flash extent density.",
        "title"                     => "GLM L2 Lightning Detection Gridded Product",
        "dataset_name"              => filename,
        "orbital_slot"              => "GOES-East",
        "platform_ID"               => PLATFORM,
        "production_data_source"    => "Postprocessed",
        "spatial_resolution"        => "$(resolution_km)km at nadir",
        "time_coverage_end"         => Dates.format(end_time, "YYYY-mm-ddTHH:MM:SS.ss"),
        "time_coverage_start"       => Dates.format(start_time, "YYYY-mm-ddTHH:MM:SS.ss"),
        "timeline_id"               => "ABI Mode 3",
    )
    FED = ClimArray(A, dims; name="flash_extent_density", attrib=attrib)
    files_per_group = ClimArray(length.(groups), (Ti(tim_dim),); name="files_per_group")

    (FED, files_per_group), filename
end

function ncwrite_compressed(file::String, Xs; globalattr = Dict())

    # TODO: Fixing this is very easy. Simply make a `"ncells"` dimension, and then write
    # the `"lon"` and `"lat"` cfvariables to the nc file by decomposing the coordinates
    # into longitude and latitude.
    if any(X -> hasdim(X, Coord), Xs)
        error("""
        Outputing `UnstructuredGrid` coordinates to .nc files is not yet supported,
        but it is an easy fix, see source of `ncwrite`.
        """)
    end
  
    ds = NCDataset(file, "c"; attrib = globalattr)
    # NCDataset("file.nc", "c"; attrib = globalattr) do ds
        for (i, X) in enumerate(Xs)
            n = string(X.name)
            if n == ""
                n = "x$i"
                @warn "$i-th ClimArray has no name, naming it $(n) instead."
            end
            ClimateBase.add_dims_to_ncfile!(ds, dims(X))
            attrib = X.attrib
            isnothing(attrib) && (attrib = Dict())
            dnames = ClimateBase.dim_to_commonname.(dims(X))
            data = Array(X)
            ClimateBase.defVar(ds, n, data, (dnames...,); attrib, deflatelevel=1)
        end
        close(ds)
    # end
end

function ncwrite_compressed(file::String, X::ClimArray; globalattr = Dict())
    ncwrite_compressed(file, (X,); globalattr)
end



function run(indir, outdir, resolution_km, dt)
    fs = readdir(indir; join=true)
    #p = Progress(length(fs))
    (lon_edge, lat_edge) = get_GOESR_edges(resolution_km)
    mkpath(outdir)
    @showprogress for f in fs
        glm = deserialize(f)
        groups = group_by_interval(glm, dt)
        A, filename = create_FED_clim_array(groups, (lon_edge, lat_edge), dt, resolution_km)
        @suppress ncwrite_compressed(joinpath(outdir, filename), A)
    end
#=     results = tmap(f -> begin
        glm = deserialize(f)
        groups = group_by_interval(glm, dt)
        A, filename = create_FED_clim_array(groups, (lon_edge, lat_edge), dt, resolution_km)
        next!(p)
        (A, filename)
    end, 4,fs)
    for (A, filename) in results
        @suppress ncwrite_compressed(joinpath(outdir, filename), A)
    end =#
end
