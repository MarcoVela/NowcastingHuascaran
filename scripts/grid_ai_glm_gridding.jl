using StatsBase
using Dates
using NCDatasets
using ProgressMeter
using DataStructures
using ClimateBase

function get_GOESR_grid_specifications(resolution_km = 2.0)
  spanEW = 0.151872*2.0
  spanNS = 0.151872*2.0
  centerEW = 0.0
  centerNS = 0.0
  resolution = resolution_km * 28e-6
  pixelsEW = floor(Int64, spanEW / resolution)
  pixelsNS = floor(Int64, spanNS / resolution)
  nadir_lon = -75.0
  return (
    spanEW = spanEW,
    spanNS = spanNS,
    centerEW = centerEW,
    centerNS = centerNS,
    resolution = resolution,
    pixelsEW = pixelsEW,
    pixelsNS = pixelsNS,
    nadir_lon = nadir_lon,
  )
end


function get_GOESR_edges(resolution_km = 2.0)
  km_to_degree_lat = 1/110.574
  km_to_degree_lon = 1/111.32
  step_lat = resolution_km * km_to_degree_lat
  step_lon = resolution_km * km_to_degree_lon 
  lat = -51:step_lat:13
  lon = -84:step_lon:-31
  (lon, lat)
end

# Usar esto para escribir al archivo netcdf
function get_out_grid(edges, pads)
  edges
end


function flash_extent_density(vs::Tuple{N, T}, edges) where {N,T}
  hist = fit(Histogram{Float32}, vs, edges)
  hist.weights
end


function parse_glm_filename_time(time_str)
  d = DateTime(time_str[2:5]*time_str[9:end-1], dateformat"yyyyHHMMSS") + 
  Day(parse(Int, time_str[6:8]) - 1)
  d + Millisecond(parse(Int, time_str[end]) * 100)
end


function parse_glm_filename(filename)
  parts = split(replace(filename, ".nc" => ""), '_')
  ops_environment = parts[1]
  algorithm = parts[2]
  platform = parts[3]
  start_time = parse_glm_filename_time(parts[4])
  end_time = parse_glm_filename_time(parts[5])
  created_time = parse_glm_filename_time(parts[6])
  return (
    environment = ops_environment,
    algorithm,
    platform,
    start_time,
    end_time,
    created_time
  )
end

function create_glm_time(datetime)
  y = lpad(year(datetime), 4, '0')
  d = lpad(dayofyear(datetime), 3, '0')
  h = lpad(hour(datetime), 2, '0')
  m = lpad(minute(datetime), 2, '0')
  s = lpad(second(datetime), 2, '0')
  t = millisecond(datetime) ÷ 100
  return "$y$d$h$m$s$t"
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
  fname = "$(environment)_GLM-L2-GLMF-M3_$(platform)_s$(start_time)_e$(end_time)_c$(creation_time).nc"
end

# Excepts the filepaths sorted
function group_filenames(filepaths, dt)
  first_filepath = filepaths[begin]
  first_file_info = parse_glm_filename(basename(first_filepath))
  group = [(first_filepath, first_file_info)]
  groups = Vector{typeof(group)}()
  for filepath in filepaths[2:end]
    if length(group) == 0
      fileinfo = parse_glm_filename(basename(filepath))
      push!(group, (filepath, fileinfo))
      continue
    end
    fileinfo = parse_glm_filename(basename(filepath))
    _, info = first(group)
    push!(group, (filepath, fileinfo))
    if fileinfo.end_time >= info.start_time + dt
      push!(groups, group)
      group = Vector{Tuple{String, typeof(first_file_info)}}()
    end
  end
  if length(group) > 0
    push!(groups, group)
  end
  groups
end

function readdirall(path)
  names = String[]
  for (root, _, files) in walkdir(path)
    append!(names, joinpath.(root, files))
  end
  names
end



function read_flash_coordinates(fs)
  flash_x = Float32[]
  flash_y = Float32[]
  for f in fs
    ds = NCDataset(f)
    f_x, f_y = ds["flash_lon"][:], ds["flash_lat"][:]
    append!(flash_x, f_x)
    append!(flash_y, f_y)
    close(ds)
  end

  (
    flash_x,
    flash_y
  )
end

struct GLMGriddedProduct{S,L,F,D}
  spatial_resolution::S
  time_coverage_start::DateTime
  time_coverage_end::DateTime
  environment::D
  platform::D
  x::L
  y::L
  flash_extent_density::F
end



function sub_grid_files(fs, dt, edges)
  groups = group_filenames(fs, dt)
  x_edge, y_edge = edges
  FEDs = Array{Float32, 3}(undef, length(x_edge)-1, length(y_edge)-1, length(groups))
  time = Vector{DateTime}(undef, length(groups))
  for (i, group) in enumerate(groups)
    files = first.(group)
    metadata = last.(group)
    mean_start_time = mean(datetime2unix, x.start_time for x in metadata)
    mean_end_time = mean(datetime2unix, x.end_time for x in metadata)
    mean_time = unix2datetime(mean([mean_start_time, mean_end_time]))
    coords = read_flash_coordinates(files)
    FEDs[:, :, i] = flash_extent_density(coords, edges)

    time[i] = mean_time
  end
  FEDs, time
end

function edges_to_dims(edges)
  xedge, yedge = edges
  x_coord = (xedge[begin:end-1] .+ xedge[begin+1:end]) / 2.0
  y_coord = (yedge[begin:end-1] .+ yedge[begin+1:end]) / 2.0
  (
    x_coord,
    y_coord,
  )
end

function grid_group(edges, group, dt_grids, resolution_km)
  lon, lat = edges_to_dims(edges)
  files, metadata = group
  platform = metadata[begin].platform
  environment = metadata[begin].environment
  start_time = metadata[begin].start_time
  end_time = metadata[end].end_time
  FEDs, time_dim = sub_grid_files(files, dt_grids, edges)
  dims = (Lon(lon), Lat(lat), ClimateBase.Time(time_dim))
  filename = create_glm_filename(environment, platform, start_time, end_time)
  attrib = OrderedDict(
    "cdm_data_type"             => "Image",
    "keywords"                  => "ATMOSPHERE > ATMOSPHERIC ELECTRICITY > LIGHTNING, ATMOSPHERE > ATMOSPHERIC PHENOMENA > LIGHTNING",
    "keywords_vocabulary"       => "NASA Global Change Master Directory (GCMD) Earth Science Keywords, Version 7.0.0.0.0",
    "summary"                   => "The Lightning Detection Gridded product generates fields starting from the GLM Lightning Detection Events, Groups, Flashes product.  It consists of flash extent density.",
    "title"                     => "GLM L2 Lightning Detection Gridded Product",
    "dataset_name"              => filename,
    "orbital_slot"              => "GOES-East",
    "platform_ID"               => platform,
    "production_data_source"    => "Postprocessed",
    "spatial_resolution"        => "$(resolution_km)km at nadir",
    "time_coverage_end"         => Dates.format(end_time, "YYYY-mm-ddTHH:MM:SS.ss"),
    "time_coverage_start"       => Dates.format(start_time, "YYYY-mm-ddTHH:MM:SS.ss"),
    "timeline_id"               => "ABI Mode 3",
  )
  ClimArray(FEDs, dims; name="flash_extent_density", attrib=attrib), filename
end

function grid_files(resolution_km, fs, dt_file, dt_grids, outdir="./")
  edges = get_GOESR_edges(resolution_km)
  groups = group_filenames(fs, dt_file)
  outdirs = Set{String}()
  lon, lat = edges_to_dims(edges)
  @showprogress for group in groups
    files = first.(group)
    metadata = last.(group)
    platform = metadata[begin].platform
    environment = metadata[begin].environment
    start_time = metadata[begin].start_time
    end_time = metadata[end].end_time
    FEDs, time_dim = sub_grid_files(files, dt_grids, edges)
    dims = (Lon(lon), Lat(lat), ClimateBase.Time(time_dim))
    filename = create_glm_filename(environment, platform, start_time, end_time)
    attrib = OrderedDict(
      "cdm_data_type"             => "Image",
      "keywords"                  => "ATMOSPHERE > ATMOSPHERIC ELECTRICITY > LIGHTNING, ATMOSPHERE > ATMOSPHERIC PHENOMENA > LIGHTNING",
      "keywords_vocabulary"       => "NASA Global Change Master Directory (GCMD) Earth Science Keywords, Version 7.0.0.0.0",
      "summary"                   => "The Lightning Detection Gridded product generates fields starting from the GLM Lightning Detection Events, Groups, Flashes product.  It consists of flash extent density.",
      "title"                     => "GLM L2 Lightning Detection Gridded Product",
      "dataset_name"              => filename,
      "orbital_slot"              => "GOES-East",
      "platform_ID"               => platform,
      "production_data_source"    => "Postprocessed",
      "spatial_resolution"        => "$(resolution_km)km at nadir",
      "time_coverage_end"         => Dates.format(end_time, "YYYY-mm-ddTHH:MM:SS.ss"),
      "time_coverage_start"       => Dates.format(start_time, "YYYY-mm-ddTHH:MM:SS.ss"),
      "timeline_id"               => "ABI Mode 3",
    )
    A = ClimArray(FEDs, dims; name="flash_extent_density", attrib=attrib)
    outdir_group = joinpath(outdir, Dates.format(start_time, "YYYY/mm"))
    if outdir_group ∉ outdirs
      mkpath(outdir_group)
      push!(outdirs, outdir_group)
    end
    ncwrite_compressed(joinpath(outdir_group, filename), A)
    sleep(0.5)
  end
end

function main(resolution_km = 2.0, 
  dt = Hour(6), 
  fs = readdirall("."),
  outdir = "./"
  )
  groups = group_filenames(fs, dt)
  (x_edges, y_edges) = get_GOESR_edges(resolution_km)
  x_out, y_out = (x_edges, y_edges)

  @showprogress for group in groups
    files = first.(group)
    metadatas = last.(group)
    platform = metadatas[1].platform
    environment = metadatas[1].environment
    start_time = minimum([x.start_time for x in metadatas])
    end_time = maximum([x.end_time for x in metadatas])
    coords = read_flash_coordinates(files)
    hist = flash_extent_density(coords, (x_edges, y_edges))
    firstinfo = first(metadatas)
    griddedproduct = GLMGriddedProduct(
      resolution_km,
      start_time,
      end_time,
      environment,
      platform,
      x_out,
      y_out,
      hist
    )
    filename = create_glm_filename(environment, platform, start_time, end_time)
    pathname = joinpath(outdir, filename)
    write_ncdf(griddedproduct, pathname)
  end
end

function ncreadall(fs, var)
  ds = NCDataset(fs[1])
  lat = ds["latitude"][:]
  lon = ds["longitude"][:]
  A = Array(ds[var])
  v = Array{eltype(A)}(undef, size(A)..., length(fs))
  v[:, :, 1] = A
  times = [ds["time"][1]]
  cfvar = ds[var]
  attrib = ClimateBase.get_attributes_from_var(ds, cfvar, var)

  close(ds)

  for (i, f) in enumerate(fs[2:end])
    ds = NCDataset(f)
    @assert lat == ds["latitude"][:]
    @assert lon == ds["longitude"][:]
    v[:, :, i + 1] = Array(ds[var])
    push!(times, ds["time"][1])
    close(ds)
  end
  ClimArray(v, (Lon(lon), Lat(lat), ClimateBase.Ti(times)); name=Symbol(var), attrib=attrib)
end


function write_ncdf(data, pathname)
  ds = NCDataset(pathname,"c", attrib = OrderedDict(
    "cdm_data_type"             => "Image",
    "keywords"                  => "ATMOSPHERE > ATMOSPHERIC ELECTRICITY > LIGHTNING, ATMOSPHERE > ATMOSPHERIC PHENOMENA > LIGHTNING",
    "keywords_vocabulary"       => "NASA Global Change Master Directory (GCMD) Earth Science Keywords, Version 7.0.0.0.0",
    "summary"                   => "The Lightning Detection Gridded product generates fields starting from the GLM Lightning Detection Events, Groups, Flashes product.  It consists of flash extent density, event density, average flash area, average group area, total energy, flash centroid density, and group centroid density.",
    "title"                     => "GLM L2 Lightning Detection Gridded Product",
    "dataset_name"              => basename(pathname),
    "orbital_slot"              => "GOES-East",
    "platform_ID"               => data.platform,
    "production_data_source"    => "Postprocessed",
    "spatial_resolution"        => "$(data.spatial_resolution)km at nadir",
    "time_coverage_end"         => Dates.format(data.time_coverage_end, "YYYY-mm-ddTHH:MM:SS.ssZ"),
    "time_coverage_start"       => Dates.format(data.time_coverage_start, "YYYY-mm-ddTHH:MM:SS.ssZ"),
    "timeline_id"               => "ABI Mode 3",
  ))

  # Dimensions

  ds.dim["y"] = length(data.y)
  ds.dim["x"] = length(data.x)

  # Declare variables
  ncgoes_imager_projection = defVar(ds,"goes_imager_projection", Int32, (), attrib = OrderedDict(
    "long_name"                 => "GOES-R ABI fixed grid projection",
    "grid_mapping_name"         => "geostationary",
    "perspective_point_height"  => 3.5786023e7,
    "semi_major_axis"           => 6.378137e6,
    "semi_minor_axis"           => 6.35675231414e6,
    "inverse_flattening"        => 298.2572221,
    "latitude_of_projection_origin" => 0.0,
    "longitude_of_projection_origin" => -75.0,
    "sweep_angle_axis"          => "x",
  ))

  ncy = defVar(ds,"y", Int16, ("y",), attrib = OrderedDict(
      "_FillValue"                => Int16(-999),
      "axis"                      => "Y",
      "long_name"                 => "GOES fixed grid projection y-coordinate",
      "standard_name"             => "projection_y_coordinate",
      "units"                     => "rad",
      "add_offset"                => 0.151844,
      "scale_factor"              => -5.6e-5,
  ))

  ncx = defVar(ds,"x", Int16, ("x",), attrib = OrderedDict(
      "_FillValue"                => Int16(-999),
      "axis"                      => "X",
      "long_name"                 => "GOES fixed grid projection x-coordinate",
      "standard_name"             => "projection_x_coordinate",
      "units"                     => "rad",
      "add_offset"                => -0.151844,
      "scale_factor"              => 5.6e-5,
  ))

  ncflash_extent_density = defVar(ds,"flash_extent_density", Float32, ("x", "y"), attrib = OrderedDict(
      "_FillValue"                => Float32(0.0),
      "standard_name"             => "flash_extent_density",
      "long_name"                 => "Flash extent density",
      "units"                     => "Count per nominal   50176 microradian^2 pixel per 1.0 min",
      "grid_mapping"              => "goes_imager_projection",
  ))

  if hasfield(typeof(data), :total_energy)
    nctotal_energy = defVar(ds,"total_energy", Float32, ("x", "y"), attrib = OrderedDict(
        "_FillValue"                => Float32(0.0),
        "standard_name"             => "total_energy",
        "long_name"                 => "Total radiant energy",
        "units"                     => "nJ",
        "grid_mapping"              => "goes_imager_projection",
    ))
    nctotal_energy[:] = data.total_energy
  end

  if hasfield(typeof(data), :group_extent_density)
    ncgroup_extent_density = defVar(ds,"group_extent_density", Float32, ("x", "y"), attrib = OrderedDict(
        "_FillValue"                => Float32(0.0),
        "standard_name"             => "group_extent_density",
        "long_name"                 => "Group extent density",
        "units"                     => "Count per nominal   50176 microradian^2 pixel per 1.0 min",
        "grid_mapping"              => "goes_imager_projection",
    ))
    ncgroup_extent_density[:] = data.group_extent_density
  end




  # Define variables
  ncgoes_imager_projection[:] = rand(Int32, 1)
  ncy[:] = data.y
  ncx[:] = data.x
  ncflash_extent_density[:] = data.flash_extent_density
  # ncflash_centroid_density[:] = ...
  # ncaverage_flash_area[:] = ...
  # nctotal_energy[:] = ...
  # ncgroup_extent_density[:] = ...
  # ncgroup_centroid_density[:] = ...
  # ncaverage_group_area[:] = ...
  # ncminimum_flash_area[:] = ...

  close(ds)
end

function ncwrite_compressed(file::String, X::ClimArray; globalattr = Dict())
  ncwrite_compressed(file, (X,); globalattr)
end


function add_dimss_to_ncfile!(ds::NCDatasets.AbstractDataset, dimensions::Tuple)
  dnames = ClimateBase.dim_to_commonname.(dimensions)
  for (i, d) ∈ enumerate(dnames)
      haskey(ds, d) && continue
      v = dimensions[i].val
      # this conversion to DateTime is necessary because CFTime.jl doesn't support Date
      eltype(v) == Date && (v = DateTime.(v))
      l = length(v)
      defDim(ds, d, l) # add dimension entry
      attrib = dimensions[i].metadata
      if (isnothing(attrib) || attrib == DimensionalData.NoMetadata()) && haskey(ClimateBase.DEFAULT_ATTRIBS, d)
          attrib = ClimateBase.DEFAULT_ATTRIBS[d]
      end
      # write dimension values as a variable as well (mandatory)
      defVar(ds, d, v, (d, ); attrib = attrib)
  end
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
          defVar(ds, n, data, (dnames...,); attrib, deflatelevel=1)
      end
      close(ds)
  # end
end

function parse_fed_filename(str)
  _,_,d = split(str, '_')
  d = d[begin:end-3]
  DateTime(d, dateformat"yyyymmdd-HHMMSS")
end

function group_filenames_joao(filepaths, dt)
  group = Vector{String}()
  groups = Vector{Vector{String}}()
  for filepath in filepaths
    if length(group) == 0
      push!(group, filepath)
      continue
    end
    file_start_time = parse_fed_filename(basename(filepath))
    first_file_start_time = parse_fed_filename(basename(first(group)))
    if file_start_time >= first_file_start_time + dt
      push!(groups, group)
      group = Vector{String}()
    end
    push!(group, filepath)
  end
  if length(group) > 0
    push!(groups, group)
  end
  groups
end

function write_ncdf_group(fs, outdir = "./")
  environment = "OR"
  platform = "G16"
  climarr = ncreadall(fs, "density")
  start_time = parse_fed_filename(basename(fs[begin]))
  end_time = parse_fed_filename(basename(fs[end]))
  fname = create_glm_filename(environment, platform, start_time, end_time)
  ncwrite_compressed(joinpath(outdir, fname), climarr)
end

function run(dataset_path, resolution_km, temporal_resolution, outdir="./")
  days_folders = readdir(dataset_path; join=true)
  @showprogress for folder in days_folders
    fs = readdirall(folder)
    edges = get_GOESR_edges(resolution_km)
    metadata = parse_glm_filename.(basename.(fs))
    group = (fs, metadata)
    A, filename = grid_group(edges, group, temporal_resolution, resolution_km)
    ncwrite_compressed(joinpath(outdir, filename), A)
  end
end