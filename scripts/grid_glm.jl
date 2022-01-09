using StatsBase
using Dates
using NCDatasets
using ProgressMeter
using DataStructures

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
  x_pad = 100*28.0e-6
  y_pad = 100*28.0e-6

  specs = get_GOESR_grid_specifications(resolution_km)
  nx, ny = Float64(specs.pixelsEW), Float64(specs.pixelsNS)
  dx = dy = specs.resolution
  x_ctr, y_ctr = specs.centerEW, specs.centerNS
  x_edges = ((0.0:nx) .- ((nx+1)/2.)) * dx .+ x_ctr .+ dx/2.
  y_edges = ((0.0:ny) .- ((ny+1)/2.)) * dy .+ y_ctr .+ dy/2.

  n_x_pad = floor(Int64, x_pad/dx)
  n_y_pad = floor(Int64, y_pad/dy)
  x_pad = n_x_pad*dx
  y_pad = n_y_pad*dy

  x_edges = (minimum(x_edges) - x_pad):step(x_edges):(maximum(x_edges) + x_pad)
  y_edges = (minimum(y_edges) - y_pad):step(y_edges):(maximum(y_edges) + y_pad)

  return (x_edges, y_edges), (n_x_pad, n_y_pad)
end

# Usar esto para escribir al archivo netcdf
function get_out_grid(edges, pads)
  n_x_pad, n_y_pad = pads
  xedge, yedge = edges
  x_coord = ((xedge[begin:end-1] .+ xedge[begin+1:end]) / 2.0)[n_x_pad+1:end-n_x_pad]
  y_coord = ((yedge[begin:end-1] .+ yedge[begin+1:end]) / 2.0)[n_y_pad+1:end-n_y_pad]
  (
    x_coord,
    y_coord
  )
end


function flash_extent_density(vs::Tuple{N, T}, edges) where {N,T}
  hist = fit(Histogram{Float32}, vs, edges)
  hist.weights
end

function flash_extent_density(vs::Tuple{N, T}, edges, pads) where {N,T}
  n_x_pad, n_y_pad = pads
  hist = flash_extent_density(vs, edges)[n_x_pad+1:end-n_x_pad, n_y_pad+1:end-n_y_pad]
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

function ltg_ellpse_rev(date)
  if date < Date(2018,10,15)
    return 0
  else
    return 1
  end
end

const lightning_ellipse_rev = [
  # Values at launch
  (6.394140e6, 6.362755e6),
  
  # DO.07, late 2018. First Virts revision.
  # The GRS80 altitude + 6 km differs by about 3 m from the value above
  # which is the exact that was provided at the time of launch. Use the
  # original value instead of doing the math.
  # 6.35675231414e6+6.0e3
  (6.378137e6 + 14.0e3, 6.362755e6),
]

function ltg_ellps_lon_lat_to_fixed_grid(lon, lat, sat_lon, 
  ellipse_rev, 
  re_grs80=6.378137e6, 
  rp_grs80=6.35675231414e6,
  sat_grs80_height=35.786023e6)

  re_ltg_ellps, rp_ltg_ellps = lightning_ellipse_rev[begin + ellipse_rev]
  ff_ltg_ellps = (re_ltg_ellps - rp_ltg_ellps)/re_ltg_ellps
  ff_grs80 = (re_grs80 - rp_grs80)/re_grs80 # 0.003352810704800 
  sat_H = sat_grs80_height + re_grs80 # 42.164e6 

  # center longitudes on satellite, and ensure between +/- 180
  dlon = lon .- sat_lon
  dlon[dlon .< -180.] .+= 360.
  dlon[dlon .> 180.] .-= 360.

  lon_rad = deg2rad.(dlon)
  lat_rad = deg2rad.(lat)

  lat_geocent = atan.((1. - ff_grs80)^2 * tan.(lat_rad))

  # We assume geocentric latitude
  sincos_factor = sincos.(lat_geocent)
  cos_factor = last.(sincos_factor)
  sin_factor = first.(sincos_factor)

  R = re_ltg_ellps*(1-ff_ltg_ellps) ./ sqrt.(1.0 .- ff_ltg_ellps*(2.0-ff_ltg_ellps)*cos_factor.^2)
  
  lon_sincos = sincos.(lon_rad)
  lon_sin_factor = first.(lon_sincos)
  lon_cos_factor = last.(lon_sincos)
  vx = R .* cos_factor .* lon_cos_factor .- sat_H
  vy = R .* cos_factor .* lon_sin_factor
  vz = R .* sin_factor
  vmag = sqrt.(vx .^ 2 .+ vy .^ 2 .+ vz .^ 2)
  vx ./= -vmag # minus signs flip so x points to earth, z up, y left
  vy ./= -vmag
  vz ./= vmag

  # Microradians
  alpha = atan.(vz ./ vx) #* 1e6
  beta = -asin.(vy) #* 1e6
  return beta, alpha
end

function fixed_grid_to_lat_lon((x,y), lon_origin, r_eq, r_pol, perspective_point_height, semi_major_axis)
  H = perspective_point_height + semi_major_axis
  lambda_0 = (lon_origin*π)/180.0
  x_sincos = sincos.(x)
  x_sin_factor = first.(x_sincos)
  x_cos_factor = last.(x_sincos)

  y_sincos = sincos.(y)
  y_sin_factor = first.(y_sincos)
  y_cos_factor = last.(y_sincos)
  a_var = x_sin_factor .^ 2.0 .+ 
  (
    (x_cos_factor .^ 2.0) .*
    (
      y_cos_factor .^ 2.0 .+
      (
        ((r_eq*r_eq)/(r_pol*r_pol)) *
        y_sin_factor .^ 2.0
      )
    )
  )
  b_var = -2.0*H*x_cos_factor .* y_cos_factor
  c_var = (H ^ 2.0)-(r_eq ^ 2.0)
  r_s = (-b_var .- sqrt.(Complex.((b_var .^ 2) .- (4*a_var*c_var)))) ./ (2.0*a_var)
  s_x = r_s .* x_cos_factor .* y_cos_factor
  s_y = -r_s .* x_sin_factor
  s_z = r_s .* x_cos_factor .* y_sin_factor

  lat = (180.0/π) * 
  (
    atan.(
      ((r_eq*r_eq)/(r_pol*r_pol))*
      (
        s_z ./ sqrt.(((H .- s_x) .^ 2) .+ (s_y .^ 2))
      )
    )
  )
  lon = (180.0/π) * (lambda_0 .- atan.(s_y ./ (H .- s_x)))

  (lat, lon)
end

function read_flash_coordinates(fs)
  flash_x = Float32[]
  flash_y = Float32[]
  for f in fs
    ds = NCDataset(f)
    nadir_lon = ds["lon_field_of_view"][1]
    pt = ds["product_time"][1]
    ellipse_rev = ltg_ellpse_rev(pt)
    f_x, f_y = ltg_ellps_lon_lat_to_fixed_grid(
      ds["flash_lon"][:], ds["flash_lat"][:],
      nadir_lon, ellipse_rev
    )
    append!(flash_x, f_x)
    append!(flash_y, f_y)
    close(ds)
  end

  (
    flash_x,
    flash_y
  )
end

struct GLMGriddedProduct{S,T,F,D}
  spatial_resolution::S
  time_coverage_start::DateTime
  time_coverage_end::DateTime
  environment::D
  platform::D
  x::T
  y::T
  flash_extent_density::F
end


function main(resolution_km = 2.0, 
  dt = Hour(6), 
  fs = readdirall("/home/socosani/Documents/VSCode/nowcasting-huascaran/data/exp_raw/GLM-L2-LCFA/2020/001"),
  outdir = "./"
  )
  groups = group_filenames(fs, dt)
  (x_edges, y_edges), pads = get_GOESR_edges(resolution_km)
  x_out, y_out = get_out_grid((x_edges, y_edges), pads)

  @showprogress for group in groups
    files = first.(group)
    metadatas = last.(group)
    platform = metadatas[1].platform
    environment = metadatas[1].environment
    start_time = minimum([x.start_time for x in metadatas])
    end_time = maximum([x.end_time for x in metadatas])
    coords = read_flash_coordinates(files)
    hist = flash_extent_density(coords, (x_edges, y_edges), pads)
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


using ClimateBase
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
          add_dimss_to_ncfile!(ds, dims(X))
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

