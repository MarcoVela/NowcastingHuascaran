using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

function ArgParse.parse_item(::Type{NTuple{N, T}}, x::AbstractString) where {N, T}
  parsed = [parse(T, el) for el in split(x, ',')]
  ntuple(i -> parsed[i], N)
end

@add_arg_table s begin
  "--file", "-f"
    help = "file to process (expects to be the output of grid_fed.jl)"
    arg_type = String
    action = :append_arg
    range_tester = isfile
  "--windows", "-w"
    help = "size of moving window for each dimension (lon, lat, time)"
    arg_type = NTuple{3, Int}
    default = (0, 0, 0)
    range_tester = t -> all(>=(0), t)
  "--dimensions", "-d"
    help = "size of the patches (lon ,lat, time)"
    arg_type = NTuple{3, Int}
    default = (64, 64, 20)
    range_tester = t -> all(>(0), t)
  "--radius", "-r"
    help = "radius for dbscan"
    arg_type = Float64
    required = true
  "--min_neighbors"
    help = "minimum neighbors for dbscan"
    arg_type = Int
    required = true
  "--min_cluster_size"
    help = "minimum cluster size for dbscan"
    arg_type = Int
    required = true
  "--time_scale"
    help = "factor to scale time for dbscan"
    arg_type = Float64
    default = 1.0
  "--binary"
    help = "to transform the dataset into binary classification"
    action = :store_true
  "--compression"
    help = "compression level for output file"
    arg_type = Int
    default = 1
    range_tester = x -> (0 <= x <= 9)
  "--folder"
    help = "Folder instead of files"
    range_tester = isdir
  "--threshold"
    help = "threshold for filtering and binarization"
    default = zero(Float32)
    arg_type = Float32
  "--padding"
    help = "Padding to add to windows (lon, lat, time)"
    arg_type = NTuple{3, Int}
    default = (0, 0, 0)
    range_tester = t -> all(>=(0), t)
  "--single-file"
    help = "File to output"
    arg_type = Union{String, Nothing}
    default = nothing
end

parsed_args = parse_args(ARGS, s; as_symbols=true)

@assert length(parsed_args[:file]) > 0 || length(readdir(parsed_args[:folder])) > 0


files = parsed_args[:file]

if length(files) === 0
  files = readdir(parsed_args[:folder]; join=true)
end

using HDF5

function join_all(folder, fname)
  files = readdir(folder; join=true)
  ds = Dict()
  FED = [h5read(f, "FED") for f in files]
  lon = [h5read(f, "lon") for f in files]
  lat = [h5read(f, "lat") for f in files]
  time = [h5read(f, "time") for f in files]
  ds["FED"] = cat(FED...; dims=4)
  ds["lon"] = cat(lon...; dims=2)
  ds["lat"] = cat(lat...; dims=2)
  ds["time"] = cat(time...; dims=2)
  h5open(fname, "w") do file
    for (key, val) in ds
      file[key, deflate=parsed_args[:compression]] = val
    end
  end
end


include(srcdir("dataset", "fed_grid.jl"))
include(srcdir("dataset", "cluster_dbscan.jl"))

_, input_folder_params, _ = parse_savename(parsed_args[:folder])
experiment_id = savename((; 
binary=parsed_args[:binary],
spatial=input_folder_params["spatial"],
temporal=input_folder_params["temporal"],
threshold=parsed_args[:threshold],
radius=parsed_args[:radius], 
min_neighbors=parsed_args[:min_neighbors], 
min_cluster_size=parsed_args[:min_cluster_size],
t_scale=parsed_args[:time_scale],
windows=parsed_args[:windows],
dimensions=parsed_args[:dimensions],
padding=parsed_args[:padding],
), sort=false, allowedtypes=[Any])

@info "Starting processing"
for file in files
  parent_folder = datadir("exp_pro", "GLM-L2-LCFA-BOXES", experiment_id)
  mkpath(parent_folder)

  @info "Read file $(file)"
  climarr = read_fed(file)
  dataset = nothing
  @info "Processing $(file)"
  try
    dataset = generate_dataset(climarr, parsed_args[:dimensions];
      radius=parsed_args[:radius], 
      min_neighbors=parsed_args[:min_neighbors], 
      min_cluster_size=parsed_args[:min_cluster_size],
      t_scale=parsed_args[:time_scale],
      windows=parsed_args[:windows],
      threshold=parsed_args[:threshold],
      padding=parsed_args[:padding],
    )
  catch e
    @warn "Skipping" basename(file) e
    continue
  end
  _, instance_id, _ = parse_savename(file)
  _, instance_folder_id, _ = parse_savename(dirname(file))
  instance_id = (; basename=instance_folder_id["year"], month=instance_id["month"], compression=parsed_args[:compression])
  
  filepath = datadir("exp_pro", "GLM-L2-LCFA-BOXES", experiment_id, savename(instance_id, "h5"; sort=false))
  threshold = parsed_args[:threshold]
  if parsed_args[:binary]
    @. dataset["FED"] = dataset["FED"] >= threshold
  end
  @info "saving clusters" basename(filepath)
  h5open(filepath, "w") do file
    for (key, val) in dataset
      file[key, deflate=parsed_args[:compression]] = val
    end
  end
end

if !isnothing(parsed_args[Symbol("single-file")])
  folder_path = datadir("exp_pro", "GLM-L2-LCFA-BOXES", experiment_id)
  @info "joining into single file"
  p = joinpath("data/training", parsed_args[Symbol("single-file")])
  mkpath(dirname(p))
  join_all(folder_path, p)
end


