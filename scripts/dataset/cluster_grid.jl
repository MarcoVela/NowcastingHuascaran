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
    arg_type = Bool
    default = true
  "--compression"
    help = "compression level for output file"
    arg_type = Int
    default = 1
    range_tester = x -> (0 <= x <= 9)
  "--folder"
    help = "Folder instead of files"
    range_tester = isdir
  "--threshold"
    help = "threshold for binarization"
    default = zero(Float32)
    arg_type = Float32
end

parsed_args = parse_args(ARGS, s; as_symbols=true)

@assert length(parsed_args[:file]) > 0 || length(readdir(parsed_args[:folder])) > 0


files = parsed_args[:file]

if length(files) === 0
  files = readdir(parsed_args[:folder]; join=true)
end

using HDF5

include(srcdir("dataset", "fed_grid.jl"))
include(srcdir("dataset", "cluster_dbscan.jl"))


for file in files
  _, input_folder_params, _ = parse_savename(dirname(file))

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
  ), sort=false, allowedtypes=[Any])
  
  parent_folder = datadir("exp_pro", "GLM-L2-LCFA-BOXES", experiment_id)
  mkpath(parent_folder)
  
  climarr = read_fed(file)
  dataset = nothing
  try
    dataset = generate_dataset(climarr, parsed_args[:dimensions];
      radius=parsed_args[:radius], 
      min_neighbors=parsed_args[:min_neighbors], 
      min_cluster_size=parsed_args[:min_cluster_size],
      t_scale=parsed_args[:time_scale],
      windows=parsed_args[:windows],
      threshold=parsed_args[:threshold],
    )
  catch
    @warn "Skipping" basename(file)
    continue
  end
  _, instance_id, _ = parse_savename(file)
  
  instance_id = (; basename=instance_id["basename"], month=instance_id["month"], compression=parsed_args[:compression])
  
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



