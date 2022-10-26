using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "--metric", "-m"
    help = "Metrics to compare"
    arg_type = Symbol
    required = true
  "--dir", "-d"
    help = "Directories of experiments to compare"
    range_tester = isdir
    action = :append_arg
  "--field"
    help = "Field to include (must be a dict)"
    arg_type = Symbol
    action = :append_arg
  "--sort"
    help = "Sorting for selecting the best experiment in a folder"
    default = "min"
    range_tester = ∈(["min", "max"])
  "--parent_dir", "-p"
    help = "Parent dir to select"
    range_tester = isdir
    action = :append_arg
  "--models_dir"
    help = "Main dir to select"
    range_tester = isdir
    arg_type = String
end

args = parse_args(s; as_symbols=true)
include(srcdir("evaluation", "load_experiment.jl"))

metrics = args[:metric]

directories = args[:dir]
parent_dirs = args[:parent_dir]
if !isnothing(args[:models_dir])
  append!(parent_dirs, readdir(args[:models_dir]; join=true))
end
if length(parent_dirs) > 0
  append!.(Ref(directories), readdir.(parent_dirs; join=true))
end
directories = unique(directories)
filter!(d -> length(readdir(d)) > 0, directories)
experiments_from_folders = getindex.(parse_best_experiment.(directories, args[:metric]; sort=args[:sort]), 2)
filter!(!isnothing, experiments_from_folders)
using OrderedCollections

function extract_relevant_fields(metadata)
  metrics = Dict(Symbol.(:metric_, metadata[:metrics]) .=> get.(Ref(metadata), metadata[:metrics], missing))
  epoch = metadata[:epoch]
  architecture = Dict(Symbol.(:architecture_, keys(metadata[:architecture])) .=> values(metadata[:architecture]))
  metadata_optimiser = metadata[:optimiser]
  if isa(metadata_optimiser, Dict)
    opt = Dict(:optimiser_type => metadata[:optimiser][:type], :optimiser_1 => "$(metadata[:optimiser][:type])($(get(metadata[:optimiser], :lr, 0)))")
  else
    opt = Dict(:optimiser_type => get(metadata, :optimiser_type, missing), (Symbol.(:optimiser_, 1:length(metadata[:optimiser])) .=> metadata[:optimiser])...)
  end
  dataset = Dict(Symbol.(:dataset_, keys(metadata[:dataset])) .=> values(metadata[:dataset]))
  id = get(metadata, :id, missing)
  OrderedDict(
    :id => id,
    architecture...,
    dataset...,
    metrics...,
    :epoch => epoch,
    opt...,
  )
end

function normalize!(dicts)
  keys_list = keys.(dicts)
  
  for k in unique(Iterators.flatten(keys_list)), d in dicts
    !haskey(d, k) && (d[k] = missing)
  end
  dicts
end

using DataFrames

relevant_fields = normalize!(extract_relevant_fields.(experiments_from_folders))

df = DataFrame(relevant_fields)

sort!(df, Symbol(:metric_, args[:metric]), rev=args[:sort]=="max")

show(df)

using InteractiveUtils

clipboard(sprint(show, "text/tab-separated-values", df))
@info "Copied to clipboard!"

