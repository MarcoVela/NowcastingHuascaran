using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

function ArgParse.parse_item(::Type{Dict}, x::AbstractString)
  savename_dict = DrWatson.parse_savename(x)[2]
  Dict(Symbol.(keys(savename_dict)) .=> values(savename_dict))
end

@add_arg_table s begin
  "--metric", "-m"
    help = "Metric to select best experiment in a folder"
    default = "epoch"
  "--sort"
    help = "Sorting for selecting the best experiment in a folder"
    default = "min"
    range_tester = ∈(["min", "max"])
  "--file", "-f"
    help = "Files of models to compare"
    range_tester = isfile
    action = :append_arg
  "--dir", "-d"
    help = "Directories of experiments to compare"
    range_tester = isdir
    action = :append_arg
  "--override_dataset"
    help = "Overrides dataset info"
    required = false
    arg_type = Dict
    range_tester = x -> (haskey(x, :type) && haskey(x, :batchsize) && haskey(x, :splitratio))
  "--override_dataset_path"
    help = "Overrides dataset path"
    required = false
  "--test_index"
    help = "Index of test dataset sample"
    default = 1
    arg_type = Int
    range_tester = >(0)
end
args = parse_args(s; as_symbols=true)

using Flux

include(srcdir("evaluation", "load_experiment.jl"))

experiments_from_files = []
experiments_from_folders = []
@info "Loading experiments"

if !iszero(length(args[:file]))
  @info "Loading from files"
  experiments_from_files = load_experiment.(args[:file])
end

if !iszero(length(args[:dir]))
  metric = args[:metric]
  sorting = args[:sort]
  @info "Loading best experiment from folders" metric sorting 
  experiments_from_folders = load_best_experiment.(args[:dir], args[:metric]; sort=args[:sort])
end

experiments = [experiments_from_files..., experiments_from_folders...]
datasets = [params[:dataset] for (_, params) in experiments]
unique_datasets = unique(x -> x[:type], datasets)

if isnothing(args[:override_dataset]) && !isone(length(unique_datasets))
  model_datasets = Dict(experiments .=> datasets)
  @error "All models should use the same dataset" model_datasets
end

dataset_path = first([params[:dataset_path] for (_, params) in experiments])
dataset = first(datasets)
dataset_type = pop!(dataset, :type)
dataset[:path] = dataset_path

include(srcdir("dataset", "$dataset_type.jl"))


@info "Loading dataset"
_, test_data = get_dataset(; dataset...)

tx, ty = collect(test_data)[args[:test_index]]

tx = copy(selectdim(tx, 4, 1:1))
ty = copy(selectdim(ty, 4, 1:1))
W,H = size(tx)[1:2]
ty = reshape(ty, W, H, :)
rtx = reshape(tx, W, H, :)

include(srcdir("utils", "plots.jl"))

@info "Making predictions"
models = [m for (m, _) in experiments]
preds = [cat(rtx, reshape(model(tx), W, H, :); dims=3) for model in models]

Ti_truth = size(reshape(first(models)(tx), W, H, :), 3)

labels = [["model_$i" for i in 1:length(models)]..., "Truth"]
samples = [preds..., cat(rtx, ty; dims=3)]

plot_many(samples, labels, Ti_truth; layout=(1, length(labels)), size=(900, 400))
