using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

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

if !isone(length(unique_datasets))
  model_datasets = Dict(experiments .=> datasets)
  @error "All models should use the same dataset" model_datasets
end

dataset = first(datasets)
dataset_type = pop!(dataset, :type)

include(srcdir("dataset", "$dataset_type.jl"))


@info "Loading dataset"
_, test_data = get_dataset(; dataset...)

tx, ty = first(test_data)

tx = copy(selectdim(tx, 4, 1:1))
ty = copy(selectdim(ty, 4, 1:1))
W,H = size(tx)[1:2]
ty = reshape(ty, W, H, :)
rtx = reshape(tx, W, H, :)

include(srcdir("utils", "plots.jl"))

@info "Making predictions"
models = [m for (m, _) in experiments]
preds = [cat(rtx, reshape(model(tx), W, H, :); dims=3) for model in models]

labels = [["model_$i" for i in 1:length(models)]..., "Truth"]
samples = [preds..., cat(rtx, ty; dims=3)]

plot_many(samples, labels; layout=(1, length(labels)), size=(900, 400))
