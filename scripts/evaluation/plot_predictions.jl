using DrWatson
@quickactivate

if iszero(length(ARGS))
  @error "ARGS must not be empty"
  exit(1)
end

const issingle = isone(length(ARGS))

if issingle
  if !isfile(ARGS[1])
    @error "If single it must be a file"
    exit(1)
  end
else
  is_dir_ARGS = isdir.(ARGS)
  if !reduce(&, is_dir_ARGS)
    not_dirs = ARGS[(!).(is_dir_ARGS)]
    @error "All arguments must be directories" not_dirs
    exit(1)
  end
end

using Flux

include(srcdir("evaluation", "load_experiment.jl"))

@info "Loading experiments"
if issingle
  experiments = [load_experiment(ARGS[1])]
else
  experiments = load_best_experiment.(ARGS)
end
datasets = [params[:dataset] for (_, params) in experiments]
unique_datasets = unique(datasets)

if !isone(length(unique_datasets))
  model_datasets = Dict(ARGS .=> datasets)
  @error "All models should use the same dataset" model_datasets
  exit(1)
end

dataset = first(datasets)

include(srcdir("dataset", "$dataset.jl"))

dataset_params = haskey(first(experiments)[2], :eval_dataset_params) ? first(experiments)[2][:eval_dataset_params] : get_default_evaluation_params()

@info "Loading dataset"
_, test_data = get_dataset(; dataset_params...)

tx, ty = first(test_data)
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
