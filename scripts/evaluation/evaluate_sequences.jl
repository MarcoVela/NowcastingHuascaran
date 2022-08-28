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
    arg_type = Symbol
    required = true
    action = :append_arg
  "--file", "-f"
    help = "Files of models to compare"
    range_tester = isfile
    arg_type = String
    required = true
  "--override_dataset"
    help = "Overrides dataset info"
    required = false
    arg_type = Dict
    range_tester = x -> (haskey(x, :type) && haskey(x, :batchsize) && haskey(x, :splitratio))
  "--override_dataset_path"
    help = "Overrides dataset path"
    required = false
  "--clipboard"
    help = "Copy plot to clipboard"
    action = :store_true
  "--show"
    help = "Show plot on screen"
    action = :store_true
  "--subset"
    help = "Pick a subset to evaluate"
    default = -1
    arg_type = Int
end
args = parse_args(s; as_symbols=true)

using Flux

include(srcdir("evaluation", "load_experiment.jl"))

@info "Loading from files"
experiment = load_experiment(args[:file])
dataset = experiment[2][:dataset]
model = experiment[1]
dataset_path = experiment[2][:dataset_path]
dataset_type = pop!(dataset, :type)
!isnothing(dataset_path) && (dataset[:path] = dataset_path)

include(srcdir("dataset", "$dataset_type.jl"))


@info "Doing inference"
B = 32
_, test_data = get_dataset(; dataset..., batchsize=B)
test_data = collect(test_data)
if args[:subset] > 0
  subset = min(args[:subset], length(test_data))
  test_data = test_data[1:subset]
end
N = sum(x->size(x[1],4), test_data)
tx, ty = first(test_data)
W,H,C = size(tx)[1:3]
T = size(tx, 5)
test_y = zeros(Float32, W,H,C, N, T)
pred_y = deepcopy(test_y)
using ProgressMeter
Flux.testmode!(model)
@showprogress for (i,(tx, ty)) in enumerate(test_data)
  Flux.reset!(model)
  p_y = model(tx)
  pred_y[:,:,:,(i-1)*B+1:i*B,:] .= p_y[:,:,:,:,:]
  test_y[:,:,:,(i-1)*B+1:i*B,:] .= ty[:,:,:,:,:]
end

include(srcdir("evaluation", "loss.jl"))

for metric in args[:metric]
  @info "Calculating metric" metric
  metric = get_metric(metric)

  scores = metric(pred_y, test_y)
  
  using Plots
  
  if typeof(scores) <: AbstractDict
    xs = collect(keys(scores))
  else
    xs = 1:length(scores)
  end

  p = plot(scores, title=metric, xticks=xs, label=nothing; marker=:circle)
  
  if args[:clipboard]
    using ImageClipboard
    temp_path, io = mktemp()
    show(io, MIME("image/png"), p)
    close(io)
    clipboard_img(load(temp_path))
    @info "Plot copied to clipboard"
  end
  
  if args[:show]
    display(p)
    @info "Showing plot, press enter to continue..."
    readline()
  end
end

