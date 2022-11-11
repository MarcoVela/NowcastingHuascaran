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
  "--device"
    help = "Device to run experiments on"
    default = :cpu
    arg_type = Symbol
    range_tester = in([:cpu, :gpu])
  "--persistence"
    action = :store_true
    help = "Compare against persistence method"
end
args = parse_args(s; as_symbols=true)

using Flux
const devices = Dict(:cpu => cpu, :gpu => gpu)
const device = devices[args[:device]]

include(srcdir("evaluation", "load_experiment.jl"))

@info "Loading from files"
experiment = load_experiment(args[:file])
dataset = experiment[2][:dataset]
model = experiment[1] |> device
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
Tx = size(tx, 5)
Ty = size(ty, 5)

test_y = zeros(Float32, W,H,C, N, Ty)
test_x = zeros(Float32, W,H,C, N, Tx)

pred_y = deepcopy(test_y)
using ProgressMeter
Flux.testmode!(model)
@showprogress for (i,(tx, ty)) in enumerate(test_data)
  Flux.reset!(model)
  p_y = cpu(model(device(tx)))
  pred_y[:,:,:,(i-1)*B+1:i*B,:] .= p_y[:,:,:,:,:]
  test_y[:,:,:,(i-1)*B+1:i*B,:] .= ty[:,:,:,:,:]
  test_x[:,:,:,(i-1)*B+1:i*B,:] .= tx[:,:,:,:,:]
end

test_y = device(test_y)
test_x = device(test_x)

pred_y = device(pred_y)

include(srcdir("evaluation", "loss.jl"))

persistence_model = deepcopy(test_y)
persistence_model[:,:,:,:,:] .= test_x[:,:,:,:,end:end]

for metric in args[:metric]
  if metric === :f1_threshold
    global pred_y = cpu(pred_y)
    global test_y = cpu(test_y)
  end
  @info "Calculating metric" metric
  metric = get_metric(metric)

  scores = cpu(metric(pred_y, test_y))

  using Plots
  
  if typeof(scores) <: AbstractDict
    xs = collect(keys(scores))
  else
    xs = 1:length(scores)
  end

  p = plot(scores, title=metric, xticks=xs, label="model"; marker=:circle)
  if args[:persistence]
    scores_persistence = cpu(metric(persistence_model, test_y))
    plot!(scores_persistence, label="persistence"; marker=:circle)
  end

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

