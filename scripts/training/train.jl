using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

function ArgParse.parse_item(::Type{NamedTuple}, x::AbstractString)
    savename_dict = DrWatson.parse_savename(x)[2]
    (; Dict(Symbol.(keys(savename_dict)) .=> values(savename_dict))...)
end

function ArgParse.parse_item(::Type{Dict}, x::AbstractString)
  savename_dict = DrWatson.parse_savename(x)[2]
  Dict(Symbol.(keys(savename_dict)) .=> values(savename_dict))
end

function ArgParse.parse_item(::Type{Vector{Symbol}}, x::AbstractString)
  isempty(x) && return Symbol[]
  Symbol.(split(x, ','))
end


@add_arg_table s begin
  "--optimiser", "-o"
      help = "Optimiser settings"
      required = true
      arg_type = String
      action = :append_arg
  "--architecture", "-a"
      help = "Architecture settings"
      arg_type = Dict
      required = true
      range_tester = Base.Fix2(haskey, :type)
  "--base_model"
      help = "Path of base artifact that will be passed"
      range_tester = isfile
      required = false
  "--dataset", "-d"
      help = "Dataset settings"
      arg_type = Dict
      required = true
      range_tester = x -> (haskey(x, :type) && haskey(x, :batchsize) && haskey(x, :splitratio))
  "--dataset_path"
      help = "Dataset path"
      required = false
  "--epochs"
      help = "Number of epochs"
      arg_type = Int
      required = true
      range_tester = >(0)
  "--device"
      help = "Device (gpu|cpu)"
      arg_type = Symbol
      default = :gpu
  "--loss" # TODO: Revisar si será necesario pasar argumentos extras
      help = "Loss function"
      arg_type = Symbol
      required = true
  "--metrics"
      help = "Metrics to monitor in testing"
      arg_type = Vector{Symbol}
      required = true
      range_tester = x -> length(x) > 0
  "--throttle"
      help = "Time in seconds for execution of callbacks"
      arg_type = Int
      default = 30
  "--early_stop"
      help = "Number of epochs to early stop training (0 to deactivate)"
      arg_type = Int
      default = 0
      range_tester = >=(0)
  "--plateau"
      help = "Number of epochs to stop on plateau training (0 to deactivate)"
      arg_type = Int
      default = 0
      range_tester = >=(0)
end

args = parse_args(s; as_symbols=true)
loss_name = args[:loss]
if loss_name ∉ args[:metrics]
  @warn "Loss function should be monitored as a metric, including"
  push!(args[:metrics], loss_name)
end

original_args = deepcopy(args)

architecture_type = pop!(args[:architecture], :type)
dataset_type = pop!(args[:dataset], :type)
dataset_path = pop!(args, :dataset_path)
if !isnothing(dataset_path)
  args[:dataset][:path] = dataset_path
end

optimisers = args[:optimiser]

if length(optimisers) == 1
  optimiser_type = only(optimisers)
else
  optimiser_type = "Optimiser"
end

loss_name = args[:loss]
batchsize = args[:dataset][:batchsize]

using UUIDs

const exp_id=string(uuid4())
const model_id = savename((; architecture=architecture_type, dataset=dataset_type))
const experiment_id = savename((; loss=loss_name, batchsize, opt=optimiser_type, id=exp_id), sort=false)

@info "Including source" architecture_type dataset_type

include(srcdir("dataset", "$dataset_type.jl"))
include(srcdir("architecture", "$architecture_type.jl"))
include(srcdir("optimisers", "optimiser.jl"))
include(srcdir("utils", "logging.jl"))
include(srcdir("training", "train.jl"))
include(srcdir("evaluation", "loss.jl"))

const opt = build_optimiser(optimisers)

using Flux
using Flux: throttle, params
using CUDA
using Statistics
using Dates

const loss_f = get_metric(args[:loss])

CUDA.functional(args[:device] == :gpu)

const accel_device = args[:device] == :gpu ? gpu : cpu

@info "Building model"
if !isnothing(args[:base_model])
  using BSON
  args[:architecture][:base_model] = BSON.load(args[:base_model])[:model]
end

const model, ps = build_model(; device=accel_device, args[:architecture]...)
show(stdout, "text/plain", cpu(model))
println(stdout)

@info "Obtaining dataset"
const train_data, test_data = @time get_dataset(; args[:dataset]...)


function loss(X, y)
  Flux.reset!(model)
  X_dev = accel_device(X)
  y_pred = cpu(model(X_dev))
  loss_f(y_pred, y)
end

const logfile = datadir("models", model_id, experiment_id, "logs.log")


const train_sample_x, train_sample_y = first(train_data)
const test_sample_x, test_sample_y = first(test_data)



const epoch_losses = Float64[]

function test_loss()
  epoch_losses[end]
end

const metrics = get_metric.(args[:metrics])

@info "Time of first gradient"
CUDA.@time Flux.gradient(loss, train_sample_x, train_sample_y);

ispath(dirname(logfile)) && error("Folder $(dirname(logfile)) must be empty")
mkpath(dirname(logfile))
isfile(logfile) && Base.unlink(logfile)
const logger, close_logger = get_logger(logfile)

train_losses = Vector{Float32}()
test_losses = Vector{Float32}()

function log_loss(epoch)
  val_test, exec_time_test = CUDA.@timed loss(test_sample_x, test_sample_y)
  val_train, exec_time_train = CUDA.@timed loss(train_sample_x, train_sample_y)
  exec_time = mean((exec_time_test, exec_time_train))
  push!(train_losses, val_train)
  push!(test_losses, val_test)
  Base.with_logger(logger) do 
    @info "LOSS_DURING_TRAIN" test_loss=val_test train_loss=val_train epoch exec_time
  end
end

Base.with_logger(logger) do 
  @info "START_PARAMS" train_size=length(train_data) test_size=length(test_data) original_args...
end

@info "Starting training for $(args[:epochs]) epochs" id=exp_id

stop_callbacks = []
stop_functions = [
  Flux.early_stopping,
  Flux.plateau,
]
if get!(args, :early_stop, 0) > 0
  push!(stop_callbacks, Flux.early_stopping(test_loss, args[:early_stop]; init_score=Inf))
end
if get!(args, :plateau, 0) > 0
  push!(stop_callbacks, Flux.plateau(test_loss, args[:early_stop]; init_score=Inf, min_dist=2f-5))
end

for epoch in 1:args[:epochs]
  log_loss_cb = throttle(() -> log_loss(epoch), args[:throttle])
  @info "Training..." epoch
  trainmode!(model)
  train_time = CUDA.@elapsed train_single_epoch!(ps, loss, train_data, opt, cb=log_loss_cb)
  Base.with_logger(logger) do
    @info "EPOCH_TRAIN" epoch exec_time=train_time
  end
  @info "Metrics during train" mean_train_loss=mean(train_losses) mean_test_loss=mean(test_losses)

  @info "Testing..." epoch
  testmode!(model)
  metrics_dict, test_time = CUDA.@timed metrics_single_epoch(model, metrics, ((accel_device(X), y) for (X,y) in test_data))
  Flux.reset!(model)
  original_metrics = deepcopy(metrics_dict)
  metrics_dict[:test_loss] = metrics_dict[loss_name]
  push!(epoch_losses, metrics_dict[loss_name])
  @info "Metrics during test" metrics_dict...

  Base.with_logger(logger) do
    @info "EPOCH_TEST" epoch exec_time=test_time metrics_dict...
  end
  args_dict = merge(deepcopy(original_args), metrics_dict)
  args_dict[:model] = cpu(model)
  args_dict[:epoch] = epoch
  args_dict[:date] = Dates.now()
  args_dict[:architecture][:type] = architecture_type
  args_dict[:dataset][:type] = dataset_type
  args_dict[:optimiser_type] = optimiser_type
  args_dict[:id] = exp_id
  @tag!(args_dict, storepatch=true)

  iteration_id = savename((; epoch, original_metrics...), "bson"; digits=5, sort=false)

  filename = datadir("models", model_id, experiment_id, iteration_id)
  safesave(filename, args_dict)

  stop_callbacks_result = [callback() for callback in stop_callbacks]
  if any(stop_callbacks_result)
    stop_cause = stop_functions[stop_callbacks_result]
    @info "STOP_TRAIN" epoch stop_cause
    break
  end
  empty!(train_losses)
  empty!(test_losses)
end

@info "Finished training" datetime=now()

Base.with_logger(logger) do 
  @info "FINISH" 
end

close_logger()
