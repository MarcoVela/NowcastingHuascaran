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
      arg_type = Dict
      required = true
      range_tester = x -> (haskey(x, :type) && haskey(x, :lr))
  "--architecture", "-a"
      help = "Architecture settings"
      arg_type = Dict
      required = true
      range_tester = Base.Fix2(haskey, :type)
  "--dataset", "-d"
      help = "Dataset settings"
      arg_type = Dict
      required = true
      range_tester = x -> (haskey(x, :type) && haskey(x, :batchsize) && haskey(x, :splitratio))
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
original_args = deepcopy(args)

architecture_type = pop!(args[:architecture], :type)
dataset_type = pop!(args[:dataset], :type)

optimiser_type = pop!(args[:optimiser], :type)
loss_name = args[:loss]
batchsize = args[:dataset][:batchsize]
lr = args[:optimiser][:lr]


const model_id = savename((; architecture=architecture_type, dataset=dataset_type))
const experiment_id = savename((; batchsize, loss=loss_name, lr, opt=optimiser_type))

@info "Including source" architecture_type dataset_type optimiser_type

include(srcdir("dataset", "$dataset_type.jl"))
include(srcdir("architecture", "$architecture_type.jl"))
include(srcdir("optimisers", "$optimiser_type.jl"))
include(srcdir("utils", "logging.jl"))
include(srcdir("training", "train.jl"))
include(srcdir("evaluation", "loss.jl"))

using Flux
using Flux: throttle, params
using CUDA
using Statistics
using Dates

const loss_f = get_metric(args[:loss])

CUDA.functional(args[:device] == :gpu)

const device = args[:device] == :gpu ? gpu : cpu

@info "Building model"
const model = device(build_model(; args[:architecture]...))
show(stdout, "text/plain", cpu(model))
println(stdout)

@info "Obtaining dataset"
const train_data, test_data = @time get_dataset(; args[:dataset]...)

function loss(X, y)
  Flux.reset!(model)
  X_dev = device(X)
  y_pred = cpu(model(X_dev))
  loss_f(y_pred, y)
end

const logfile = datadir("models", model_id, experiment_id, "logs.log")
ispath(dirname(logfile)) && error("Folder $(dirname(logfile)) must be empty")
mkpath(dirname(logfile))
isfile(logfile) && Base.unlink(logfile)
const logger, close_logger = get_logger(logfile)

const train_sample_x, train_sample_y = first(train_data)
const test_sample_x, test_sample_y = first(test_data)

function log_loss(epoch)
  val_test, exec_time_test = CUDA.@timed loss(test_sample_x, test_sample_y)
  val_train, exec_time_train = CUDA.@timed loss(train_sample_x, train_sample_y)
  exec_time = mean((exec_time_test, exec_time_train))
  Base.with_logger(logger) do 
    @info "LOSS_DURING_TRAIN" test_loss=val_test train_loss=val_train epoch exec_time
  end
end

const ps = params(model)
const opt = get_opt(; args[:optimiser]...)

@info "Time of first gradient"
CUDA.@time Flux.gradient(loss, train_sample_x, train_sample_y);

Base.with_logger(logger) do 
  @info "START_PARAMS" train_size=length(train_data) test_size=length(test_data) original_args...
end

@info "Starting training for $(args[:epochs]) epochs"

const metrics = get_metric.(args[:metrics])

stop_callbacks = []
if args[:early_stop] > 0
  push!(stop_callbacks, Flux.early_stopping(loss, args[:early_stop]; init_score=Inf))
end
if args[:plateau] > 0
  push!(stop_callbacks, Flux.plateau(loss, args[:early_stop]; init_score=Inf))
end

for epoch in 1:args[:epochs]
  log_loss_cb = throttle(() -> log_loss(epoch), args[:throttle])
  @info "Training..." epoch
  train_time = CUDA.@elapsed train_single_epoch!(ps, loss, train_data, opt, cb=log_loss_cb)
  Base.with_logger(logger) do
    @info "EPOCH_TRAIN" epoch exec_time=train_time
  end
  @info "Testing..." epoch
  metrics_dict, test_time = CUDA.@timed metrics_single_epoch(model, metrics, ((device(X), y) for (X,y) in test_data))
  Flux.reset!(model)
  Base.with_logger(logger) do
    @info "EPOCH_TEST" epoch exec_time=test_time metrics_dict...
  end
  args_dict = merge(deepcopy(args), metrics_dict)
  args_dict[:model] = cpu(model)
  args_dict[:epoch] = epoch
  args_dict[:date] = Dates.now()
  args_dict[:architecture][:type] = architecture_type
  args_dict[:dataset][:type] = dataset_type
  args_dict[:optimiser][:type] = optimiser_type
  @tag!(args_dict, storepatch=true)

  iteration_id = savename((; epoch, metrics_dict...), "bson"; digits=5, sort=false)

  filename = datadir("models", model_id, experiment_id, iteration_id)
  safesave(filename, args_dict)

  stop_callbacks_result = [callback() for callback in stop_callbacks]
  if any(stop_callbacks_result)
    stop_cause = stop_callbacks[stop_callbacks_result]
    @info "STOP_TRAIN" epoch stop_cause
    break
  end
end

@info "Finished training"

Base.with_logger(logger) do 
  @info "FINISH" 
end

close_logger()