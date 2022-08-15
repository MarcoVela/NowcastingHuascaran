using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

function ArgParse.parse_item(::Type{Dict}, x::AbstractString)
  savename_dict = DrWatson.parse_savename(x)[2]
  Dict(Symbol.(keys(savename_dict)) .=> values(savename_dict))
end

@add_arg_table s begin
  "--file", "-f"
    help = "File describing the experiment to continue"
    required = true
    range_tester = isfile
  "--epochs"
    help = "Number of epochs"
    arg_type = Int
    required = true
    range_tester = >(0)
  "--dataset", "-d"
    help = "Dataset settings"
    arg_type = Dict
    range_tester = x -> (haskey(x, :type) && haskey(x, :batchsize) && haskey(x, :splitratio))
  "--dataset_path"
    help = "Dataset path"
    required = false
  "--force"
    help = "Recreates the repository in the same status as of training"
    action = :store_true
  "--model_file"
    help = "Specific file to load"
    range_tester = isfile
  "--loss" # TODO: Revisar si será necesario pasar argumentos extras
    help = "Loss function"
    arg_type = Symbol
  "--optimiser", "-o"
    help = "Optimiser settings"
    arg_type = Dict
    range_tester = x -> (haskey(x, :type) && haskey(x, :lr))
  "--device"
    help = "Device (gpu|cpu)"
    arg_type = Symbol
    default = :gpu
end

args = parse_args(s; as_symbols=true)
include(srcdir("evaluation", "load_experiment.jl"))
include(srcdir("utils", "parse_logs.jl"))

logfile = args[:file]
override_dataset = args[:dataset]
override_dataset_path = args[:dataset_path]
force = args[:force]
model_file = args[:model_file]
epochs_continue = args[:epochs]
device_acc = args[:device]
(; first_log, train_logs, test_logs, last_log) = read_log_file(logfile)

log_args = first_log.payload

parse_dict(x::Any) = x

function parse_dict(maybe_dict::Union{Dict, NamedTuple})
  Dict([(Symbol(x), parse_dict(y)) for (x, y) in pairs(maybe_dict)])
end

log_args = parse_dict(log_args)

epochs = log_args[:epochs]
if !force && !isnothing(last_log)
  if isnothing(args[:epochs]) || (args[:epochs] < log_args[:epochs])
    @error "Training already finished after $(log_args[:epochs])"
  else
    @info "Training trained for $(log_args[:epochs]), continuing up to $(args[:epochs])"
    epochs = args[:epochs]
  end
end
prev_args = args

args = log_args

if !isnothing(model_file)
  model_raw, args = load_experiment(model_file)
else
  model_raw, model_dict = load_best_experiment(dirname(logfile), args[:loss]; sort="min")
end

if !isnothing(override_dataset)
  args[:dataset] = override_dataset
end
if !isnothing(override_dataset_path)
  args[:dataset_path] = override_dataset_path
  args[:dataset][:path] = override_dataset_path
end

original_args = deepcopy(args)
architecture_type = pop!(args[:architecture], :type)
dataset_type = pop!(args[:dataset], :type)
dataset_path = pop!(args, :dataset_path)

if !isnothing(prev_args[:optimiser])
  args[:optimiser] = prev_args[:optimiser]
end

optimiser_type = Symbol(pop!(args[:optimiser], :type))
loss_name = Symbol(args[:loss])
if !isnothing(prev_args[:loss])
  loss_name = prev_args[:loss]
end



batchsize = args[:dataset][:batchsize]
lr = args[:optimiser][:lr]

const model_id = savename((; architecture=architecture_type, dataset=dataset_type))
const experiment_id = savename((; batchsize, loss=loss_name, lr, opt=optimiser_type, gitstatus=gitdescribe()), sort=false)

@info "Including source" architecture_type dataset_type optimiser_type

include(srcdir("dataset", "$dataset_type.jl"))
include(srcdir("architecture", "$architecture_type.jl"))
include(srcdir("optimisers", "optimiser.jl"))
include(srcdir("utils", "logging.jl"))
include(srcdir("training", "train.jl"))
include(srcdir("evaluation", "loss.jl"))

using Flux
using Flux: throttle, params
using CUDA
using Statistics
using Dates

const loss_f = get_metric(Symbol(args[:loss]))

CUDA.functional(device_acc == :gpu)

const device_accelerator = device_acc == :gpu ? gpu : cpu

@info "Building model"


const model = device_accelerator(model_raw)
const ps = Flux.params(model)

show(stdout, "text/plain", cpu(model))
println(stdout)

@info "Obtaining dataset"
const train_data, test_data = @time get_dataset(; args[:dataset]...)

const opt = get_opt(optimiser_type)(; args[:optimiser]...)

function loss(X, y)
  Flux.reset!(model)
  X_dev = device_accelerator(X)
  y_pred = cpu(model(X_dev))
  loss_f(y_pred, y)
end

const logfile2 = datadir("models", model_id, experiment_id, "logs.log")

if logfile2 !== logfile
  logfile = logfile2
end
mkpath(dirname(logfile))
const logger, close_logger = get_logger(logfile)

const folder = dirname(logfile)

const train_sample_x, train_sample_y = first(train_data)
const test_sample_x, test_sample_y = first(test_data)



const epoch_losses = Float64[]

function test_loss()
  epoch_losses[end]
end

const metrics = get_metric.(Symbol.(args[:metrics]))

@info "Time of first gradient"
CUDA.@time Flux.gradient(loss, train_sample_x, train_sample_y);

function log_loss(epoch)
  val_test, exec_time_test = CUDA.@timed loss(test_sample_x, test_sample_y)
  val_train, exec_time_train = CUDA.@timed loss(train_sample_x, train_sample_y)
  exec_time = mean((exec_time_test, exec_time_train))
  Base.with_logger(logger) do 
    @info "LOSS_DURING_TRAIN" test_loss=val_test train_loss=val_train epoch exec_time
  end
end

prev_experiments = length(readdir(folder))

Base.with_logger(logger) do 
  @info "CONTINUE_TRAIN" train_size=length(train_data) test_size=length(test_data) original_args...
end

@info "Starting training for $(args[:epochs]) epochs"

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

for epoch in prev_experiments:(prev_experiments+epochs_continue)
  log_loss_cb = throttle(() -> log_loss(epoch), args[:throttle])
  @info "Training..." epoch
  train_time = CUDA.@elapsed train_single_epoch!(ps, loss, train_data, opt, cb=log_loss_cb)
  Base.with_logger(logger) do
    @info "EPOCH_TRAIN" epoch exec_time=train_time
  end
  @info "Testing..." epoch
  metrics_dict, test_time = CUDA.@timed metrics_single_epoch(model, metrics, ((device_accelerator(X), y) for (X,y) in test_data))
  Flux.reset!(model)
  original_metrics = deepcopy(metrics_dict)
  metrics_dict[:test_loss] = metrics_dict[loss_name]
  push!(epoch_losses, metrics_dict[loss_name])
  Base.with_logger(logger) do
    @info "EPOCH_TEST" epoch exec_time=test_time metrics_dict...
  end
  args_dict = merge(deepcopy(original_args), metrics_dict)
  args_dict[:model] = cpu(model)
  args_dict[:epoch] = epoch
  args_dict[:date] = Dates.now()
  args_dict[:architecture][:type] = architecture_type
  args_dict[:dataset][:type] = dataset_type
  args_dict[:optimiser][:type] = optimiser_type
  @tag!(args_dict, storepatch=true)

  iteration_id = savename((; epoch, original_metrics...), "bson"; digits=5, sort=false)

  filename = joinpath(folder, iteration_id)
  safesave(filename, args_dict)

  stop_callbacks_result = [callback() for callback in stop_callbacks]
  if any(stop_callbacks_result)
    stop_cause = stop_functions[stop_callbacks_result]
    @info "STOP_TRAIN" epoch stop_cause
    break
  end
end

@info "Finished training"

Base.with_logger(logger) do 
  @info "FINISH" 
end

close_logger()
