using DrWatson
@quickactivate

const architecture = get(ENV, "architecture", nothing) #"convlstm_enc_dec_01"
const optimiser = get(ENV, "optimiser", nothing) #"ADAMExpDecay"
if isnothing(architecture)
  @error "architecture env var must not be empty"
  exit(1)
end
if isnothing(optimiser)
  @error "optimiser env var must not be empty"
  exit(1)
end
const dataset = "moving_mnist"

@info "Including source" architecture dataset optimiser

include(srcdir("dataset", "$dataset.jl"))
include(srcdir("models", "$architecture.jl"))
include(srcdir("utils", "logging.jl"))
include(srcdir("training", "train.jl"))
include(srcdir("evaluation", "metrics.jl"))
include(srcdir("optimisers", "$optimiser.jl"))

using Base: @kwdef
using Flux
using Flux.Losses: binarycrossentropy, logitbinarycrossentropy
using Flux: throttle, params
using CUDA
using Statistics
using NamedTupleTools
using Dates

@kwdef mutable struct Args{F}
  lr::Float64 = 2e-4  # Learning rate
  batchsize::Int = 1  # Batch size
  throttle::Int = 30  # Throttle timeout
  epochs::Int = 2     # Number of Epochs
  split_ratio::Float64 = .9
  dropout::Float64 = .2
  opt::String
  loss::F
end

DrWatson.default_allowed(::Args) = (Any,)
DrWatson.allaccess(::Args) = (:lr, :opt, :batchsize, :loss)
const args = Args(loss=binarycrossentropy, opt=optimiser)
args.lr = DrWatson.readenv("lr", args.lr)
args.batchsize = DrWatson.readenv("batchsize", args.batchsize)
args.throttle = DrWatson.readenv("throttle", args.throttle)
args.epochs = DrWatson.readenv("epochs", args.epochs)
args.split_ratio = DrWatson.readenv("split_ratio", args.split_ratio)
args.dropout = DrWatson.readenv("dropout", args.dropout)
const foldername = savename(args)
const TOTAL_FRAMES = 20
const N = 12
const device = CUDA.functional(true) ? gpu : cpu

@info "Building model"

model = build_model(; N_out=TOTAL_FRAMES-N, device=device, dropout=args.dropout)

show(stdout, "text/plain", cpu(model))
println(stdout)

@info "Obtaining dataset"

const train_data, test_data = @time get_dataset(; train_test_ratio=args.split_ratio, batchsize=args.batchsize, N=N)


function loss(X, y)
  Flux.reset!(model)
  X_dev = device(X)
  y_pred = cpu(model(X_dev))
  args.loss(y_pred, y)
end

train_x, train_y = first(train_data)
test_x, test_y = first(test_data)

const logfile = datadir("models", 
                     savename((; architecture, dataset); connector="-"), 
                     foldername, 
                     "logs.log")
mkpath(dirname(logfile))
const logger, close_logger = get_logger(logfile)

Base.with_logger(logger) do 
  nt = ntfromstruct(args)
  @info "START_PARAMS" train_size=length(train_data) test_size=length(test_data) nt...
end

function log_loss(epoch)
  val_test, exec_time_test = CUDA.@timed loss(test_x, test_y)
  val_train, exec_time_train = CUDA.@timed loss(train_x, train_y)
  exec_time = mean((exec_time_test, exec_time_train))
  Base.with_logger(logger) do 
    @info "LOSS_DURING_TRAIN" test_loss=val_test train_loss=val_train epoch exec_time
  end
end

const ps = params(model)
const opt = get_opt(args.lr)

@info "Time of first gradient"
@time Flux.gradient(loss, train_x, train_y);

@info "Starting training for $(args.epochs) epochs"

function bestcsi(x, y)
  thresholds = 0:.05:1
  y = y .> .8
  max_csi, i = findmax(thresholds) do t
    csi(x .> t, y)
  end
  (val=max_csi, thrs=thresholds[i])
end

metrics = [binarycrossentropy, bestcsi]

for epoch in 1:args.epochs
  log_loss_cb = throttle(() -> log_loss(epoch), args.throttle)
  @info "Training..." epoch
  train_time = CUDA.@elapsed train_single_epoch!(ps, loss, train_data, opt, cb=log_loss_cb)
  Base.with_logger(logger) do
    @info "EPOCH_TRAIN" epoch exec_time=train_time
  end
  @info "Testing..." epoch
  Flux.reset!(model)
  metrics_dict, test_time = CUDA.@timed metrics_single_epoch(model, metrics, ((device(X), y) for (X,y) in test_data))
  Flux.reset!(model)
  Base.with_logger(logger) do
    @info "EPOCH_TEST" epoch exec_time=test_time metrics_dict...
  end
  args_dict = convert(Dict, ntfromstruct(args))
  args_dict = Dict{Symbol, Any}(args_dict)
  merge(args_dict, metrics_dict)
  args_dict[:model] = cpu(model)
  args_dict[:epoch] = epoch
  args_dict[:date] = Dates.now()
  args_dict[:architecture] = architecture
  args_dict[:dataset] = dataset
  args_dict[:opt] = get_opt(args.lr)
  @tag!(args_dict, storepatch=true)
  filename = datadir("models", 
                     savename((; architecture, dataset); connector="-"), 
                     foldername, 
                     savename((; epoch, metrics_dict...), "bson"; digits=5, sort=false))
  safesave(filename, args_dict)
end

@info "Finished training"

Base.with_logger(logger) do 
  @info "FINISH" 
end

close_logger()
