using DrWatson
@quickactivate

const architecture = "convlstm_enc_dec_01"

include(srcdir("dataset", "moving_mnist.jl"))
include(srcdir("layers", "ConvLSTM2D.jl"))
include(srcdir("models", "$architecture.jl"))
include(srcdir("utils", "logging.jl"))
include(srcdir("training", "train.jl"))

using Base: @kwdef
using Flux
using Flux.Losses: binarycrossentropy, logitbinarycrossentropy
using Flux: throttle, params
using CUDA
using ProgressMeter
using Statistics
using Flux.Optimise
using NamedTupleTools

@kwdef mutable struct Args{O, F}
  lr::Float64 = 2e-4  # Learning rate
  batchsize::Int = 4  # Batch size
  throttle::Int = 30  # Throttle timeout
  epochs::Int = 2     # Number of Epochs
  split_ratio::Float64 = .9
  opt::O
  loss::F
end

DrWatson.default_allowed(::Args) = (Any,)
DrWatson.allaccess(::Args) = (:lr, :opt, :batchsize, :loss)

const args = Args(opt=ADAM, loss=binarycrossentropy)


const N = 12
const device = gpu

const model = build_model(; N_out=20-N, device=device)
const train_data, test_data = get_dataset(; train_test_ratio=args.split_ratio, batchsize=args.batchsize, N=N)


function loss(X, y)
  Flux.reset!(model)
  X_dev = device(X)
  y_pred = cpu(model(X_dev))
  args.loss(y_pred, y)
end

train_x, train_y = first(train_data)
test_x, test_y = first(test_data)

const logger, close_logger = get_logger()

Base.with_logger(logger) do 
  nt = ntfromstruct(args)
  @info "START_PARAMS" train_size=length(train_data) test_size=length(test_size) nt...
end

function log_test_loss(epoch)
  val, exec_time = CUDA.@timed loss(test_x, test_y)
  Base.with_logger(logger) do 
    @info "TEST_LOSS_DURING_TRAIN" loss=val epoch exec_time
  end
end

function log_train_loss(epoch)
  val, exec_time = CUDA.@timed loss(train_x, train_y)
  Base.with_logger(logger) do 
    @info "TRAIN_LOSS_DURING_TRAIN" loss=val epoch exec_time
  end
end

const ps = params(model)
const opt = args.opt(args.lr)


for epoch in 1:args.epochs
  log_test_loss_cb = throttle(() -> log_test_loss(epoch), args.throttle)
  log_train_loss_cb = throttle(() -> log_train_loss(epoch), args.throttle)
  p = Progress(length(train_data); showspeed=true)
  callbacks = Flux.runall([log_test_loss_cb, log_train_loss_cb])
  train_time = CUDA.@elapsed train_single_epoch!(ps, loss, train_data, opt, cb=callbacks)
  Base.with_logger(logger) do
    @info "EPOCH_TRAIN" epoch exec_time=train_time
  end
  test_losses, test_time = CUDA.@timed [loss(X, y) for (X, y) in test_data]
  mean_loss = mean(test_losses)

  Base.with_logger(logger) do
    @info "EPOCH_TEST" epoch mean_loss=mean_loss var_loss=var(test_losses) exec_time=test_time
  end
  foldername = savename(args)
  args_dict = convert(Dict, ntfromstruct(args))
  @tag!(args_dict, storepatch=true)
  Flux.reset!(model)
  args_dict[:model] = cpu(model)
  args_dict[:loss] = mean_losses
  filename = datadir("models", architecture, foldername, savename((loss=mean_loss, epoch), "bson"; digits=5))
  safesave(filename, args_dict)
end

Base.with_logger(logger) do 
  @info "FINISH" 
end

