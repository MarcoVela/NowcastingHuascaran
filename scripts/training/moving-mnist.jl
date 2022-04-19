using DrWatson
@quickactivate
using NPZ
using Parameters: @with_kw
@with_kw mutable struct Args
  lr::Float64 = 1e-2  # Learning rate
  batchsize::Int = 2  # Batch size
  throttle::Int = 30  # Throttle timeout
  epochs::Int = 2     # Number of Epochs
end


args = Args()
mnist_whole = @time NPZ.npzread(datadir("exp_raw", "moving-mnist", "mnist_test_seq.npy")) ./ Float32(255);
# T,N,H,W = size(mnist_whole)
# C = 1
# mnist_whole = reverse!(reshape(mnist_whole, W, H, C, N, T));
mnist_whole = reshape(mnist_whole, size(mnist_whole)[1:2]..., 1, size(mnist_whole)[3:4]...);
mnist_whole = permutedims(mnist_whole, reverse(1:ndims(mnist_whole)));
@show size(mnist_whole);
train_test_split = .8
TOTAL_SAMPLES = size(mnist_whole, 4)
mnist_train = view(mnist_whole, :, :, :, 1:Int(TOTAL_SAMPLES * train_test_split), :);
mnist_test = view(mnist_whole, :, :, :, Int(TOTAL_SAMPLES * train_test_split)+1:TOTAL_SAMPLES, :);

@show size(mnist_train)
@show size(mnist_test);
function broadcasted_σ(x)
  Flux.σ.(x)
end

include(srcdir("layers", "SimpleConvLSTM2D.jl"))
using Flux.Losses: binarycrossentropy

const device = gpu
const n = N = 10



function keep_last(x)
  local n = 1
  local N = length(size(x))
  inds_before = ntuple(Returns(:), N - 1)
  x[inds_before..., size(x, N)-n+1:size(x, N)]
end

function _catn(x::AbstractArray{T, N}...) where {T, N}
  cat(x...; dims=N)
end

function RepeatVector(n)
  Parallel(_catn, ntuple(Returns(identity), n))
end

function repeat_input(x)
  h = map(_ -> copy(x), Base.OneTo(n))
  sze = size(x)
  reshape(reduce(hcat, h), sze[1], sze[2], sze[3], length(h))
end

const t_steps = 10

struct MergeLastDims{N}
  n::N
  function MergeLastDims(_n)
    new{_n}()
  end
end

function (m::MergeLastDims{N2})(x::AbstractArray{T, N}) where {T,N,N2}
  reshape(x, size(x)[1:N-N2]..., :)
end

# Flux.@functor MergeLastDims

struct Reshape{D}
  dims::D
end

function (r::Reshape)(x)
  reshape(x, r.dims)
end

# Flux.@functor Reshape

const model = Chain(
  SimpleConvLSTM2D((64, 64), (5, 5), 1 => 64, pad=SamePad()),
  SimpleConvLSTM2D((64, 64), (3, 3), 64 => 64, pad=SamePad()),
  keep_last,
  RepeatVector(10),
  SimpleConvLSTM2D((64, 64), (1, 1), 64 => 64, pad=SamePad()),
  SimpleConvLSTM2D((64, 64), (1, 1), 64 => 64, pad=SamePad()),
  MergeLastDims(2),
  Conv((3, 3), 64 => 1, σ, pad=SamePad()),
  Reshape((64, 64, 1, :, 10)),
) |> device


using CUDA
using Statistics

function loss(X, y)
  Flux.reset!(model)
  X_dev = device(X)
  y_pred = cpu(model(X_dev))
  binarycrossentropy(y_pred, y)
end

using Flux.Data: DataLoader
mnist_x, mnist_y = copy(view(mnist_train, :, :, :, 1:128, 1:N)), copy(view(mnist_train, :, :, :, 1:128, N+1:20));

const batched = true
if batched
  x_d = (copy(view(mnist_train, :, :, :, t:t+args.batchsize-1, 1:N)) for t in 1:args.batchsize:size(mnist_x, 4))
  y_d = (copy(view(mnist_train, :, :, :, t:t+args.batchsize-1, N+1:20)) for t in 1:args.batchsize:size(mnist_y, 4))
  data = zip(x_d, y_d)
else
  data = zip(
    (copy(view(mnist_train, :, :, :, 1:N, t)) for t in axes(mnist_x, 5)),
    (copy(view(mnist_train, :, :, :, N+1:20, t)) for t in axes(mnist_y, 5))
  )
end

tx, ty = (copy(view(mnist_test, :,:,:,1:2,1:N)), copy(view(mnist_test, :,:,:,1:2,N+1:20)));

evalcb = () -> @show loss(tx, ty)

using Flux.Optimise: ADAM
opt = ADAM(args.lr)

using Flux: throttle, params
p = params(model);
p.params

println("Starting training!")

Flux.train!(loss, p, data, opt; cb=throttle(evalcb, 30))

# gs = gradient(ps) do
#   loss(batchmemaybe(d)...)
# end


model
using Plots

function plot_results(y_pred,y)
  @gif for i = axes(y_pred, 3)
    p1 = heatmap(y_pred[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    p2 = heatmap(y[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    plot(p1,p2, size=(800, 400))
  end
end