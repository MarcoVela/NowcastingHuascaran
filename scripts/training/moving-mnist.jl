using DrWatson
@quickactivate
using NPZ
using Parameters: @with_kw
@with_kw mutable struct Args
  lr::Float64 = 5e-2  # Learning rate
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




N = 10


const model = Chain(
  SimpleConvLSTM2D((64, 64), (5, 5), 1 => 32, pad=SamePad(), activation=Flux.relu, bias=true),
  KeepLast(
    SimpleConvLSTM2D((64, 64), (3, 3), 32 => 32, pad=SamePad(), activation=Flux.relu, bias=true)
  ),
  RepeatInput(
    10, 
    SimpleConvLSTM2D((64, 64), (1, 1), 32 => 32, pad=SamePad(), activation=Flux.relu, bias=true)
  ),
  SimpleConvLSTM2D((64, 64), (1, 1), 32 => 32, pad=SamePad(), activation=Flux.relu, bias=true),
  TimeDistributed(Conv((3, 3), 32 => 1, σ, pad=SamePad())),
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
mnist_x, mnist_y = copy(view(mnist_train, :, :, :, :, 1:N)), copy(view(mnist_train, :, :, :, :, N+1:20));

x_d = (copy(view(mnist_train, :, :, :, t:t+args.batchsize-1, 1:N)) for t in 1:args.batchsize:size(mnist_x, 4))
y_d = (copy(view(mnist_train, :, :, :, t:t+args.batchsize-1, N+1:20)) for t in 1:args.batchsize:size(mnist_y, 4))
data = zip(x_d, y_d)


tx, ty = (copy(view(mnist_test, :,:,:,1:2,1:N)), copy(view(mnist_test, :,:,:,1:2,N+1:20)));

evalcb = () -> @show loss(tx, ty)

using Flux.Optimise
opt = Optimiser(RADAM(args.lr), ExpDecay())

using Flux: throttle, params
p = params(model);
p.params

println("Starting training!")

@time Flux.train!(loss, p, data, opt; cb=throttle(evalcb, 30))

# gs = gradient(ps) do
#   loss(batchmemaybe(d)...)
# end


model
using Plots

function plot_results(y_pred,y)
  g = @animate for i = axes(y_pred, 3)
    p1 = heatmap(y_pred[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    p2 = heatmap(y[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    plot(p1,p2, size=(800, 400))
  end
  gif(g, fps=2)
end

Flux.reset!(model)
ty_pred = cpu(model(gpu(tx)));
nothing
#plot_results(ty_pred[:,:,1,1,:], ty[:,:,1,1,:])
