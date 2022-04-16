using DrWatson
@quickactivate
using NPZ
using Parameters: @with_kw
@with_kw mutable struct Args
  lr::Float64 = 1e-2  # Learning rate
  batchsize::Int = 32 # Batch size
  throttle::Int = 30  # Throttle timeout
  epochs::Int = 2     # Number of Epochs
end
args = Args()
mnist_whole = @time NPZ.npzread(datadir("exp_raw", "moving-mnist", "mnist_test_seq.npy")) ./ Float32(255);
mnist = permutedims(mnist_whole, (3,4,1,2));
mnist = reshape(mnist, (size(mnist)[1:2]..., 1, size(mnist)[3:end]...))
@show size(mnist)
train_test_split = .8
mnist_train = view(mnist, :, :, :, :, 1:Int(size(mnist, 5) * train_test_split))
mnist_test = view(mnist, :, :, :, :, Int(size(mnist, 5) * train_test_split)+1:size(mnist, 5))
@show size(mnist_train)
@show size(mnist_test);
function broadcasted_σ(x)
  Flux.σ.(x)
end

include(srcdir("layers", "ConvLSTM2D.jl"))
using Flux.Losses: binarycrossentropy

device = cpu


model = Chain(
    ConvLSTM2D((64, 64), (5, 5), 1 => 32, return_sequences=true, pad=SamePad()),
    ConvLSTM2D((64, 64), (3, 3), 32 => 32, return_sequences=false, pad=SamePad()),
    ConvLSTM2D((64, 64), (3, 3), 32 => 32, return_sequences=true, repeat_input=10, pad=SamePad()),
    ConvLSTM2D((64, 64), (1, 1), 32 => 32, return_sequences=true, pad=SamePad()),
    Conv((3, 3), 32 => 1, σ, pad=SamePad())
) |> device

using CUDA
using Statistics
function loss(X, y)
    Flux.reset!(model)
    X_dev = device(X)
    y_pred = model(X_dev)
    errors = binarycrossentropy(Array(y_pred), y; agg=identity)
    sum(mean(errors; dims=(1,2,3)))
end

function batched_loss(X, y)
  X_dev = device(X)
  X_gen = (view(X_dev, :, :, :, :, t) for t in axes(X_dev, 5))
  y_gen = (view(y, :, :, :, :, t) for t in axes(y, 5))
  mean(loss(X_n, y_n) for (X_n, y_n) in zip(X_gen, y_gen))
end

N = 10
mnist_x, mnist_y = copy(view(mnist_train, :, :, :, 1:N, 1:1024)), copy(view(mnist_train, :, :, :, N+1:20, 1:1024));
using Flux.Data: DataLoader
data = DataLoader((mnist_x, mnist_y); batchsize=args.batchsize, partial=false);

data = zip(
    (copy(view(mnist_train, :, :, :, 1:N, t)) for t in axes(mnist_train, 5)),
    (copy(view(mnist_train, :, :, :, N+1:20, t)) for t in axes(mnist_train, 5))
)

tx, ty = (view(mnist_test, :,:,:,1:N,1:64), view(mnist_test, :,:,:,N+1:20,1:64))
evalcb = () -> @show batched_loss(tx, ty)

using Flux.Optimise: ADAM
opt = ADAM(args.lr)

using Flux: throttle, params
p = params(model);
p.params

d = first(data)
gs = Flux.gradient(p) do
  loss(d...)
end
Flux.update!(opt, p, gs)

#Flux.train!(loss, p, data, opt)

model