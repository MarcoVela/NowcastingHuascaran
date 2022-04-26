using DrWatson
@quickactivate
using NPZ
using Parameters: @with_kw
@with_kw mutable struct Args
  lr::Float64 = 2e-4  # Learning rate
  batchsize::Int = 4  # Batch size
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
train_test_split = .9
TOTAL_SAMPLES = size(mnist_whole, 4)
mnist_train = view(mnist_whole, :, :, :, 1:Int(TOTAL_SAMPLES * train_test_split), :);
mnist_test = view(mnist_whole, :, :, :, Int(TOTAL_SAMPLES * train_test_split)+1:TOTAL_SAMPLES, :);

@show size(mnist_train)
@show size(mnist_test);
function broadcasted_σ(x)
  Flux.σ.(x)
end

include(srcdir("layers", "SimpleConvLSTM2D.jl"))
using Flux.Losses: binarycrossentropy, logitbinarycrossentropy
using CUDA

const device = gpu

# function Flux.ChainRulesCore.rrule(::typeof(reverse), x::AbstractArray{T, N}; dims) where {T, N}
#   reverse!(x; dims), dy -> (Zygote.NoTangent(), reverse!(dy; dims), Zygote.NoTangent())
# end



N = 12

model = Chain(
  TimeDistributed(
    Chain(
      Conv((3,3),  1=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
      Conv((3,3), 64=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
      Conv((3,3), 64=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad(), stride=2),
    
      Conv((3,3),  64=>128, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
      Conv((3,3),  128=>128, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad(), stride=2),
    ),
  ),
  KeepLast(
    SimpleConvLSTM2D((16, 16), (3, 3),  128 => 128, pad=SamePad()),
  ),
  Dropout(.25),
  RepeatInput(
    20-N,
    SimpleConvLSTM2D((16, 16), (3, 3), 128 => 128, pad=SamePad()),
  ),
  TimeDistributed(
    Chain(
      ConvTranspose((7,7), 128=>128, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
      ConvTranspose((7,7), 128=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad(), stride=2),
    
      ConvTranspose((7,7), 64=>64, Base.Fix2(leakyrelu, 0.2f0), pad=SamePad()),
      ConvTranspose((7,7), 64=>1, σ, pad=SamePad(), stride=2),
    )
  ),
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

x_d = (copy(view(mnist_train, :, :, :, t:t+args.batchsize-1, 1:N)) for t in 1:args.batchsize:size(mnist_x, 4)-args.batchsize+1)
y_d = (copy(view(mnist_train, :, :, :, t:t+args.batchsize-1, N+1:20)) for t in 1:args.batchsize:size(mnist_y, 4)-args.batchsize+1)

data = zip(x_d, y_d)


tx, ty = (copy(view(mnist_test, :,:,:,1:2,1:N)), copy(view(mnist_test, :,:,:,1:2,N+1:20)));
tx2, ty2 = (copy(view(mnist_test, :,:,:,1:10,1:N)), copy(view(mnist_test, :,:,:,1:10,N+1:20)));


evalcb = () -> (@show loss(tx, ty); nothing)

using Flux.Optimise
opt = ADAM(args.lr)

using Flux: throttle, params
p = params(model);
p.params

println("Starting training!")

@time Flux.train!(loss, params(model), data, opt; cb=throttle(evalcb, 30))

# gs = gradient(ps) do
#   loss(batchmemaybe(d)...)
# end
println("Sleeping for 10 secs")
GC.gc(true)
sleep(10)
CUDA.reclaim()
println("Wake up! Time to plot")


# model
using Plots

function plot_results(x, y_pred, y)
  ps = []
  for i = axes(x, 3)
    p = heatmap(x[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    push!(ps, (p, p))
  end
  for i = axes(y_pred, 3)
    p1 = heatmap(y_pred[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    p2 = heatmap(y[:,:,i], clims=(0,1), c=[:black, :white], colorbar=nothing)
    push!(ps, (p1, p2))
  end
  g = @animate for (p1, p2) in ps
    plot(p1, p2, size=(800, 400))
  end
  gif(g, fps=2)
end

Flux.reset!(model)
ty_pred = cpu(model(device(tx)));
i = 2;
nothing
#plot_results(ty_pred[:,:,1,1,:], ty[:,:,1,1,:])

using Statistics

function plot_crossentropy_framewise(y_pred, y)
  cross = binarycrossentropy(y_pred, y; agg=identity)
  framewise = mean(cross; dims=(1,2))[:]
  plot(framewise, ylims=(0, maximum(framewise) * 1.5), legend=false)
end

function csi_framewise(y_pred, y; threshold = 0.5)
  cortados = y_pred .> threshold
  cort_y = y .> .9
  a = sum(cortados .== cort_y .== 1; dims=(1,2))[:]
  b = sum((cortados .== 1) .& (cort_y .== 0); dims=(1,2))[:]
  c = sum((cortados .== 0) .& (cort_y .== 1); dims=(1,2))[:]
  # d = sum(cortados .== cort_y .== 0; dims=(1,2))[:]
  @. a / (a + b + c)
end

function plot_csi_framewise(y_pred, y; thresholds=0.1:.15:.9)
  t = size(y, 3)
  ps = map(x -> begin
    d = csi_framewise(y_pred, y; threshold=x)
    label = "umbral: $x - maximo: $(round(maximum(d), digits=2))"
    plot(d, label=label, xticks=1:t, ylims=(0,1))
  end, thresholds)
  plot(ps...; layout=(length(thresholds), 1), size=(800, 900))
end

function accuracy_framewise(y_pred, y; threshold = 0.5)
  cortados = y_pred .> threshold
  cort_y = y .> .9
  a = sum(cortados .== cort_y .== 1; dims=(1,2))[:]
  b = sum((cortados .== 1) .& (cort_y .== 0); dims=(1,2))[:]
  c = sum((cortados .== 0) .& (cort_y .== 1); dims=(1,2))[:]
  d = sum(cortados .== cort_y .== 0; dims=(1,2))[:]
  @. (a + d) / (a + b + c + d)
end

function plot_acc_framewise(y_pred, y; thresholds=0.1:.15:.9)
  t = size(y, 3)
  ps = map(x -> begin
    d = accuracy_framewise(y_pred, y; threshold=x)
    label = "umbral: $x - maximo: $(round(maximum(d), digits=2))"
    plot(d, label=label, xticks=1:t, ylims=(0,1))
  end, thresholds)
  plot(ps...; layout=(length(thresholds), 1), size=(900, 900))
end

