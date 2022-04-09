using DrWatson
using JLD
using Flux

dataset = load(datadir("mnist", "raw", "mnist_test_seq.jld"))["data"][:, :, :, begin:1000]

include(srcdir("Conv2dLSTM.jl"))

dataset = reshape(dataset, (64, 64, 1, 20, 1000)) / Float32(255)

model = Chain(
    Conv2dLSTM((64, 64), (5, 5), 1 => 64),
    x -> relu.(x),
    BatchNorm(64),
    Conv2dLSTM((64, 64), (3, 3), 64 => 64),
    x -> relu.(x),
    BatchNorm(64),
    Conv2dLSTM((64, 64), (1, 1), 64 => 64),
    x -> relu.(x),
    Conv((3,3), 64 => 1, pad=SamePad())
)
