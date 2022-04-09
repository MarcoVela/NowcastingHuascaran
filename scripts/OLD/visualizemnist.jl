using DrWatson
using Plots
@quickactivate

data = load(datadir("mnist", "raw", "mnist_test_seq.jld"))["data"]

@gif for n = 1:10, i = 1:20
    heatmap(data[:, :, i, n], c=:grays)
end