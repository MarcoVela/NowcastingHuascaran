using NPZ
using DrWatson
using Plots

@quickactivate

data = NPZ.npzread(datadir("mnist", "raw", "mnist_test_seq.npy"))
datarotated = permutedims(data, (3, 4, 1, 2))

for i = 1:size(datarotated, 4), j = 1:size(datarotated, 3)
    datarotated[:, :, j, i] = reverse(datarotated[:, :, j, i]'; dims=2)'
end

save(datadir("mnist", "raw", "mnist_test_seq.jld"), "data", datarotated)


@gif for i = 1:20
    heatmap(datarotated[:, :, i, 1], c=:grays)
end

