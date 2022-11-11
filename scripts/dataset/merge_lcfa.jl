using DrWatson
@quickactivate

using JLD2

include(srcdir("dataset", "l2_lcfa_merge.jl"))

folder = ARGS[1]

!isdir(folder) && error("first argument must be a directory, received $folder instead")

depth = length(ARGS) > 1 ? parse(Int, ARGS[2]) : 0

save_path = datadir("exp_pro", "GLM-L2-LCFA", "$(basename(folder)).jld2")

wsave(save_path, consolidate_lcfa(folder, depth))


