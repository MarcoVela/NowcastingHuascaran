module NowcastingHuascaran

include("dataset/cluster_dbscan.jl")
include("dataset/fed_grid.jl")

export generate_dataset
export read_fed

end