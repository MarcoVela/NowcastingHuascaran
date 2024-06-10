module NowcastingHuascaran
include("dataset/cluster_dbscan.jl")
include("dataset/fed_grid.jl")
include("structs/FlashRecords.jl")
include("architecture/Default.jl")


export generate_dataset
export read_fed
export FlashRecords
export build_model

end