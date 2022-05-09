using ProgressMeter
using OrderedCollections

include("../structs/FlashRecords.jl")

function maybereaddir(dirpath; join)
  endswith(dirpath, ".nc") && return [dirpath]
  readdir(dirpath; join)
end


function consolidate_folder_lfca(folder, depth)
  files = readdir(folder; join=true)
  for _ in 1:depth
    files = collect(Iterators.flatten(maybereaddir.(files; join=true)))
  end
  filter!(endswith(".nc"), files)
  flashes = FlashRecords[]
  sizehint!(flashes, length(files))
  @showprogress "Progress for $(basename(folder)): " for file in files
    NCDataset(file, "r") do ds
      push!(flashes, FlashRecords(ds))
    end
  end
  flashes
end


function consolidate_lcfa(basepath, depth::Int, bottom_depth::Int = 2)
  subpaths = readdir(basepath; join=true) # Expect to be years
  for _ in 1:depth
    subpaths = collect(Iterators.flatten(readdir.(subpaths; join=true)))
  end
  response = Dict{String, Vector{FlashRecords}}()
  for folder in subpaths
    @info "Consolidating folder $folder"
    response[basename(folder)] = consolidate_folder_lfca(folder, bottom_depth-depth)
  end
  response
end

function consolidate_lcfa(basepath, lvl::Symbol)
  depth = 0
  if lvl === :year
    depth = 0
  elseif lvl === :day
    depth = 1
  elseif lvl === :hour
    depth = 2
  else
    error("Invalid value for lvl, expected (:year, :day, :hour) but received :$lvl")
  end
  consolidate_lcfa(basepath, depth)
end
