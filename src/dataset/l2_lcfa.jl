using Dates

struct FlashLCFA
  lat::Vector{Float32}
  lon::Vector{Float32}
  area::Vector{Union{Missing, Float32}}
  energy::Vector{Union{Missing, Float32}}
  start_time::DateTime
  end_time::DateTime
end


function merge_datasets()


end