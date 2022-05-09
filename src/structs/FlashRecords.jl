using NCDatasets
using Dates

struct FlashRecords
  latitude::Vector{Float32}
  longitude::Vector{Float32}
  quality::Vector{Bool}
  energy::Vector{Union{Missing, Float32}}
  area::Vector{Union{Missing, Float32}}
  time_start::DateTime
  time_end::DateTime
  dataset_name::String
end

function FlashRecords(ds::NCDataset)
  FlashRecords(
    ds["flash_lat"][:],
    ds["flash_lon"][:],
    ds["flash_quality_flag"][:] .== zero(Int16),
    ds["flash_energy"][:],
    ds["flash_area"][:],
    DateTime(ds.attrib["time_coverage_start"], dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
    DateTime(ds.attrib["time_coverage_end"], dateformat"yyyy-mm-ddTHH:MM:SS.sZ"),
    ds.attrib["dataset_name"]
  )
end