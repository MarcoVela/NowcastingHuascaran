using ProgressMeter
using Dates
using Unitful
using DrWatson
using Flux
using GDAL_jll

@quickactivate

include("../../src/structs/FlashRecords.jl")
include("../../src/dataset/fed_grid.jl")
include("../../src/evaluation/load_experiment.jl")

include(srcdir("structs", "FlashRecords.jl"))
include(srcdir("dataset", "fed_grid.jl"))
include(srcdir("evaluation", "load_experiment.jl"))


# Fix for Upsample layers not accepting SubArray, instead of view we use 
# getindex
function Flux.eachlastdim(A::AbstractArray{T,N}) where {T,N}
  inds_before = ntuple(_ -> :, N-1)
  return (getindex(A, inds_before..., i) for i in axes(A, N))
end


function togoesdate(t)
  years = lpad(year(t), 4, '0')
  days = lpad(dayofyear(t), 3, '0')
  hours = lpad(hour(t), 2, '0')
  minutes = lpad(minute(t), 2, '0')
  seconds = lpad(second(t), 2, '0')
  tenth = millisecond(t) ÷ 1000
  return "$(years)$(days)$(hours)$(minutes)$(seconds)$(tenth)"
end

function floor_seconds(t, secs)
  stamp = datetime2unix(t)
  unix2datetime(stamp - (stamp % secs))
end

function get_start_date_goes_file(fname)
  bname = basename(fname)
  s = length("OR_GLM-L2-LCFA_G16_s") + 1
  l = 4 + 3 + 2 + 2 + 2 + 1
  bname[s:s+l-1]
end

function get_files(root_folder, time_from, time_to)
  # example OR_GLM-L2-LCFA_G16_s20232440903200_e20232440903400_c20232440903418.nc
  # use UTC dates
  files = readdir(root_folder, join=true)
  filter!(f -> occursin("OR_GLM-L2-LCFA_G16", f), files)
  start_time = togoesdate(time_from)
  end_time = togoesdate(time_to)
  filter!(f -> start_time <= get_start_date_goes_file(f) <= end_time, files)  
end


function read_flashes(files)
  flashes = FlashRecords[]
  @showprogress "Reading flashes" for fname in files
    NCDataset(fname, "r") do ds
      push!(flashes, FlashRecords(ds))
    end
  end
  flashes
end


function main()
  spatial_resolution = 4u"km"
  temporal_resolution = Minute(15)
  folder = "."

  model_steps = 10

  start = now(UTC) - Year(1)
  finish = start - temporal_resolution * model_steps

  climarr = generate_climarray(read_flashes(get_files(folder, finish, start)), spatial_resolution, temporal_resolution)
  model, _ = load_experiment(datadir("experiments", "final-presentation", "epoch=3.bson"))

  pred = ones(Float32, size(climarr, 1), size(climarr, 2), 10)

  @showprogress for i in 1:64:size(climarr, 1)-64, j in 1:64:size(climarr, 2)-64
    Flux.reset!(model)
    pred[i:i+64-1, j:j+64-1, :] = reshape(
      model(
        reshape(climarr.data[i:i+64-1, j:j+64-1, :], 64, 64, 1, 1, :)
      ),
      64, 64, :,
    )
  end
  time = dims(climarr, Ti)[end]+Minute(15):Minute(15):dims(climarr, Ti)[end]+Minute(15)*10
  pred_uint = floor.(UInt8, pred * (2^8-1))
  for (i, t) in enumerate(time)
    result = ClimArray(pred_uint[:, :, i], (dims(climarr, Lon), dims(climarr, Lat)), "probability of flash")
    tf = tempname() * ".nc"
    ncwrite(tf, result)
    GDAL_jll.gdal_translate_exe() do exe
      run(`$exe -ot Byte $tf ./$(togoesdate(t)).tif`)
    end
  end

end
