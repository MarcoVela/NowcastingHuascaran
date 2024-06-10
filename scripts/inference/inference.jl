using DrWatson
@quickactivate

using ProgressMeter
using Dates
using Unitful
using Flux
using GDAL_jll


include("../../src/structs/FlashRecords.jl")
include("../../src/dataset/fed_grid.jl")
include("../../src/evaluation/load_experiment.jl")

include(srcdir("structs", "FlashRecords.jl"))
include(srcdir("dataset", "fed_grid.jl"))
include(srcdir("evaluation", "load_experiment.jl"))

model, _ = load_experiment(datadir("experiments", "final-presentation", "epoch=2.bson"))


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
  @show start_time end_time
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


function pad_matrix(mat, pad_size)
  m,n,t = size(mat)
  padded_input = zeros(Float32, m + 2 * pad_size, n + 2 * pad_size, t)
  padded_input[pad_size+1:pad_size+m, pad_size+1:pad_size+n, :] .= mat
  return padded_input
end

function predict_matrix(mat, stride)
  w = 64
  padded_input = pad_matrix(mat, 32)
  m,n,t = size(padded_input)
  out_matrix = zeros(Float32, m, n, 10)
  counts = zeros(Float32, m, n)
  for i in 1:stride:(m-w+1), j in 1:stride:(n-w+1)
    pred = predict(view(padded_input, i:i+w-1, j:j+w-1, :))
    out_matrix[i:i+w-1, j:j+w-1, :] += pred
    counts[i:i+w-1, j:j+w-1, :] .+= 1
  end
  y = out_matrix ./ counts
  y[32+1:end-32, 32+1:end-32, :]
end




function main()

  spatial_resolution = 4u"km"
  temporal_resolution = Minute(15)
  folder = datadir("exp_raw", "22")

  model_steps = 10
  W = 64
  stride = 16

  start = DateTime(2023, 6, 1)# now(UTC) - Year(1)
  finish = DateTime(2023, 5, 31)# start - temporal_resolution * model_steps

  flashes = read_flashes(get_files(folder, finish, start))
  @show length(flashes)
  climarr = generate_climarray(flashes, spatial_resolution, temporal_resolution)

  for t in dims(climarr, Ti)
    result = ClimArray(floor.(UInt8, min.(1, climarr[Ti=At(t)].data) * (2^8-1)), (dims(climarr, Lon), dims(climarr, Lat)), "ocurrence of flash")
    tf = tempname() * ".nc"
    ncwrite(tf, result)
    GDAL_jll.gdal_translate_exe() do exe
      run(`$exe -ot Byte $tf ./$(togoesdate(t)).tif`)
    end
  end

  input_array = Flux.pad_zeros(climarr.data, (W ÷ 2, W ÷ 2, 0))
  m, n, _ = size(input_array)

  pred = zeros(Float32, m, n, model_steps)
  counts = zeros(Float32, m, n)

  @showprogress for (i, j) in Base.Iterators.product(1:stride:m-W, 1:stride:n-W)
    Flux.reset!(model)
    pred[i:i+W-1, j:j+W-1, :] += reshape(
      model(
        reshape(input_array[i:i+W-1, j:j+W-1, :], W, W, 1, 1, :)
      ),
      W, W, :,
    )
    counts[i:i+W-1, j:j+W-1, :] .+= 1
  end
  pred = (pred ./ counts)[W÷2+1:end-W÷2, W÷2+1:end-W÷2, :]
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

if abspath(PROGRAM_FILE) == @__FILE__
  main()
end
