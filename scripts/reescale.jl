import Pkg

Pkg.add("NCDatasets")
Pkg.add("DataStructures")
Pkg.add("ProgressMeter")

using Distributed


READER_PROCS = 2

REESCALE_PROCS = 1

STORE_PROCS = 2


if nprocs() < READER_PROCS+REESCALE_PROCS+STORE_PROCS
  addprocs(READER_PROCS+REESCALE_PROCS+STORE_PROCS - nprocs(); exeflags=["--threads=auto"])
end
@everywhere using NCDatasets, DataStructures, ProgressMeter, Dates, Base.Threads


#=
Util functions
=#

@everywhere function readdirall(path)
  names = String[]
  for (root, _, files) in walkdir(path)
    append!(names, joinpath.(root, files))
  end
  names
end


@everywhere function Base.cumsum(A::AbstractArray, dims::NTuple)
  out = copy(A)
  for i in dims
      cumsum!(out, out; dims=i)
  end
  return out
end



@everywhere function integralimage(A::Matrix)
  n,m = size(A)
  out = zeros(eltype(A), n+1, m+1)
  out[2:end, 2:end] = A
  cumsum(out, (1,2))
end


@everywhere function sumgrid(grid, k)
  summed = integralimage(grid)
  w₁, h₁ = size(grid)
  w₂, h₂ = w₁÷k, h₁÷k
  reescaled = zeros(eltype(grid), w₂, h₂)
  for x in 1:w₂, y in 1:h₂
      reescaled[x, y] = summed[begin+x*k,     begin+y*k] + 
                        summed[begin+(x-1)*k, begin+(y-1)*k] - 
                        summed[begin+x*k,     begin+(y-1)*k] - 
                        summed[begin+(x-1)*k, begin+y*k]
  end
  reescaled
end


@everywhere function recreatedataset(ds_struct)
  k = ds_struct.k
	ds = NCDataset(ds_struct.path, "c", attrib = OrderedDict(
			"description"               => "GOES-16 Satellite",
			"source"                    => "NOAA-NASA",
			"author"                    => "Joao Henry Huaman Chinchay (joaohenry23@gmail.com) & Marco Vela (mvelar@uni.pe)",
	))

	# Dimensions

	ds.dim["time"] = Inf # unlimited dimension
	ds.dim["level"] = Inf # unlimited dimension
	ds.dim["latitude"] = length(ds_struct.latitude)
	ds.dim["longitude"] = length(ds_struct.longitude)
	ds.dim["bounds"] = length(ds_struct.time_bounds)
	ds.dim["files"] = length(ds_struct.file_names_used)

	# Declare variables

	nctime = defVar(ds,"time", Float64, ("time",), attrib = OrderedDict(
			"standard_name"             => "time",
			"long_name"                 => "time",
			"units"                     => "hours since 2000-1-1 00:00:00",
			"calendar"                  => "standard",
			"axis"                      => "T",
	))

	nclevel = defVar(ds,"level", Float32, ("level",), attrib = OrderedDict(
			"standard_name"             => "level",
			"long_name"                 => "level",
			"units"                     => "millibars",
			"positive"                  => "down",
			"axis"                      => "Z",
	))

	nclatitude = defVar(ds,"latitude", Float64, ("latitude",), attrib = OrderedDict(
			"standard_name"             => "latitude",
			"long_name"                 => "latitude",
			"units"                     => "degrees_north",
			"axis"                      => "Y",
	))

	nclongitude = defVar(ds,"longitude", Float64, ("longitude",), attrib = OrderedDict(
			"standard_name"             => "longitude",
			"long_name"                 => "longitude",
			"units"                     => "degrees_east",
			"axis"                      => "X",
	))

	ncdensity = defVar(ds,"density", UInt16, ("longitude", "latitude"), attrib = OrderedDict(
			"standard_name"             => "flash density",
			"long_name"                 => "flash density of GLM",
			"units"                     => "flash density in $(2*k)x$(2*k) Km",
			"axis"                      => "YX",
	), shuffle=true, deflatelevel=9)

	nctime_bounds = defVar(ds,"time_bounds", Float64, ("bounds",), attrib = OrderedDict(
			"standard_name"             => "time bounds",
			"long_name"                 => "time bounds of density",
			"units"                     => "hours since 2000-1-1 00:00:00",
			"calendar"                  => "standard",
			"axis"                      => "T",
	))

	ncfile_names_used = defVar(ds,"file_names_used", String, ("files",), attrib = OrderedDict(
			"standard_name"             => "file names used",
			"long_name"                 => "file names used in density",
	))


	# Define variables

	nctime[:] = [ds_struct.time]
	nclevel[:] = [ds_struct.level]
	nclatitude[:] = ds_struct.latitude
	nclongitude[:] = ds_struct.longitude
	ncdensity[:] = ds_struct.density
	nctime_bounds[:] = ds_struct.time_bounds
	ncfile_names_used[:] = ds_struct.file_names_used
  close(ds)
end

#=
Strucs
=#

@everywhere struct OwnDataset
  density::Array{UInt16, 2}
  latitude::Vector{Float64}
  longitude::Vector{Float64}
  time::DateTime
  level::Float32
  time_bounds::Vector{DateTime}
  file_names_used::Vector{String}
  path::String
  k::Int64
end

@everywhere function OwnDataset(ds::NCDataset, path, k)
  OwnDataset(
    UInt16.(Array(ds["density"])),
    ds["latitude"][:],
    ds["longitude"][:],
    ds["time"][1],
    ds["level"][1],
    ds["time_bounds"][:],
    ds["file_names_used"][:],
    path,
    k
  )
end

# Reads paths and k and outs OwnDataset
@everywhere function reader_proc(k, in_channel, out_channel)
  while true
    (path, newpath) = take!(in_channel)
		olddataset = NCDataset(path)
    put!(out_channel, OwnDataset(olddataset, newpath, k))
    close(olddataset)
  end
end

@everywhere function reescale(ds::OwnDataset)
  k = ds.k
  OwnDataset(
    sumgrid(ds.density, k),
    ds.latitude[begin:k:end],
    ds.longitude[begin:k:end],
    ds.time,
    ds.level,
    ds.time_bounds,
    ds.file_names_used,
    ds.path,
    ds.k
  )
end


# In channel are ds and outs escaled attributes
@everywhere function reescaler_proc(in_channel, out_channel)
  @threads for _ in 1:nthreads()
    x = 0
    while true
      ds = take!(in_channel)
      put!(out_channel, reescale(ds))
      x += 1
      x % 1000 == 0 && GC.gc()
    end
  end
end

@everywhere function store_proc(in_channel, out_channel)
  while true
    ds = take!(in_channel)
    recreatedataset(ds)
    put!(out_channel, true)
  end
end



function main()

  K = 4

  println("Ruta actual: $(pwd())")

  print("Ruta base: ")
  base_path = readline()
  print("Carpeta: ")
  base_folder = readline()
  print("Carpeta objetivo: ")
  target_folder = readline()
  rootfolder = abspath(joinpath(base_path, base_folder))
	files = filter(endswith(".nc"), readdirall(rootfolder))
	newfolder = abspath(joinpath(base_path, target_folder))
	mkpath(newfolder)
	newfiles = replace.(files, rootfolder => newfolder)
	println("Working with $(nprocs())")
	mkpath.(unique(dirname.(newfiles)))

  paths = collect(zip(files, newfiles))
  cproc = 1
  progress = Progress(length(paths) ;desc="Recreating datasets...", showspeed=true)

  paths_chan = RemoteChannel(() -> Channel{Tuple{String,String}}(length(paths)))
  reescaled_chan = RemoteChannel(() -> Channel{OwnDataset}(1_000))
  own_dataset_chan = RemoteChannel(() -> Channel{OwnDataset}(1_000))
  finish_chan = RemoteChannel(() -> Channel{Bool}(length(paths)))

  for x in paths
    put!(paths_chan, x)
  end

  for i = cproc+1:READER_PROCS+cproc
    @async remote_do(reader_proc, i, K, paths_chan, own_dataset_chan)
    cproc += 1
  end

  for i = cproc+1:REESCALE_PROCS+cproc
    @async remote_do(reescaler_proc, i, own_dataset_chan, reescaled_chan)
    cproc += 1
  end

  for i = cproc+1:STORE_PROCS+cproc
    @async remote_do(store_proc, i, reescaled_chan, finish_chan)
    cproc += 1
  end


  queued = length(paths)
  while queued > 0
    take!(finish_chan)
    queued % 1000 == 0 && GC.gc()
    queued -= 1
    next!(progress)
  end

end

main()