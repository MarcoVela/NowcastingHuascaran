import Pkg

Pkg.add("NCDatasets")
Pkg.add("DataStructures")
Pkg.add("ProgressMeter")

using Distributed



PROCS_LIM = 2

if nprocs() < PROCS_LIM
	addprocs(PROCS_LIM - nprocs())
end

@everywhere using NCDatasets, DataStructures, ProgressMeter

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



@everywhere function recreatedataset(dataset, newpath, k)
	ds = NCDataset(newpath,"c", attrib = OrderedDict(
			"description"               => "GOES-16 Satellite",
			"source"                    => "NOAA-NASA",
			"author"                    => "Joao Henry Huaman Chinchay (joaohenry23@gmail.com) & Marco Vela (mvelar@uni.pe)",
	))

	# Dimensions

	ds.dim["time"] = Inf # unlimited dimension
	ds.dim["level"] = Inf # unlimited dimension
	ds.dim["latitude"] = dataset.dim["latitude"] ÷ k
	ds.dim["longitude"] = dataset.dim["longitude"] ÷ k
	ds.dim["bounds"] = 2
	ds.dim["files"] = 15

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

	nctime[:] = dataset["time"][:]
	nclevel[:] = dataset["level"][:]
	nclatitude[:] = dataset["latitude"][begin:k:end]
	nclongitude[:] = dataset["longitude"][begin:k:end]
	ncdensity[:] = sumgrid(UInt16.(Array(dataset["density"])), k)
	nctime_bounds[:] = dataset["time_bounds"][:]
	ncfile_names_used[:] = dataset["file_names_used"][:]
	close(ds)
end




function recreate(path, foldername, newfoldername, k)
	rootfolder = abspath(joinpath(path, foldername))
	println("Primer paso: Listar archivos a reescalar.")
	files = [x for x in readdirall(rootfolder) if endswith(x, ".nc")]
	println("$(length(files)) archivos encontrados.")
	println("Segundo paso: Reescalar archivos.")
	newfolder = abspath(joinpath(path, newfoldername))
	println("Carpeta de destino=$newfolder")
	mkpath(newfolder)
	newfiles = replace.(files, rootfolder => newfolder)
	mkpath.(unique(dirname.(newfiles)))

	@showprogress 1 "Recreando archivos..." pmap(zip(files, newfiles)) do (file, newfile)
		olddataset = NCDataset(file)
		recreatedataset(olddataset, newfile, k)
		close(olddataset)
	end
end


println("Ruta actual: $(pwd())")

print("Ruta base: ")
base_path = readline()
print("Carpeta: ")
base_folder = readline()
print("Carpeta objetivo: ")
target_folder = readline()


recreate(base_path, base_folder, target_folder, 4)