include("../dataset/fed_grid.jl")

function load_dataset(folder, T₀, T₁;compression=1, N, S, E, W)
  N_val = Float32(uconvert(u"°", N).val)
  S_val = Float32(uconvert(u"°", S).val)
  E_val = Float32(uconvert(u"°", E).val)
  W_val = Float32(uconvert(u"°", W).val)
  dr = Dates.firstdayofmonth(T₀):Month(1):Dates.firstdayofmonth(T₁ + Month(1))
  datasets = map(dr) do d
    basename = year(d)
    m = lpad(month(d), 2, '0')
    filepath = joinpath(folder, savename((; basename, month=m, compression); sort=false) * ".nc")
    ds = read_fed(filepath)
    ds[Lon(W_val..E_val), Lat(S_val..N_val), Ti(T₀..T₁)]
  end
  climarr = cat(datasets...; dims=3)
  lon, lat, tdim = dims(climarr)
  orig_step = step(dims(datasets[begin], 3))
  new_tdim = Ti(first(tdim):orig_step:last(tdim); metadata=tdim.val.metadata)
  ClimArray(climarr.data, (lon, lat, new_tdim))
end

# Funcion para calcular windows
# Funcion para predecir solo en intervalo de T dado y espacio
# funcion para cortar el dataset en muchos intervalos

using Base.Iterators

function get_new_axis(dataset; dimensions, steps)
  axs = axes(dataset)
  for i in 1:3
    d = dims(dataset, i).val
    s = dimensions[i]
    if (length(d) - s) % steps[i] !== 0 
      d_size = length(d)
      ax = axs[i]
      last_ax = last((first(ax):steps[i]:(last(ax)-s)))+s-1
      @warn "DimensionMismatch: Dataset $(d.metadata["standard_name"]) has dimensions $(d_size) but patch size is $(s) and step size is $(steps[i]). Reshaping to $(last_ax)."
      axs = Base.setindex(axs, range(first(ax), last_ax; step=step(ax)), i)
    end
  end
  axs
end

function generate_grids(dataset, axs; dimensions, steps)
  starts = ntuple(i -> range(first(axs[i]), last(axs[i]); step=steps[i]), Val(3))
  (dataset[ntuple(i -> idx[i]:(idx[i]+dimensions[i]-1), Val(3))...] for idx in Iterators.product(starts...))
end

# Correr modelo para todos los climarrs
function predict_climarrs(model, climarrs; device_in, device_out)
  lon, lat, tim = size(climarrs[begin])
  N = length(climarrs)
  out_arr = similar(climarrs)
  ds_in = Array{eltype(climarrs[1]), 5}(undef, lon, lat, 1, N, tim)
  for (i, arr) in enumerate(climarrs)
    ds_in[:, :, 1, i, :] = arr.data
  end
  ds_out = device_out(model(device_in(ds_in)))
  tim_out = size(ds_out, 5)
  for i in axes(ds_out, 4)
    londim, latdim, tdim = dims(climarrs[i])
    dim_tim_out = Ti((last(tdim) + step(tdim)):step(tdim):(last(tdim) + step(tdim) * tim_out); metadata=tdim.val.metadata)
    out_arr[i] = ClimArray(ds_out[:,:,1,i,:], (londim, latdim, dim_tim_out))
  end
  out_arr
end

# Unir arrays en uno solo
# TODO: Refactorizar en una implementación más limpia sin tantas allocations
function join_climarrs(climarrs)
  lon, lat, tim = size(climarrs)
  new_climarrs = collect((cat(climarrs[i, j, :]...; dims=3) for (i,j) in Iterators.product(1:lon, 1:lat)))
  new_climarr = collect((cat(new_climarrs[i, :]...; dims=2) for i in 1:lon))
  cat(new_climarr...; dims=1)
end

using Flux

function generate_mosaic(dataset, model; device_in, device_out, dimensions, steps, batchsize)
  axs = get_new_axis(dataset; dimensions, steps)
  grids = collect(generate_grids(dataset, axs; dimensions, steps))
  out_grids = similar(grids)
  for i in Iterators.partition(1:length(grids), batchsize)
    Flux.reset!(model)
    out_grids[i] .= predict_climarrs(model, grids[i]; device_in, device_out)
  end
  join_climarrs(out_grids)
end

using GMT

function plot_mosaic(climarr, figname=tempname()*".png"; loadfig=true)
  lon = dims(climarr, Lon)
  lat = dims(climarr, Lat)
  region = [minimum(lon), maximum(lon), minimum(lat), maximum(lat)]
  coast = (;DCW=((country=:PE,pen=(0.5,:white)), (continent=:SA, pen=(.5, :white))), region=region)
  topo = makecpt(color=(:yellow, :red), range=(1, 1.1), continuous=false)
  G2 = mat2grid(climarr.data', x=Float64.(lon.val), y=Float64.(lat.val))
  basemap(R=region, proj=:guess, show=false)
  grdimage!(G2; color=topo, colorbar=true, coast=coast, show=false, savefig=figname, fmt=:png)
  loadfig && load(figname)
end

function plot_single_image_factory(climarr)
  lon = dims(climarr, Lon)
  lat = dims(climarr, Lat)
  region = [minimum(lon), maximum(lon), minimum(lat), maximum(lat)]
  coast = (;DCW=((country=:PE,pen=(0.5,:white)), (continent=:SA, pen=(.5, :white))), region=region)
  function ()
    topo = makecpt(color=(:yellow, :red), range=(1, 1.1), continuous=false)
    basemap(R=region, proj=:guess, show=false, compass=true)
    grdimage!("MOVIE_FRAME"; color=topo, colorbar=true, coast=coast, show=false, fmt=:png)
  end
end

function animate_mosaic(climarr, filename)
  lon = Float64.(dims(climarr, Lon))
  lat = Float64.(dims(climarr, Lat))
  tim = dims(climarr, Ti)
  region = [minimum(lon), maximum(lon), minimum(lat), maximum(lat)]
  coast = (;DCW=((country=:PE,pen=(0.5,:white)), (continent=:SA, pen=(.5, :white))), region=region)
  topo = makecpt(color=(:yellow, :red), range=(.99, 1.01), continuous=false)
  figures = map(tim) do t
    G2 = mat2grid(climarr[Ti(At(t))].data', x=lon, y=lat)
    figname=tempname()*".png"
    basemap(R=region, proj=:guess, show=false)
    grdimage!(G2; color=topo, colorbar=true, coast=coast, show=false, savefig=figname, fmt=:png)
    load(figname)
  end
  save(filename, cat(figures...; dims=Val(3)))
end

