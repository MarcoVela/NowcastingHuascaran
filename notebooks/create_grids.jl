### A Pluto.jl notebook ###
# v0.17.1

using Markdown
using InteractiveUtils

# ╔═╡ 75a6694e-41e1-11ec-0901-ff012a65e379
begin
	import Pkg
	Pkg.activate("../")
end

# ╔═╡ 771feccd-03d6-44f9-b95d-10f5c40dd0ae
using DrWatson

# ╔═╡ a579af42-c4f1-4f87-9e48-bed80169ca70
@quickactivate

# ╔═╡ 88213d99-cb46-430c-bf05-7f3766ccbe5e
using Geodesy

# ╔═╡ 25bf0188-03e9-4a9a-b60a-c24eae3cdf6a
using Plots

# ╔═╡ 238cfd38-8962-4a59-9855-cc3fbf4c0806
begin
	include(srcdir("grids.jl"))
	include(srcdir("netcdf.jl"))
	include(srcdir("goes.jl"))
end

# ╔═╡ 494956b8-b611-430d-b0b3-bf8fed69ad05
function ingredients(path::String)
	# this is from the Julia source code (evalfile in base/loading.jl)
	# but with the modification that it returns the module instead of the last object
	name = Symbol(basename(path))
	m = Module(name)
	Core.eval(m,
        Expr(:toplevel,
             :(eval(x) = $(Expr(:core, :eval))($name, x)),
             :(include(x) = $(Expr(:top, :include))($name, x)),
             :(include(mapexpr::Function, x) = $(Expr(:top, :include))(mapexpr, $name, x)),
             :(include($path))))
	m
end

# ╔═╡ b07ea5d4-d06d-4b9a-a2ac-54b615ddd4e3
fs = collect_tostruct(datadir("exp_raw", "GLMtest"))

# ╔═╡ d2695e28-e697-4cbe-b587-e94cc5fd2b76
corner = UTM(LLA(0.0, -84.0), 17, false, wgs84)

# ╔═╡ b72bc0f1-31c3-4b95-b660-aa93213c5d1c
flash_coordslla = vcat([x.flash_coord for x in fs]...)

# ╔═╡ 069a141b-f5cc-47b0-b0fb-33bb1ec8665d


# ╔═╡ c546949c-772b-4583-a766-5086d238a646
flash_coords = [
	UTM(y, 17, false, wgs84) 
	for y in vcat([x.flash_coord for x in fs]...)
	if (-20 < y.lat < 0) && (-84 < y.lon < -66)
]

# ╔═╡ c5814647-c09e-44d5-b284-f05095111d2e
flash_coords

# ╔═╡ 548ca73c-d510-406e-a5e8-ccdc024f3b4e
grid = GeoGrid(corner, 8, zeros(Int64, 200, 200))

# ╔═╡ cced29f0-9f2a-427c-99f3-fd6dcdbcdbda
accumulate!(grid, flash_coords)

# ╔═╡ a1fdb536-14ad-4a6c-be8b-353b36b5341e


# ╔═╡ 83686fad-1834-4256-935e-8ac7e894e7bf
heatmap(-grid.grid, color = :greys)

# ╔═╡ 92e8799a-661a-49f7-9bcd-eb1c21b9694c


# ╔═╡ 35024385-7e60-4c20-b445-80bccba077ca


# ╔═╡ Cell order:
# ╠═75a6694e-41e1-11ec-0901-ff012a65e379
# ╠═771feccd-03d6-44f9-b95d-10f5c40dd0ae
# ╠═a579af42-c4f1-4f87-9e48-bed80169ca70
# ╠═88213d99-cb46-430c-bf05-7f3766ccbe5e
# ╠═494956b8-b611-430d-b0b3-bf8fed69ad05
# ╠═238cfd38-8962-4a59-9855-cc3fbf4c0806
# ╠═b07ea5d4-d06d-4b9a-a2ac-54b615ddd4e3
# ╠═d2695e28-e697-4cbe-b587-e94cc5fd2b76
# ╠═b72bc0f1-31c3-4b95-b660-aa93213c5d1c
# ╠═069a141b-f5cc-47b0-b0fb-33bb1ec8665d
# ╠═c5814647-c09e-44d5-b284-f05095111d2e
# ╠═c546949c-772b-4583-a766-5086d238a646
# ╠═548ca73c-d510-406e-a5e8-ccdc024f3b4e
# ╠═cced29f0-9f2a-427c-99f3-fd6dcdbcdbda
# ╠═a1fdb536-14ad-4a6c-be8b-353b36b5341e
# ╠═25bf0188-03e9-4a9a-b60a-c24eae3cdf6a
# ╠═83686fad-1834-4256-935e-8ac7e894e7bf
# ╠═92e8799a-661a-49f7-9bcd-eb1c21b9694c
# ╠═35024385-7e60-4c20-b445-80bccba077ca
