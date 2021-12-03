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

# ╔═╡ 4490a68c-2654-4afb-b376-5d21ca75f363
using AWSS3, Dates

# ╔═╡ 2622beb9-ddfb-4c5c-ba4a-43feba01ea2d
using Serialization

# ╔═╡ 494956b8-b611-430d-b0b3-bf8fed69ad05
using ImageShow

# ╔═╡ 238cfd38-8962-4a59-9855-cc3fbf4c0806
begin
	include(srcdir("grids.jl"))
	include(srcdir("netcdf.jl"))
	include(srcdir("goes.jl"))
end

# ╔═╡ bee9bd99-e43c-4f0d-9250-62a9970ccbd4
begin
	product_str = "|" * sprint(show_products; context=:compact => true)
	product_str = replace(product_str, '\t' => "|")
	product_str = replace(product_str, '\n' => "|\n")
	Markdown.parse(product_str)
end;

# ╔═╡ c5acad28-a3b3-4189-8897-1a8bb128b62e
begin
	ayer = startofday(now() - Day(1))
	ayer_str = Dates.format(ayer, "yyyymmdd")
	save_dir = datadir("exp_raw", "GLM", ayer_str)
end

# ╔═╡ 81d6712c-c422-4290-91a7-5cff52da664a
processed_path = datadir("exp_pro", "GLM", ayer_str * ".jls")

# ╔═╡ d2944cb1-fcf2-407b-bd06-613f7bd98cb8


# ╔═╡ b07ea5d4-d06d-4b9a-a2ac-54b615ddd4e3


# ╔═╡ db0bec96-a690-4f2f-bf5c-f4fe7a0e08c1
if isfile(processed_path)
	fs = deserialize(processed_path)
else
	download_satellite_data("goes16", "GLM-L2-LCFA", ayer; 
		path=save_dir, show_progress=false)
	fs = collect_tostruct(save_dir)
	serialize(processed_path, fs)
end

# ╔═╡ b7fc72b1-25b8-40a0-a897-2990e1f03f6a
Base.summarysize(fs) / 1024 ^ 2

# ╔═╡ d2695e28-e697-4cbe-b587-e94cc5fd2b76
corner = UTM(LLA(0.0, -84.0), 17, false, wgs84)

# ╔═╡ b72bc0f1-31c3-4b95-b660-aa93213c5d1c
flash_coordslla = vcat([x.flash_coord for x in fs]...)

# ╔═╡ 069a141b-f5cc-47b0-b0fb-33bb1ec8665d


# ╔═╡ c5814647-c09e-44d5-b284-f05095111d2e


# ╔═╡ c546949c-772b-4583-a766-5086d238a646
flash_coords = [
	UTM(y, 17, false, wgs84) 
	for y in vcat([x.flash_coord for x in fs]...)
	if (-20 < y.lat < 0) && (-84 < y.lon < -66)
]

# ╔═╡ c7929b55-8071-4605-b7f5-51f9f10749a7


# ╔═╡ 548ca73c-d510-406e-a5e8-ccdc024f3b4e
begin
	grid = GeoGrid(corner, 2, zeros(Int64, 650, 650))
	accumulate!(grid, flash_coords)
	plot(load(plotsdir("utm17.png")))
	heatmap!(grid.grid, color = cgrad([RGBA(1,1,1,0), RGBA(0,0,0,1)]) )
end

# ╔═╡ 89b0daa7-5d38-42ca-8275-6940df2f5e7c
begin
	bottom_corner = UTM(
		corner.x + grid.spacialresolution * size(grid.grid, 1) * 1e3, 
		corner.y - grid.spacialresolution * size(grid.grid, 1) * 1e3,
		corner.z)
	trans = LLAfromUTM(17, false, wgs84)
	trans(bottom_corner)
end

# ╔═╡ cced29f0-9f2a-427c-99f3-fd6dcdbcdbda
1600 / 650

# ╔═╡ 83686fad-1834-4256-935e-8ac7e894e7bf


# ╔═╡ 35024385-7e60-4c20-b445-80bccba077ca


# ╔═╡ Cell order:
# ╠═75a6694e-41e1-11ec-0901-ff012a65e379
# ╠═771feccd-03d6-44f9-b95d-10f5c40dd0ae
# ╠═a579af42-c4f1-4f87-9e48-bed80169ca70
# ╠═88213d99-cb46-430c-bf05-7f3766ccbe5e
# ╠═25bf0188-03e9-4a9a-b60a-c24eae3cdf6a
# ╠═4490a68c-2654-4afb-b376-5d21ca75f363
# ╠═2622beb9-ddfb-4c5c-ba4a-43feba01ea2d
# ╠═494956b8-b611-430d-b0b3-bf8fed69ad05
# ╠═238cfd38-8962-4a59-9855-cc3fbf4c0806
# ╠═bee9bd99-e43c-4f0d-9250-62a9970ccbd4
# ╠═c5acad28-a3b3-4189-8897-1a8bb128b62e
# ╠═81d6712c-c422-4290-91a7-5cff52da664a
# ╠═d2944cb1-fcf2-407b-bd06-613f7bd98cb8
# ╠═b07ea5d4-d06d-4b9a-a2ac-54b615ddd4e3
# ╠═db0bec96-a690-4f2f-bf5c-f4fe7a0e08c1
# ╠═b7fc72b1-25b8-40a0-a897-2990e1f03f6a
# ╠═d2695e28-e697-4cbe-b587-e94cc5fd2b76
# ╠═b72bc0f1-31c3-4b95-b660-aa93213c5d1c
# ╠═069a141b-f5cc-47b0-b0fb-33bb1ec8665d
# ╠═c5814647-c09e-44d5-b284-f05095111d2e
# ╠═89b0daa7-5d38-42ca-8275-6940df2f5e7c
# ╠═c546949c-772b-4583-a766-5086d238a646
# ╠═c7929b55-8071-4605-b7f5-51f9f10749a7
# ╠═548ca73c-d510-406e-a5e8-ccdc024f3b4e
# ╠═cced29f0-9f2a-427c-99f3-fd6dcdbcdbda
# ╠═83686fad-1834-4256-935e-8ac7e894e7bf
# ╠═35024385-7e60-4c20-b445-80bccba077ca
