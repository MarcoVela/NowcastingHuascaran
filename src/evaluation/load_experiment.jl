using Random
using BSON
using Dates

function load_experiment(filename)
  architecture = BSON.parse(filename)[:architecture]
  include(srcdir("architecture", "$architecture.jl"))
  params = BSON.load(filename)
  model = pop!(params, :model)
  model, params
end

function load_best_experiment(folder; suffix="bson")
  files = readdir(folder)
  filter!(endswith(suffix), files)
  dics = getindex.(parse_savename.(files), 2)
  _, i = findmin(Base.Fix2(getindex, "loss"), dics)
  filename = joinpath(folder, files[i])
  load_experiment(filename)
end
