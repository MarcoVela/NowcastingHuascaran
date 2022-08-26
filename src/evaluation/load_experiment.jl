using Random
using BSON
using Dates

function load_experiment(filename)
  architecture = BSON.parse(filename)[:architecture][:type]
  include(srcdir("architecture", "$architecture.jl"))
  params = BSON.load(filename)
  model = pop!(params, :model)
  model, params
end

function parse_experiment(filename)
  initial = BSON.parse(filename)
  delete!(initial, :model)
  nothing, BSON.raise_recursive(initial, Main)
end

function load_best_experiment(folder, metric; suffix="bson", sort)
  files = readdir(folder)
  filter!(endswith(suffix), files)
  dics = getindex.(parse_savename.(files), 2)
  findbest = sort == "min" ? findmin : findmax
  _, i = findbest(Base.Fix2(getindex, string(metric)), dics)
  filename = joinpath(folder, files[i])
  load_experiment(filename)
end

function parse_best_experiment(folder, metric; suffix="bson", sort)
  files = readdir(folder)
  filter!(endswith(suffix), files)
  dics = getindex.(parse_savename.(files), 2)
  findbest = sort == "min" ? findmin : findmax
  try
    _, i = findbest(Base.Fix2(getindex, string(metric)), dics)
    filename = joinpath(folder, files[i])
    parse_experiment(filename)
  catch
    return nothing, nothing
  end
end
