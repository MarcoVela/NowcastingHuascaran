using DrWatson
@quickactivate

include("../load_experiment.jl")
include("../../dataset/SequenceFED.jl")
include("../evaluate.jl")
include("../loss.jl")
include("../metrics.jl")



using Flux
using UUIDs
using Plots
using ProgressMeter
using DataFrames
using CSV

function create_metrics(dir)
  _, metadata = parse_best_experiment(dir, :csi; sort="max")
end

function evaluate_model(path, evaluation_folder)
  model, metadata = @eval load_experiment($path)
  include(srcdir("dataset", "$(metadata[:dataset][:type]).jl"))
  metadata[:dataset][:path] = metadata[:dataset_path]
  _, test_data = get_dataset(; metadata[:dataset]...)
  gpu_model = @eval gpu($model)
  (test_x, test_y, pred_y) = evaluate(gpu_model, test_data; device=gpu)
  persistence_y = deepcopy(test_y)
  persistence_y[:,:,:,:,:] .= test_x[:,:,:,:,end:end]
  gpu_persistence = gpu(persistence_y)

  evaluation_path = joinpath(evaluation_folder, metadata[:id])
  mkpath(evaluation_path)
  gpu_test_y = gpu(test_y)
  gpu_pred_y = gpu(pred_y)
  xs = 1:size(pred_y, ndims(pred_y))

  csi_por_frame_modelo = MyLosses.csit(gpu_pred_y, gpu_test_y)
  p = plot(csi_por_frame_modelo, title="CSI promedio por fotograma", xticks=xs; marker=:circle, ylims=(0, 1))
  savefig(p, joinpath(evaluation_path, "csi_modelo.png"))

  csi_por_frame_persistencia = MyLosses.csit(gpu_persistence, gpu_test_y)  
  p = plot(csi_por_frame_persistencia, title="CSI promedio por fotograma", xticks=xs; marker=:circle, ylims=(0, 1))
  savefig(p, joinpath(evaluation_path, "csi_persistencia.png"))

  f1_thresholds_modelo = MyLosses.f1_threshold(pred_y, test_y, 0.05:0.05:0.95)
  p = plot(f1_thresholds_modelo, title="F1 score por umbral", xticks=0.05:0.05:0.95; marker=:circle, ylims=(0, 1))
  savefig(p, joinpath(evaluation_path, "f1_modelo.png"))

  f1_thresholds_persistencia = MyLosses.f1_threshold(persistence_y, test_y, 0.05:0.05:0.95)
  p = plot(f1_thresholds_persistencia, title="F1 score por umbral", xticks=0.05:0.05:0.95; marker=:circle, ylims=(0, 1))
  savefig(p, joinpath(evaluation_path, "f1_persistencia.png"))

  tabla_confusion_modelo = DataFrame(confmatrix(pred_y, test_y, 0.05:0.05:0.95))
  CSV.write(joinpath(evaluation_path, "confusion_modelo.csv"), tabla_confusion_modelo)

  tabla_confusion_modelo = DataFrame(confmatrix(persistence_y, test_y, 0.05:0.05:0.95))
  CSV.write(joinpath(evaluation_path, "confusion_persistencia.csv"), tabla_confusion_modelo)

  BSON.@save joinpath(evaluation_path, "scores.bson") test_y pred_y test_x persistence_y
  
end

function main()
  models = readdir(datadir("models"); join=true)
  experiments = String[]
  append!.(Ref(experiments), readdir.(models; join=true))
  results = arg_parse_best_experiment.(experiments, :csi; sort="max")
  for r in results
    isnothing(r[3]) && continue
    r[3][:file_path] = r[1]
  end

  fed_results = map(last, filter(x -> !isnothing(x[3]) && x[3][:dataset][:type] == "SequenceFED", results))

  evaluation_folder = datadir("evaluation", replace(string(now()), ":" => ""))

  
  for r in fed_results
    try
      evaluate_model(r[:file_path], evaluation_folder)
    catch e
      println(e)
    end
  end
  

end


function fix_ids!()
  models = readdir(datadir("models"); join=true)
  experiments = String[]
  append!.(Ref(experiments), readdir.(models; join=true))
  ids = Dict{String, String}()
  for e in experiments
    ids[e] = string(uuid4())
  end
  for e in experiments
    instances = readdir(e; join=true)
    iszero(length(instances)) && continue
    metadata = []
    for i in instances
    !endswith(i, "bson") && continue
      try
        push!(metadata, BSON.parse(i))
      catch e
        println(i)
        throw(e)
      end
    end
    
    to_save = []
    for (loc, p) in zip(instances, metadata)
      try
        if !(typeof(p) <: Dict)
          to_save = []
          break
        end
        if !haskey(p, :id)
          p[:id] = ids[e]
          push!(to_save, (loc, p))
        end
      catch e
        println(loc)
        println(p)
        throw(e)
      end
    end
    println("saving $(length(to_save)) experiments at $e")
    for (loc, p) in to_save
      bson(loc, p)
    end
  end
end

