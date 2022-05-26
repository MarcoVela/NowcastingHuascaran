using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "--file", "-f"
    help = "File describing the experiment to continue"
    required = true
    range_tester = isfile
  "--epochs"
    help = "Number of epochs"
    arg_type = Int
    range_tester = >(0)
  "--use_dataset_path"
    help = "Overrides the use of dataset path from previous training"
    action = :store_true
  "--force_git"
    help = "Recreates the repository in the same status as of training"
    action = :store_true
end

args = parse_args(s; as_symbols=true)

include(srcdir("utils", "parse_logs.jl"))
logfile = args[:file]
use_dataset_path = args[:use_dataset_path]
force_git = args[:force_git]

(; first_log, train_logs, test_logs, last_log) = read_log_file(logfile)

log_args = first_log[:payload]

epochs = log_args[:epochs]
if !isnothing(last_log)
  if isnothing(args[:epochs]) || (args[:epochs] < log_args[:epochs])
    @error "Training already finished after $(log_args[:epochs])"
  else
    @info "Training trained for $(log_args[:epochs]), continuing up to $(args[:epochs])"
    epochs = args[:epochs]
  end
end

args = log_args

original_args = deepcopy(args)
architecture_type = pop!(args[:architecture], :type)
dataset_type = pop!(args[:dataset], :type)
dataset_path = pop!(args, :dataset_path)

if use_dataset_path && !isnothing(dataset_path)
  args[:dataset][:path] = dataset_path
end

optimiser_type = Symbol(pop!(args[:optimiser], :type))
loss_name = args[:loss]
batchsize = args[:dataset][:batchsize]
lr = args[:optimiser][:lr]

const experiment_id = basename(dirname(logfile))

const _, experiment_params = parse_savename(experiment_id)

const gitcommitid = experiment_params["gitstatus"]

if endswith(gitcommitid, "-dirty")
  @warn "Experiment was in dirty status"
  if force_git
    
  end
end