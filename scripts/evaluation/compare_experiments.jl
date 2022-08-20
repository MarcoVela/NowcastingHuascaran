using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "--metric", "-m"
    help = "Metrics to compare"
    arg_type = Symbol
    action = :append_arg
  "--dir", "-d"
    help = "Directories of experiments to compare"
    range_tester = isdir
    action = :append_arg
  "--field"
    help = "Field to include (must be a dict)"
    arg_type = Symbol
    action = :append_arg
end

args = parse_args(s; as_symbols=true)
include(srcdir("evaluation", "load_experiment.jl"))

metrics = args[:metric]

experiments_from_folders = load_best_experiment.(args[:dir], args[:metric]; sort=args[:sort])
