using DrWatson
@quickactivate

if ispath(ARGS[1])
  metrics = Symbol[]
  possible_filenames = ARGS
else
  metrics = Symbol.(split(ARGS[1], ','))
  possible_filenames = ARGS[2:end]
end

is_file_ARGS = isfile.(possible_filenames)

if !reduce(&, is_file_ARGS)
  not_files = possible_filenames[(!).(is_file_ARGS)]
  @error "all arguments must be files" not_files
  exit(1)
end
filenames = possible_filenames

include(srcdir("utils", "parse_logs.jl"))
include(srcdir("evaluation", "plot_logs.jl"))


logs_structs = read_log_file.(filenames)

p = plot_logs(logs_structs[begin], metrics; prefix="1")

for (i, log_struct) in enumerate(logs_structs[2:end])
  plot_logs!(log_struct, metrics, prefix="$(i+1)")
end

display(p)