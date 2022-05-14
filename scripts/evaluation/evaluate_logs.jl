using DrWatson
@quickactivate

metrics = Symbol.(split(ARGS[1], ','))
possible_filenames = ARGS[2:end]

iszero(length(possible_filenames)) && error("Must provide files")

is_file_ARGS = isfile.(possible_filenames)

if !reduce(&, is_file_ARGS)
  not_files = possible_filenames[(!).(is_file_ARGS)]
  @error "all arguments must be files" not_files
  exit(1)
end
filenames = possible_filenames
using ImageClipboard

include(srcdir("utils", "parse_logs.jl"))
include(srcdir("evaluation", "plot_logs.jl"))


logs_structs = read_log_file.(filenames)

pref = length(logs_structs) == 1 ? "" : "1"

p = plot_logs(logs_structs[begin], metrics; prefix=pref)

for (i, log_struct) in enumerate(logs_structs[2:end])
  plot_logs!(log_struct, metrics, prefix="$(i+1)")
end

temp_path, io = mktemp()

show(io, MIME("image/png"), p)
close(io)
display(p)

clipboard_img(load(temp_path))
