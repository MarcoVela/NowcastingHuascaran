using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "--metric", "-m"
    help = "Metrics to plot"
    arg_type = Symbol
    action = :append_arg
  "--file", "-f"
    help = "Files to compare"
    range_tester = isfile
    action = :append_arg
  "--x_left"
    help = "Start plot at position x"
    arg_type = Float64
    default = NaN
  "--y_top"
    help = "Top limit for y axis"
    arg_type = Float64
    default = NaN
  "--clipboard"
    help = "Copy plot to clipboard"
    action = :store_true
  "--show"
    help = "Show plot on screen"
    action = :store_true
end

args = parse_args(s; as_symbols=true)
filenames = args[:file]
metrics = args[:metric]

using ImageClipboard

include(srcdir("utils", "parse_logs.jl"))
include(srcdir("evaluation", "plot_logs.jl"))


logs_structs = read_log_file.(filenames)

pref = length(logs_structs) == 1 ? "" : "1"

p = plot_logs(logs_structs[begin], metrics; prefix=pref)

for (i, log_struct) in enumerate(logs_structs[2:end])
  plot_logs!(log_struct, metrics, prefix="$(i+1)")
end

using Plots
x_left, x_right = xlims(p)
y_bot, y_top = ylims(p)

!isnan(args[:y_top]) && (y_top = args[:y_top])
!isnan(args[:x_left]) && (x_left = args[:x_left])

xlims!((x_left, x_right))
ylims!((y_bot, y_top))

if args[:clipboard]
  temp_path, io = mktemp()
  show(io, MIME("image/png"), p)
  close(io)
  clipboard_img(load(temp_path))
  @info "Copied to clipboard"
end

if args[:show]
  display(p)
  @info "Showing plot, press any key to continue..."
  readline()
end

