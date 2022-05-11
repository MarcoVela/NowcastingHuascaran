using DrWatson
@quickactivate

using ArgParse

s = ArgParseSettings()

@add_arg_table s begin
  "--spatial", "-s"
      help = "spatial resolution of grid in kilometers"
      arg_type = Int
      default = 8
  "--temporal", "-t"
      help = "temporal resolution of grid in minutes"
      arg_type = Int
      default = 5
  "--north", "-N"
      help = "degrees of top limit of grid"
      arg_type = Float32
      default = NaN32
  "--south", "-S"
      help = "degrees of bottom limit of grid"
      arg_type = Float32
      default = NaN32
  "--east", "-E"
      help = "degrees of left limit of grid"
      arg_type = Float32
      default = NaN32
  "--west", "-W"
      help = "degrees of right limit of grid"
      arg_type = Float32
      default = NaN32
  "--file", "-f"
      help = "file to process (expects to be the output of merge_lcfa.jl)"
      arg_type = String
      required = true
      range_tester = isfile
  "--compression"
      help = "compression level for output file"
      arg_type = Int
      default = 1
      range_tester = x -> (0 <= x <= 9)
end

parsed_args = parse_args(ARGS, s; as_symbols=true)

include(srcdir("dataset", "fed_grid.jl"))

N = isnan(parsed_args[:north]) ? PERU_N : parsed_args[:north]u"°"
S = isnan(parsed_args[:south]) ? PERU_S : parsed_args[:south]u"°"
E = isnan(parsed_args[:east ]) ? PERU_E : parsed_args[:east]u"°"
W = isnan(parsed_args[:west ]) ? PERU_W : parsed_args[:west]u"°"

corners = (; N, S, E, W)

using JLD2

@info "loading lcfa file (this may take a while)"

lcfa_merged = @time load(parsed_args[:file])

records = collect(Iterators.flatten(values(lcfa_merged)))

number_of_entries = length(keys(lcfa_merged))
number_of_records = length(records)
extrema_number_of_records_per_entry = extrema(length, values(lcfa_merged))

@info "stats of file" number_of_entries number_of_records extrema_number_of_records_per_entry

spatial_resolution = parsed_args[:spatial]u"km"
temporal_resolution = Minute(parsed_args[:temporal])

@info "generating ClimateArray"
fed = generate_climarray(records, spatial_resolution, temporal_resolution; corners...)

lcfa_merged = nothing
records = nothing
GC.gc()
props = (; file=split(basename(parsed_args[:file]), '.')[1],
           spatial=parsed_args[:spatial], 
           temporal=parsed_args[:temporal], 
           compression=parsed_args[:compression])

mkpath(datadir("exp_pro", "GLM-L2-LCFA-GRID"))

out_path = datadir("exp_pro", "GLM-L2-LCFA-GRID", savename(props)*".nc")
if ispath(out_path)
  @info "Removing old dataset"
  Base.unlink(out_path)
end

@info "saving results" out_path props...
ncwrite_compressed(out_path, fed; deflatelevel=parsed_args[:compression])