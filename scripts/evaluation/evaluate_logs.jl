using DrWatson
@quickactivate
import Pkg
Pkg.add("ArgParse")
using ArgParse

s = ArgParseSettings()
@add_arg_table! s begin
  "-f"
    help = "Files to include"
    action = :append_arg
end
parsed_args = parse_args(s)
paths = parsed_args["f"]

if length(paths) == 0
  
end