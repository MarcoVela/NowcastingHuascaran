import Pkg
Pkg.add(["Logging", "LoggingExtras", "JSON3", "Dates"])

using Logging
using LoggingExtras
using JSON3
using Dates

lvlstr(lvl::Logging.LogLevel) = lvl >= Logging.Error ? "error" :
                                lvl >= Logging.Warn  ? "warn"  :
                                lvl >= Logging.Info  ? "info"  : "debug"

logging_transform(v) = v
logging_transform(v::Function) = string(v)


function JSONFormat(io, args)
  logmsg = Dict(
      :level => lvlstr(args.level),
      :message => string(args.message),
      :id => string(args.id),
      :date => Dates.now(),
      :payload => Dict((k => logging_transform(v) for (k, v) in args.kwargs)...)
  )
  JSON3.write(io, logmsg)
  println(io)
end

function get_logger(filename::AbstractString)
  f = open(filename, "a")
  function finish_logger()
    flush(f)
    yield()
    close(f)
  end
  FormatLogger(JSONFormat, f), finish_logger
end