import Pkg

using JSON3
using Dates
using StructTypes

struct TrainPayload
  exec_time::Float64
  train_loss::Float64
  test_loss::Float64
  epoch::Int
end

struct StartPayload
  batchsize::Int
  opt::String
  test_size::Int
  loss::String
  split_ratio::Float64
  train_size::Int
  throttle::Int
  epochs::Int
  dropout::Float64
  lr::Float64
  architecture::String
end

StructTypes.StructType(::Type{TrainPayload}) = StructTypes.Struct()
StructTypes.StructType(::Type{StartPayload}) = StructTypes.Struct()

struct LogRecord{P}
  payload::P
  date::DateTime
end

payloadtype(::Type{LogRecord{P}}) where {P} = P;

StructTypes.StructType(::Type{<:LogRecord}) = StructTypes.Struct()



function read_log_file(path::AbstractString)
  lines = readlines(path)
  first_line = popfirst!(lines)
  last_line = lines[end]
  if isnothing(findfirst("FINISH", last_line))
    @warn "Last line should be FINISH, program may have been interrupted"
    last_line = nothing
  else
    pop!(lines)
  end
  if isnothing(findfirst("START_PARAMS", first_line)) && isnothing(findfirst("CONTINUE_TRAIN", first_line))
    error("First line must be START_PARAMS or CONTINUE_TRAIN")
  end
  df = dateformat"Y-m-dTH:M:S.s"
  first_log = JSON3.read(first_line, LogRecord{NamedTuple}; dateformat=df)
  train_logs = [
    JSON3.read(x, LogRecord{TrainPayload}; dateformat=df) 
    for x in lines if !isnothing(findfirst("LOSS_DURING_TRAIN", x))
  ]
  test_logs = [
    JSON3.read(x, LogRecord{NamedTuple}; dateformat=df)
    for x in lines if !isnothing(findfirst("EPOCH_TEST", x))
  ]
  sort!(train_logs; by=Base.Fix2(getfield, :date))
  sort!(test_logs; by=Base.Fix2(getfield, :date))
  (; first_log, train_logs, test_logs, last_log = last_line)
end
