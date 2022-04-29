using Flux
using Flux.Optimise: StopException, SkipException
using ProgressMeter

function train_single_epoch!(ps, loss, data, opt; cb=() -> ())
  cb = Flux.runall(cb)
  itrsz = Base.IteratorSize(typeof(data))
  n = (itrsz == Base.HasLength()) || (itrsz == Base.HasShape{1}()) ? length(data) : 0
  p = Progress(n; showspeed=true, enabled=!iszero(n))
  for (X, y) in data
    try
      gs = Flux.gradient(ps) do
        loss(X, y)
      end
      Flux.update!(opt, ps, gs)
      cb();
    catch ex
      if ex isa StopException
        break
      elseif ex isa SkipException
        continue
      else
        rethrow()
      end
    finally
      ProgressMeter.next!(p)
    end
  end
end