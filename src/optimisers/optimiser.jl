
module MyOptimisers
  using Flux
  using Flux.Optimise
  function ADAMExpDecay(; lr, decay, decaystep)
    Optimiser(Flux.Optimise.ADAM(lr), ExpDecay(1.0, decay, decaystep))
  end

  ADAM(; lr) = Flux.Optimise.ADAM(lr)

  RMSProp(; lr, rho) = Flux.Optimise.RMSProp(lr, rho)
end


function get_opt(s::Symbol)
  getproperty(MyOptimisers, s)
end