using Flux.Optimise

function get_opt(lr)
  Optimiser(ADAM(lr), ExpDecay(1.0))
end