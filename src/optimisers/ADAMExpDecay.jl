using Flux.Optimise

function get_opt(lr)
  Optimiser(ADAM(lr), ExpDecay(1.0, .5, 5_000))
end