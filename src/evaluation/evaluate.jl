using ProgressMeter

function evaluate(model, dataset; device)
  test_data = collect(dataset)
  tx, ty = first(test_data)
  B = size(tx, 4)
  N = sum(x->size(x[1],4), test_data)
  W,H,C = size(tx)[1:3]
  Tx = size(tx, 5)
  Ty = size(ty, 5)

  test_y = zeros(Float32, W,H,C, N, Ty)
  test_x = zeros(Float32, W,H,C, N, Tx)
  pred_y = deepcopy(test_y)
  Flux.testmode!(model)
  @showprogress for (i,(tx, ty)) in enumerate(test_data)
    Flux.reset!(model)
    p_y = cpu(model(device(tx)))
    pred_y[:,:,:,(i-1)*B+1:i*B,:] .= p_y[:,:,:,:,:]
    test_y[:,:,:,(i-1)*B+1:i*B,:] .= ty[:,:,:,:,:]
    test_x[:,:,:,(i-1)*B+1:i*B,:] .= tx[:,:,:,:,:]
  end
  
  return (test_x, test_y, pred_y)
end
