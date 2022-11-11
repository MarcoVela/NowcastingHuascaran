using HDF5
using DrWatson

#WxHxTxCxN
function get_dataset(; splitratio, batchsize, N, path=datadir("exp_raw", "moving-mnist", "mnist_test_seq.h5"))
  mnist = h5read(path, "moving_mnist")
  mnist_whole = permutedims(mnist, (1,2,5,3,4))
  TOTAL_SAMPLES = size(mnist_whole, 5)
  TOTAL_FRAMES = size(mnist_whole, 3)
  last_train_sample_index = Int(TOTAL_SAMPLES * splitratio)
  mnist_train = view(mnist_whole, :, :, :, :, 1:last_train_sample_index)
  mnist_test = view(mnist_whole, :, :, :, :, last_train_sample_index+1:TOTAL_SAMPLES)

  x_train = (copy(view(mnist_train, :, :, 1:N, :, t:t+batchsize-1)) for t in 1:batchsize:size(mnist_train, 5)-batchsize+1)
  y_train = (copy(view(mnist_train, :, :, N+1:TOTAL_FRAMES, :, t:t+batchsize-1)) for t in 1:batchsize:size(mnist_train, 5)-batchsize+1)
  train_data = zip(x_train, y_train)

  x_test = (copy(view(mnist_test, :, :, 1:N, :, t:t+batchsize-1)) for t in 1:batchsize:size(mnist_test, 5)-batchsize+1)
  y_test = (copy(view(mnist_test, :, :, N+1:TOTAL_FRAMES, :, t:t+batchsize-1)) for t in 1:batchsize:size(mnist_test, 5)-batchsize+1)
  test_data = zip(x_test, y_test)

  train_data, test_data
end