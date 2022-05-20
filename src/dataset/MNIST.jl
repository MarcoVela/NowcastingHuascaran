using HDF5
using DrWatson

function get_dataset(; splitratio, batchsize, path=datadir("exp_raw", "moving-mnist", "mnist_test_seq.h5"))
  mnist_whole = h5read(path, "moving_mnist")
  W,H,C = size(mnist_whole)
  mnist = reshape(mnist_whole, W, H, C, :)
  N = size(mnist, 4)
  N_train = Int(N * splitratio)
  N_test = N - N_train
  mnist_train = view(mnist, :,:,:,1:N_train)
  mnist_test = view(mnist, :,:,:,N_train+1:N)

  mnist_train_x = (mnist_train[:,:,:,t:t+batchsize-1] for t in 1:batchsize:N_train-batchsize+1)

  mnist_test_x = (mnist_test[:,:,:,t:t+batchsize-1] for t in 1:batchsize:N_test-batchsize+1)

  zip(mnist_train_x, mnist_train_x), zip(mnist_test_x, mnist_test_x)
end