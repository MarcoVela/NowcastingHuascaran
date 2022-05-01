using HDF5

function get_default_evaluation_params()
  (; N = 12, train_test_ratio=.99, batchsize=1)
end

#WxHxCxNxT
function get_dataset(; train_test_ratio, batchsize, N)
  dataset_path = datadir("exp_raw", "moving-mnist", "mnist_test_seq.h5")
  mnist_whole = h5read(dataset_path, "moving_mnist")
  TOTAL_SAMPLES = size(mnist_whole, 4)
  train_test_split = train_test_ratio
  TOTAL_FRAMES = size(mnist_whole, 5)
  last_train_sample_index = Int(TOTAL_SAMPLES * train_test_split)
  mnist_train = view(mnist_whole, :, :, :, 1:last_train_sample_index, :)
  mnist_test = view(mnist_whole, :, :, :, last_train_sample_index+1:TOTAL_SAMPLES, :)

  x_train = (copy(view(mnist_train, :, :, :, t:t+batchsize-1, 1:N)) for t in 1:batchsize:size(mnist_train, 4)-batchsize+1)
  y_train = (copy(view(mnist_train, :, :, :, t:t+batchsize-1, N+1:TOTAL_FRAMES)) for t in 1:batchsize:size(mnist_train, 4)-batchsize+1)
  train_data = zip(x_train, y_train)

  x_test = (copy(view(mnist_test, :, :, :, t:t+batchsize-1, 1:N)) for t in 1:batchsize:size(mnist_test, 4)-batchsize+1)
  y_test = (copy(view(mnist_test, :, :, :, t:t+batchsize-1, N+1:TOTAL_FRAMES)) for t in 1:batchsize:size(mnist_test, 4)-batchsize+1)
  test_data = zip(x_test, y_test)

  train_data, test_data
end