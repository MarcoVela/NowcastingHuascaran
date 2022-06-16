using HDF5
using DrWatson

function read_from_file(path)
  h5read(path, "FED")
end

function read_from_folder(path)
  paths = readdir(path; join=true)
  datasets = read_from_file.(paths)
  cat(datasets...; dims=4)
end

function get_dataset(; splitratio, batchsize, path)
  @assert ispath(path) "$path is not a path"
  if isfile(path)
    dataset = read_from_file(path)
  elseif isdir(path)
    dataset = read_from_folder(path)
  end
  W,H,C = size(dataset)
  ds = reshape(dataset, W, H, C, :)
  N = size(ds, 4)
  N_train = Int(N * splitratio)
  N_test = N - N_train
  ds_train = view(ds, :,:,:,1:N_train)
  ds_test = view(ds, :,:,:,N_train+1:N)

  ds_train_x = (ds_train[:,:,:,t:t+batchsize-1] for t in 1:batchsize:N_train-batchsize+1)

  ds_test_x = (ds_test[:,:,:,t:t+batchsize-1] for t in 1:batchsize:N_test-batchsize+1)

  zip(ds_train_x, ds_train_x), zip(ds_test_x, ds_test_x)
end
