import math
import h5py
import os
import jax.numpy as jnp
import numpy as np

def read_from_file(path):
  f = h5py.File(path, "r")
  return f["FED"][:]

def read_from_folder(path):
  paths = [os.path.join(path, x) for x in os.listdir(path)]
  datasets = [read_from_file(p) for p in paths]
  return np.concatenate(datasets, axis=1)


def get_dataset(splitratio, batchsize, N, path):
  assert os.path.exists(path)
  if os.path.isdir(path):
    dataset = read_from_folder(path)
  elif os.path.isfile(path):
    dataset = read_from_file(path)
  TOTAL_SAMPLES = dataset.shape[1]
  N_TRAIN = math.ceil(TOTAL_SAMPLES * splitratio)
  N_TEST = TOTAL_SAMPLES - N_TRAIN
  dataset_train = dataset[:, :N_TRAIN]
  dataset_test = dataset[:, N_TRAIN+1:]
  x_train = (jnp.array(dataset_train[:N, t:t+batchsize]) for t in range(0, N_TRAIN-batchsize+1, batchsize))
  y_train = (jnp.array(dataset_train[N+1:, t:t+batchsize]) for t in range(0, N_TRAIN-batchsize+1, batchsize))
  train_data = zip(x_train, y_train)
  x_test = (jnp.array(dataset_test[:N, t:t+batchsize]) for t in range(0, N_TEST-batchsize+1, batchsize))
  y_test = (jnp.array(dataset_test[N+1:, t:t+batchsize]) for t in range(0, N_TEST-batchsize+1, batchsize))
  test_data = zip(x_test, y_test)
  return (train_data, test_data)

