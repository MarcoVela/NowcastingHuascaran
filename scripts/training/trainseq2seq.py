# Copyright 2022 The Flax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""seq2seq addition example."""

# See issue #620.
# pytype: disable=wrong-keyword-args

from typing import Any, Dict, Tuple

from absl import flags
from absl import app
# from clu import metric_writers
from flax.training import train_state
import jax
import jax.numpy as jnp
import optax
from flax.training import train_state, checkpoints
import sys
import os

current_folder = os.path.dirname(os.path.realpath(__file__))

sys.path.append(os.path.join(current_folder, '../..'))
from src.architecture.Seq2Seq01 import Seq2seq
from src.dataset.SequenceFED import get_dataset
from src.utils import drwatson


Array = Any
FLAGS = flags.FLAGS
PRNGKey = Any

flags.DEFINE_string('dataset', default=None, help="")
flags.DEFINE_string('architecture', default=None, help="")
flags.DEFINE_string('dataset_path', default=".", help="")
flags.DEFINE_integer('epochs', default=None, help="")
flags.DEFINE_string('optimiser', default=None, help="")


def get_train_state(model: Seq2seq, params, lr) -> train_state.TrainState:
  """Returns a train state."""
  tx = optax.adam(lr)
  state = train_state.TrainState.create(
      apply_fn=model.apply, params=params, tx=tx)
  return state


def binary_cross_entropy_loss(y_pred: Array, y: Array, e=jnp.finfo(jnp.float32).eps) -> float:
  """Returns cross-entropy loss."""
  xe = jnp.sum(-y * jnp.log(y_pred + e) - (1. - y)*jnp.log(1 - y_pred + e), axis=(2,3,4))
  return jnp.mean(xe)


def compute_metrics(y_pred: Array, y: Array) -> Dict[str, float]:
  """Computes metrics and returns them."""
  loss = binary_cross_entropy_loss(y_pred, y)
  accuracy = jnp.mean(jnp.sum(y_pred * y, axis=(2,3,4)))
  metrics = {
      'loss': loss,
      'accuracy': accuracy,
  }
  return metrics


@jax.jit
def train_step(state: train_state.TrainState, X: Array, y: Array, lstm_rng: PRNGKey) -> Tuple[train_state.TrainState, Dict[str, float]]:
  """Trains one step."""
  lstm_key = jax.random.fold_in(lstm_rng, state.step)

  def loss_fn(params):
    preds = state.apply_fn({'params': params},
                               X,
                               rngs={'lstm': lstm_key})
    loss = binary_cross_entropy_loss(preds, y)
    return loss, preds

  grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
  (_, preds), grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  metrics = compute_metrics(preds, y)

  return state, metrics


def main(_):
  _, architecture_params, _ = drwatson.parse_savename(FLAGS.architecture)
  _, optimiser_params, _ = drwatson.parse_savename(FLAGS.optimiser)
  _, dataset_params, _ = drwatson.parse_savename(FLAGS.dataset)
  dataset_path = FLAGS.dataset_path

  epochs = FLAGS.epochs

  model = Seq2seq(out_length=architecture_params["out"], 
                  features=architecture_params["features"])


  dataset_factory = lambda : get_dataset(N=dataset_params["N"], 
                                            splitratio=dataset_params["splitratio"],
                                            batchsize=dataset_params["batchsize"],
                                            path=dataset_path)

  rng = jax.random.PRNGKey(42)

  rng1, rng2 = jax.random.split(rng)

  variables = model.init(
    {'params': rng1, 'lstm': rng2},
    jnp.ones((dataset_params["N"], 2, 64, 64, 1)),
  )

  experiment_params = {
    'lr': optimiser_params['lr'],
    'features': architecture_params['features'],
    'batchsize': dataset_params['batchsize'],
    'loss': 'binarycrossentropy',
    'opt': 'ADAM',
  }

  model_params = {
    'architecture': 'FlaxSeq2Seq',
    'dataset': 'SequenceFED',
  }

  experiment_id = drwatson.savename(experiment_params)
  model_id = drwatson.savename(model_params)

  experiment_dir = drwatson.datadir("models", model_id, experiment_id)

  # writer = metric_writers.create_default_writer(experiment_dir)

  state = get_train_state(model, variables['params'], optimiser_params['lr'])

  print("starting training")

  for epoch in range(1, 1+epochs):
    print(f"Epoch {epoch}")
    train_dataset, test_dataset = dataset_factory()
    for i, (train_x, train_y) in enumerate(train_dataset):
      state, metric = train_step(state, train_x, train_y, rng)
    # writer.write_scalars(epoch, metric)
    checkpoints.save_checkpoint(experiment_dir, target=state, step=epoch, keep_every_n_steps=1)

app.run(main)