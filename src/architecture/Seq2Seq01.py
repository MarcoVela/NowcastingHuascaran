from typing import Tuple, Any
import jax
from jax import numpy as jnp, random
from flax import linen as nn
import functools

from src.utils import drwatson

Array = Any

class EncoderConvLSTM(nn.Module):
  @functools.partial(
    nn.scan,
    variable_broadcast='params',
    in_axes=1,
    out_axes=1,
    split_rngs={'params': False}
  )
  @nn.compact
  def __call__(self, state: Array, x: Array) -> Tuple[Array, Array]:
    _, h = state
    hidden_features = h.shape[-1]
    return nn.ConvLSTM(hidden_features, (5, 5))(state, x)

  @staticmethod
  def initialize_carry(b_size: int, h_size: Tuple[int, int, int]):
    # Use a dummy key since the default state init fn is just zeros.
    return nn.ConvLSTM.initialize_carry(
      jax.random.PRNGKey(0), 
      (b_size,), 
      h_size)

class Encoder(nn.Module):
  features: int

  @nn.compact
  def __call__(self, inputs: Array):
    batch_size = inputs.shape[0]
    hidden_size = inputs.shape[2:-1]
    lstm = EncoderConvLSTM(name='encoder_convlstm')
    init_state = lstm.initialize_carry(batch_size, hidden_size + (self.features,))
    final_state, _ = lstm(init_state, inputs)
    return final_state

class DecoderConvLSTM(nn.Module):
  out_length: int


  # Scan es como un map, mapea sobre una dimension de las entradas de acuerdo a su tamaño
  # Se debe colocar una funcion que recibe dos parametros, el carry y el input
  # Ver como configurarla para tirar resultados de tamaño constante -> El decorador no puede porque
  # se genera durante definicion de clase
  # Ver que pasa si no se sobreescribe el state
  # Ver que pasa si se devuelve probs en el estado tambien
  @nn.compact
  def aux_convlstm(self, carry: Tuple[Array, Array], _) -> Array:
    lstm_state, x = carry
    hidden_features = x.shape[-1]
    lstm_state, y = nn.ConvLSTM(hidden_features, (5, 5))(lstm_state, x)
    logits = nn.Dense(features=1)(y)
    probs = nn.sigmoid(logits)
    return (lstm_state, probs), probs


  def __call__(self, carry: Tuple[Array, Array]) -> Array:
    decode_convlstm = functools.partial(nn.scan,
                                        variable_broadcast='params',
                                        length=self.out_length,
#                                         out_axis=1,
                                        in_axis=1,
                                        split_rngs={'params': False})
    return decode_convlstm(self.aux_convlstm)(carry, None)

class Decoder(nn.Module):
  init_state: Tuple[Any]
  out_length: int

  @nn.compact
  def __call__(self, inputs: Array) -> Tuple[Array, Array]:
    lstm = DecoderConvLSTM(out_length=self.out_length)
    # Revisar si esta es la forma correcta de indexar
    init_carry = (self.init_state, inputs)
    _, probs = lstm(init_carry)
    return probs

class Seq2seq(nn.Module):
  out_length: int
  features: int

  @nn.compact
  def __call__(self, inputs: Array) -> Array:
    init_decoder_state = Encoder(features=self.features)(inputs)
    predictions = Decoder(
      init_state=init_decoder_state,
      out_length=self.out_length,
    )(inputs[:, -1:-1])
    return predictions


