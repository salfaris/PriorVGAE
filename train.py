import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
from sklearn.metrics import roc_auc_score
from absl import app, flags, logging

import functools
from typing import List, Dict, Any, Tuple

from dataset import load_dataset
from loss import compute_kl_gaussian, compute_mse_loss
from model import VGAE, VGAEOutput

flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate for the optimizer.')
flags.DEFINE_integer('epochs', 200, 'Number of training epochs.')
flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension in the GAE.')
flags.DEFINE_integer('latent_dim', 16, 'Latent dimension in the GAE.')
flags.DEFINE_integer('output_dim', 5, 'Output dimension in the GAE.')
flags.DEFINE_integer('random_seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_frequency', 10, 'How often to evaluate the model.')
# flags.DEFINE_bool('is_vgae', True, 'Using Variational GAE vs vanilla GAE.')
FLAGS = flags.FLAGS


def train(dataset: List[Dict[str, Any]]) -> hk.Params:
  """Training loop."""
  # key = jax.random.PRNGKey(FLAGS.random_seed)
  rng_seq = hk.PRNGSequence(FLAGS.random_seed)
  
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']
  train_graph = graph
  train_senders = train_graph.senders
  train_receivers = train_graph.receivers
  train_labels = jnp.zeros((graph.nodes.shape[0], 5))  # TODO :- Change this!
  
  # Initialize network and optimizer.
  net = hk.transform(
    lambda x: VGAE(FLAGS.hidden_dim, FLAGS.latent_dim, FLAGS.output_dim)(x))
  optimizer = optax.adam(FLAGS.learning_rate)
  params = net.init(next(rng_seq), train_graph)
  opt_state = optimizer.init(params)
  
  @jax.jit
  def loss_fn(params: hk.Params, 
              rng_key: jnp.ndarray,
              graph: jraph.GraphsTuple,
  ) -> jnp.ndarray:
    """Computes VGAE loss."""
    outputs: VGAEOutput = net.apply(params, rng_key, graph)
    log_likelihood = compute_mse_loss(outputs.output.nodes, train_labels)
    kld = jnp.mean(compute_kl_gaussian(outputs.mean, outputs.log_std), axis=-1)
    loss = log_likelihood + kld  # want to maximize this quantity.
    return loss
  
  @jax.jit
  def update(
    params: hk.Params,
    rng_key: jnp.ndarray,
    opt_state: optax.OptState,
    graph: jraph.GraphsTuple,
  ): 
    """Updates the parameters of the network."""
    grads = jax.grad(loss_fn)(params, rng_key, graph)
    updates, new_opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, new_opt_state

  for epoch in range(FLAGS.epochs):
    params, opt_state = update(params, next(rng_seq), opt_state, train_graph)
  
    if epoch % FLAGS.eval_frequency == 0 or epoch == (FLAGS.epochs - 1):
      train_loss = loss_fn(params, next(rng_seq), train_graph)
      train_output = net.apply(params, next(rng_seq), train_graph)
      logging.info(f'epoch: {epoch}, train_loss: {train_loss:.3f}')
  logging.info('Training finished')
  return params


def main(_):
    cora_ds = load_dataset('./dataset/cora.pickle')
    _ = train(cora_ds)

if __name__ == '__main__':
    app.run(main)