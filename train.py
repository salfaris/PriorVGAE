import haiku as hk
import jax
import jax.numpy as jnp
import optax
from sklearn.metrics import roc_auc_score
from absl import app, flags, logging

import functools
from typing import List, Dict, Any

from dataset import load_dataset
from loss import compute_vgae_loss, compute_gae_loss
from model import gae_encoder, vgae_encoder

flags.DEFINE_float('learning_rate', 1e-2, 'Learning rate for the optimizer.')
flags.DEFINE_integer('epochs', 200, 'Number of training epochs.')
flags.DEFINE_integer('hidden_dim', 32, 'Hidden dimension in the AE.')
flags.DEFINE_integer('latent_dim', 16, 'Latent dimension in the AE.')
flags.DEFINE_integer('random_seed', 42, 'Random seed.')
flags.DEFINE_integer('eval_frequency', 10, 'How often to evaluate the model.')
flags.DEFINE_bool('is_vgae', True, 'Using Variational GAE vs vanilla GAE.')
FLAGS = flags.FLAGS


def train(dataset: List[Dict[str, Any]]) -> hk.Params:
  """Training loop."""
  key = jax.random.PRNGKey(FLAGS.random_seed)
  # Transform impure network to pure functions with hk.transform.
  net_fn = vgae_encoder if FLAGS.is_vgae else gae_encoder
  net_fn = functools.partial(
    net_fn, hidden_dim=FLAGS.hidden_dim, latent_dim=FLAGS.latent_dim)
  net = hk.without_apply_rng(hk.transform(net_fn))
  
  # Get a candidate graph and label to initialize the network.
  graph = dataset[0]['input_graph']
  train_graph = graph
  train_senders = train_graph.senders
  train_receivers = train_graph.receivers
  train_labels = jnp.zeros(train_senders.shape)  # TODO :- Change this!
  
  # Initialize the network.
  key, param_key = jax.random.split(key)
  params = net.init(param_key, train_graph)
  # Initialize the optimizer.
  opt_init, opt_update = optax.adam(FLAGS.learning_rate)
  opt_state = opt_init(params)
  
  if FLAGS.is_vgae:
    key, loss_key = jax.random.split(key)
    loss_fn = functools.partial(compute_vgae_loss, rng_key=loss_key)
  else:
    loss_fn = compute_gae_loss
  compute_loss_fn = functools.partial(loss_fn, net=net)
  # We jit the computation of our loss, since this is the main computation.
  compute_loss_fn = jax.jit(jax.value_and_grad(compute_loss_fn, has_aux=True))

  for epoch in range(FLAGS.epochs):
    (train_loss,
     train_preds), grad = compute_loss_fn(params, train_graph, train_senders,
                                          train_receivers, train_labels)

    updates, opt_state = opt_update(grad, opt_state, params)
    params = optax.apply_updates(params, updates)
    if epoch % FLAGS.eval_frequency == 0 or epoch == (FLAGS.epochs - 1):
      logging.info(f'epoch: {epoch}, train_loss: {train_loss:.3f}')
  logging.info('Training finished')
  return params


def main(_):
    cora_ds = load_dataset('./dataset/cora.pickle')
    _ = train(cora_ds)

if __name__ == '__main__':
    app.run(main)