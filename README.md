# PriorVGAE

PriorVGAE = PriorVAE + GNN

### Dependencies

- [JAX](https://jax.readthedocs.io/en/latest/)
- [Jraph](https://github.com/deepmind/jraph)
- [Haiku](https://github.com/deepmind/dm-haiku)
- [Optax](https://github.com/deepmind/optax)
- NetworkX
- scikit-learn

### Usage

Run either GAE (Graph Autoencoder) or VGAE (Variational GAE):

```zsh
python3 train.py --is_vgae=True
```
