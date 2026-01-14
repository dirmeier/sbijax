import jax.random as jr
import jax.numpy as jnp
from sbijax import NLE
from sbijax.nn import make_maf
from tensorflow_probability.substrates.jax import distributions as tfd

checkpoints_saved = []

def test_callback(iteration, params, train_loss, val_loss, state):
    checkpoints_saved.append(iteration)
    print(f"✓ Checkpoint {iteration}: train={train_loss:.4f}, val={val_loss:.4f}")

# Multi-dimensional example
prior = lambda: tfd.JointDistributionNamed(dict(
    theta=tfd.MultivariateNormalDiag(jnp.zeros(2), jnp.ones(2))
))

def simulator(seed, theta):
    mean = theta["theta"]
    return tfd.MultivariateNormalDiag(mean, jnp.ones(2)).sample(seed=seed)

fns = (prior, simulator)
flow = make_maf(2)
model = NLE(fns, flow)

# Generate and train
rng_key = jr.PRNGKey(0)
data, _ = model.simulate_data(rng_key, n_simulations=1000)

rng_key, train_key = jr.split(rng_key)
params, losses = model.fit(
    train_key,
    data,
    n_iter=250,
    batch_size=100,
    checkpoint_callback=test_callback,
    checkpoint_every=50,
    n_early_stopping_patience=100  # Increase patience
)

print(f"\n✅ Checkpoints saved at iterations: {checkpoints_saved}")
print(f"Training stopped at iteration: {losses.shape[0]}")  # losses is (n_iters, 2) array
print(f"Final train loss: {losses[-1, 0]:.4f}, Final val loss: {losses[-1, 1]:.4f}")

# Check that at least some checkpoints were saved
assert len(checkpoints_saved) > 0, "No checkpoints were saved!"
print(f"✅ Test passed! Saved {len(checkpoints_saved)} checkpoints")