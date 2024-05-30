import distrax
import optax

from jax import numpy as jnp, random as jr
from sbijax import SNL
from sbijax.nn import make_affine_maf


def prior_model_fns():
    return p.sample, p.log_prob


def simulator_fn(seed, theta):
    p = distrax.Normal(jnp.zeros_like(theta), 1.0)
    y = theta + p.sample(seed=seed)
    return y


prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn
model = SNL(fns, make_affine_maf(2))


y_observed = jnp.array([-1.0, 1.0])
data, a = model.simulate_data(jr.PRNGKey(0), n_simulations=5)
params, b = model.fit(jr.PRNGKey(1), data=data, optimizer=optax.adam(0.001))
posterior, c = model.sample_posterior(jr.PRNGKey(2), params, y_observed)
#
# print(posterior)

print(type(data))
print(data)