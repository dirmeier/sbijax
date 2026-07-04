Custom training and sampling loops
==================================

:func:`sbijax.train` and :func:`sbijax.sample` are convenience drivers over the
low-level primitives that every objective carries. When you need full control --
a custom schedule, gradient accumulation, your own early-stopping rule, or a
bespoke sampling loop -- drive the primitives yourself. This is the same escape
hatch dm-haiku (``init``/``apply``) and BlackJAX (``init``/``step``) provide.

The primitives
--------------

A trainable objective ``obj`` exposes bound pure functions:

.. code-block:: text

    obj.train.init_fn(optimizer, rng, batch)        -> TrainingState(params, opt_state)
    obj.train.step_fn(optimizer, rng, state, batch) -> (metrics, TrainingState)
    obj.train.eval_fn(rng, state, batch)            -> metrics
    obj.sample_fn(rng, params, observable, *, sampler=None) -> (samples, info)

``TrainingState`` is an opaque carry; ``params`` is what you extract at the end.
A ``batch`` is a dict ``{"y": array, "theta": array}`` with the parameters
flattened to a ``(batch_size, dim)`` array -- :func:`sbijax.simulate` returns
``theta`` as the prior pytree, so flatten it once before batching.

A custom training loop
----------------------

.. code-block:: python

    import jax
    import optax
    from jax import random as jr
    from jax._src.flatten_util import ravel_pytree
    from sbijax import npe, simulate
    from sbijax.nn import make_maf

    obj = npe(make_maf(2))
    optimizer = optax.adam(3e-4)

    data = simulate(jr.key(0), prior, simulator, n=10_000)
    # flatten the prior-pytree parameters to a (n, d) array
    theta = jax.vmap(lambda t: ravel_pytree(t)[0])(data["theta"])
    xy = {"y": data["y"], "theta": theta}

    def minibatches(xy, batch_size, key):
        n = xy["y"].shape[0]
        perm = jr.permutation(key, n)
        for i in range(0, n - batch_size + 1, batch_size):
            idx = perm[i : i + batch_size]
            yield {k: v[idx] for k, v in xy.items()}

    # build the carry from a first batch, then jit the step
    first = next(minibatches(xy, 128, jr.key(1)))
    state = obj.train.init_fn(optimizer, jr.key(2), first)
    step = jax.jit(lambda rng, s, b: obj.train.step_fn(optimizer, rng, s, b))

    key = jr.key(3)
    for epoch in range(100):
        for batch in minibatches(xy, 128, jr.fold_in(key, epoch)):
            key, step_key = jr.split(key)
            metrics, state = step(step_key, state, batch)
        # plug in your own validation / early stopping, e.g.:
        # val = obj.train.eval_fn(jr.fold_in(key, epoch), state, val_batch)

    params = state.params

You now hold ``params`` exactly as :func:`sbijax.train` would have returned them.

Custom sampling
---------------

For **amortized** posteriors, ``sample_fn`` is a single draw from the flow --
call it directly:

.. code-block:: python

    samples, info = obj.sample_fn(jr.key(4), params, y_obs)

For **likelihood/ratio** methods the posterior is formed at sample time. The
usual path is to pass a sampler from :func:`sbijax.mcmc.make_sampler`, but you can
also drive a kernel yourself with the low-level :mod:`sbijax.mcmc` routines by
supplying your own log-density. For example, with the slice sampler:

.. code-block:: python

    from jax import numpy as jnp
    from sbijax.mcmc import sample_with_slice

    def log_posterior(theta):
        # theta is the named prior pytree; `network` is the flow nle wraps
        theta_flat = ravel_pytree(theta)[0]
        lp_lik = network.apply(
            params, method="log_prob", y=jnp.atleast_2d(y_obs),
            x=theta_flat[None],
        )
        return jnp.sum(lp_lik) + jnp.sum(prior.log_prob(theta))

    samples, info = sample_with_slice(
        jr.key(5), log_posterior, prior,
        n_chains=4, n_samples=2_000, n_warmup=1_000,
    )

Any of the kernel handles -- ``nuts``, ``mala``, ``rmh``, ``imh`` -- can be passed
to :func:`sbijax.mcmc.make_sampler`, which wraps exactly this pattern (target
``loglik + log p(theta)``, ``N(0, I)`` initialisation, chosen kernel) behind
:func:`sbijax.sample`. Rolling it by hand lets you swap the kernel, change the
initialisation, or condition on a different prior without retraining.
