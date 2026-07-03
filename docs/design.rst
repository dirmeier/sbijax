Design philosophy
=================

``sbijax`` is written in the function-first, low-level style of
`dm-haiku <https://github.com/google-deepmind/dm-haiku>`_ and
`BlackJAX <https://github.com/blackjax-devs/blackjax>`_. There are no estimator
classes and no hidden state: every method is a **factory that returns a record
of pure functions**, and the training and sampling loops are **free functions**
that operate on those records. This page summarises the design and how it is
implemented.

The one-screen version
----------------------

.. code-block:: python

    import optax
    from jax import numpy as jnp, random as jr
    from sbijax import npe, nle, fit, sample, simulate
    from sbijax.mcmc import make_sampler, nuts
    from sbijax.nn import make_maf

    # a factory takes only the network
    obj = npe(make_maf(2))

    # the prior is used only to generate data
    data = simulate(jr.key(0), prior, simulator, n=10_000)

    # `fit` is a free driver; the optimizer is injected here
    params, info = fit(jr.key(1), obj, data, optimizer=optax.adam(3e-4))

    # `sample` is a free driver; for likelihood/ratio methods the sampler
    # (kernel + prior) is injected here
    samples, _ = sample(jr.key(2), obj, params, y_obs)                       # amortized
    samples, _ = sample(jr.key(2), nle_obj, params, y_obs,
                        sampler=make_sampler(nuts, prior=prior))             # MCMC

Principles
----------

**Factories take only the network.** ``npe(net)``, ``nle(net)``, ``nass(net)``
and friends carry no prior, no optimizer, and no sampler. Every *choice about
how* is injected at the driver that owns it. This mirrors ``hk.transform(f)``,
which knows nothing about optax or your data.

**Two free, symmetric drivers.** ``fit(rng, obj, data, *, optimizer=...)`` and
``sample(rng, obj, params, observable, *, sampler=...)`` both take the objective
first and their "how" as a keyword. This is exactly BlackJAX's split: the record
carries the bound primitives (``SamplingAlgorithm.init``/``step``) while the loop
is a free driver (``run_inference_algorithm``).

**The prior enters only where a posterior is formed.** Neural posterior methods
(``npe``/``fmpe``/``npse``) learn the posterior directly — the prior is baked
into the training data (``theta ~ prior``), so sampling just draws from the
network, prior-free. Neural likelihood/ratio methods (``nle``/``nre``) learn a
prior-*independent* object; the posterior ``p(y|theta) p(theta)`` is only formed
at sample time, so the prior travels inside the sampler:
``make_sampler(nuts, prior=prior)``. A trained likelihood can therefore be
reused under different priors without retraining — the point of the method.

**One generic training loop.** Because ``fit`` is a single function, there is no
per-method training code and no per-method ``Info`` record. Each objective
contributes only its loss (via ``step_fn``/``eval_fn``), its parameter init, and
its ``sample_fn``.

The building blocks
-------------------

A factory returns an ``ObjectiveFns`` record of four pure functions:

.. code-block:: text

    ObjectiveFns
      train.init_fn(optimizer, rng, batch)        -> TrainingState(params, opt_state)
      train.step_fn(optimizer, rng, state, batch) -> (metrics, TrainingState)   # one optim step
      train.eval_fn(rng, state, batch)            -> metrics                     # validation
      sample_fn(rng, params, observable, *, sampler=None) -> (samples, info)

``TrainingState`` is the opaque carry threaded through ``step_fn`` — the training
analogue of a BlackJAX kernel state. The optimizer is not stored in it; ``fit``
binds the optimizer into ``init_fn``/``step_fn`` by closure. Summary networks
(``nass``/``nasss``) return a ``SummaryFns`` record — the same ``train``
primitives plus a ``summarize_fn`` — and are trained by the *same* ``fit``.
Approximate Bayesian computation samplers (``sabc``/``smcabc``) do no training
and expose only ``sample``.

Samples and diagnostics
-----------------------

``sample`` returns ``(samples, info)`` where ``samples`` is the named prior
pytree (``{"theta": array}``, leaves of shape ``(n_chains, n_draws, dim)``) and
``info`` is a small sampling record (mean acceptance and, for multi-chain MCMC,
``rhat``/``ess``). There is no ``arviz``/``InferenceData`` and no plotting in the
library — build figures from the returned arrays, and check convergence with
:func:`sbijax.ess` / :func:`sbijax.rhat` (thin re-exports of BlackJAX
diagnostics).

Sequential inference
--------------------

Multi-round inference is the free driver
:func:`sbijax.run_sequential`, which simulates from the current posterior each
round, appends, and refits. It stays out of the estimator: NPE switches to its
atomic proposal-posterior loss in rounds > 0 via an ``extra(prior)`` hook, and
truncated-prior proposals (NPSE/AiO) are a drop-in ``proposal_fn``.

See :doc:`migration` for moving code from the class-based API.
