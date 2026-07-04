Migration guide: 0.3 → 0.4
==========================

``sbijax`` 0.4 replaces the object-oriented estimator classes with a **low-level
functional API**: every method is a factory that returns a record of pure
functions, and training and sampling are **free driver functions**
(:func:`sbijax.train`, :func:`sbijax.sample`) that operate on that record. See
:doc:`design` for the full rationale. This is a breaking release; this guide
maps the old API onto the new one.

At a glance
-----------

.. list-table::
    :header-rows: 1

    * - 0.3 (object-oriented)
      - 0.4 (functional)
    * - ``NLE(fns, net)``
      - ``nle(net)``
    * - ``model.simulate_data(key)``
      - ``simulate(key, prior, simulator, n=...)``
    * - ``model.fit(key, data=data)``
      - ``params, info = train(key, obj, data, optimizer=...)``
    * - ``model.sample_posterior(key, params, y)``
      - ``samples, info = sample(key, obj, params, y, sampler=...)``

Class names become factory functions
-------------------------------------

Every estimator class is now a lower-case factory that takes **only the
network**:

.. list-table::
    :header-rows: 1

    * - 0.3
      - 0.4
    * - ``NPE``, ``FMPE``
      - ``npe``, ``fmpe``
    * - ``NLE``, ``SNLE``
      - ``nle``, ``snle``
    * - ``NRE``
      - ``nre``
    * - ``SABC``, ``SMCABC``
      - ``sabc``, ``smcabc``
    * - ``NASS``, ``NASSS``
      - ``nass``, ``nasss``
    * - ``NPSE`` (``sbijax.experimental``)
      - ``npse`` (now top-level)
    * - ``CMPE``
      - ``cmpe`` (moved to ``sbijax.experimental``)
    * - ``AiO`` (``sbijax.experimental``)
      - ``aio``

Construction: the factory takes only the network
-------------------------------------------------

The old ``model_fns = (prior_fn, simulator)`` tuple is gone, and so is passing
the prior to the estimator. The prior is a ``tfd.Distribution`` used only to
generate data; the network is the sole argument to the factory.

.. code-block:: python

    # 0.3
    fns = prior_fn, simulator_fn          # prior_fn is a zero-arg factory
    model = NLE(fns, make_maf(2))

    # 0.4
    prior = tfd.JointDistributionNamed(
        dict(theta=tfd.Normal(jnp.zeros(2), 1.0)), batch_ndims=0
    )
    estimator = nle(make_maf(2))          # network only

Data simulation is a standalone module
--------------------------------------

.. code-block:: python

    # 0.3
    data, _ = model.simulate_data(jr.PRNGKey(0), n_simulations=10_000)

    # 0.4
    from sbijax import simulate, stack
    data = simulate(jr.key(0), prior, simulator_fn, n=10_000)
    # append another round drawn from a proposal:
    more = simulate(jr.key(1), prior, simulator_fn, proposal=proposal, n=10_000)
    data = stack(data, more)

Training is a free driver; the optimizer is injected here
---------------------------------------------------------

``train`` is a free function, not a method. The optimizer is passed to ``train``
(defaulting to ``optax.adam(3e-4)``), never baked into the factory. It returns
the fitted parameters plus a generic ``Info`` (``round`` +
``losses``); the per-method ``*Info`` records are gone.

.. code-block:: python

    # 0.3
    params, losses = model.fit(jr.PRNGKey(1), data=data)

    # 0.4
    from sbijax import train
    params, info = train(jr.key(1), estimator, data, optimizer=optax.adam(3e-4))
    losses = info.losses

Sampling is a free driver; the prior travels in the sampler
-----------------------------------------------------------

``sample`` (renamed from ``sample_posterior``) is a free function taking
``params`` explicitly and returning ``(samples, info)`` -- a named pytree of
draws (``{"theta": array}``, leaves of shape ``(n_chains, n_draws, dim)``) plus a
small sampling record. There is no ``arviz`` / ``InferenceData``.

For **amortized** posterior methods (``npe``/``fmpe``/``npse``), sampling needs
nothing but ``params``:

.. code-block:: python

    from sbijax import sample
    samples, info = sample(jr.key(2), estimator, params, y_obs)
    theta = samples["theta"]              # (n_chains, n_draws, dim)

For **likelihood/ratio** methods (``nle``/``nre``), the posterior is formed at
sample time, so the prior travels inside a sampler built with
:func:`sbijax.mcmc.make_sampler`:

.. code-block:: python

    from sbijax import sample
    from sbijax.mcmc import make_sampler, nuts

    samples, info = sample(
        jr.key(2), estimator, params, y_obs,
        sampler=make_sampler(nuts, prior=prior),
    )

Because the prior lives in the sampler, one trained likelihood is reusable under
different priors without retraining.

Diagnostics and plotting
-------------------------

``sbijax`` no longer ships plotting or ``arviz`` -- build figures from the
returned arrays. Convergence diagnostics are :func:`sbijax.ess` /
:func:`sbijax.rhat` (BlackJAX-backed), applied to the returned ``samples``;
calibration is :func:`sbijax.sbc`.

.. code-block:: python

    # 0.3
    sbijax.plot_posterior(inference_result)

    # 0.4
    theta = samples["theta"].reshape(-1, samples["theta"].shape[-1])
    # ... your own matplotlib figure ...
    print("R-hat:", sbijax.rhat(samples))

Sequential inference
--------------------

Multi-round inference is still :func:`sbijax.run_sequential`, now driving the
free ``train``/``sample`` internally. Pass a ``sampler`` for likelihood/ratio
methods; NPE switches to its atomic proposal-posterior loss in rounds > 0
automatically.

.. code-block:: python

    from sbijax import npe, run_sequential

    params, info = run_sequential(
        jr.key(0), npe(make_maf(2)), prior, simulator_fn, y_obs,
        n_rounds=3, n_simulations_per_round=5_000,
    )

Summary networks
----------------

Summary networks are trained by the same ``train`` and expose ``summarize_fn``:

.. code-block:: python

    from sbijax import nass, train, summarized_estimator

    sn = nass(make_nass_net(2, [64, 64]))
    sn_params, _ = train(jr.key(0), sn, data)
    summaries = sn.summarize_fn(sn_params, data["y"])

    # chain a summary net into a downstream estimator
    est = summarized_estimator(nle(make_maf(2)), sn, sn_params)
    params, _ = train(jr.key(1), est, data)
    samples, _ = sample(jr.key(2), est, params, y_obs,
                        sampler=make_sampler(nuts, prior=prior))
