Migration guide: 0.3 → 0.4
==========================

``sbijax`` 0.4 replaces the object-oriented estimator classes with a **fully
functional API**: every method is a factory function returning a record of pure
functions (an :class:`~sbijax.Estimator`, :class:`~sbijax.ABCSampler` or
:class:`~sbijax.SummaryNet`), in the idiom of dm-haiku and blackjax. This is a
breaking release. This guide maps the old API onto the new one.

At a glance
-----------

.. list-table::
    :header-rows: 1

    * - 0.3 (object-oriented)
      - 0.4 (functional)
    * - ``NLE(fns, net)``
      - ``nle(prior, net)``
    * - ``model.simulate_data(key)``
      - ``simulate(key, prior, simulator, n=...)``
    * - ``model.simulate_data_and_possibly_append(...)``
      - ``simulate(..., proposal=...)`` + ``stack(...)`` (or ``run_sequential``)
    * - ``model.fit(key, data=data)``
      - ``params, info = est.fit(key, data)``
    * - ``model.sample_posterior(key, params, y)``
      - ``idata = est.sample(key, params, y)``

Class names become factory functions
-------------------------------------

Every estimator class is now a lower-case factory:

.. list-table::
    :header-rows: 1

    * - 0.3
      - 0.4
    * - ``NPE``, ``FMPE``, ``CMPE``
      - ``npe``, ``fmpe``, ``cmpe``
    * - ``NLE``, ``SNLE``
      - ``nle``, ``snle``
    * - ``NRE``
      - ``nre``
    * - ``SABC``, ``SMCABC``
      - ``sabc``, ``smcabc``
    * - ``NASS``, ``NASSS``
      - ``nass``, ``nasss``
    * - ``NPSE``, ``AiO`` (``sbijax.experimental``)
      - ``npse``, ``aio``

Construction: prior and simulator are no longer bundled
-------------------------------------------------------

The old ``model_fns = (prior_fn, simulator)`` tuple is gone. The prior is now a
``tfd.Distribution`` passed directly, and the simulator is passed to the data
pipeline rather than held by the estimator.

.. code-block:: python

    # 0.3
    fns = prior_fn, simulator_fn          # prior_fn is a zero-arg factory
    model = NLE(fns, make_maf(2))

    # 0.4
    prior = tfd.JointDistributionNamed(
        dict(theta=tfd.Normal(jnp.zeros(2), 1.0)), batch_ndims=0
    )
    estimator = nle(prior, make_maf(2))

Likelihood/ratio methods take their MCMC sampler at construction:

.. code-block:: python

    from sbijax.mcmc import sample_with_nuts
    estimator = nle(prior, make_maf(2), sampler=sample_with_nuts)

Data simulation is a standalone module
--------------------------------------

.. code-block:: python

    # 0.3
    data, _ = model.simulate_data(jr.PRNGKey(0), n_simulations=10_000)

    # 0.4
    from sbijax import simulate, stack
    data = simulate(jr.PRNGKey(0), prior, simulator_fn, n=10_000)
    # append another round drawn from a proposal:
    more = simulate(jr.PRNGKey(1), prior, simulator_fn, proposal=proposal, n=10_000)
    data = stack(data, more)

Fitting returns ``(params, info)``
----------------------------------

``fit`` is stateless and returns the fitted parameters plus a typed per-method
``Info`` record (e.g. :class:`~sbijax.NLEInfo`). ``Info`` exposes at least a
``round`` and a ``(n_epochs, 2)`` ``losses`` array; where the 0.3 ``info`` was a
bare loss array, use ``info.losses`` now.

.. code-block:: python

    # 0.3
    params, losses = model.fit(jr.PRNGKey(1), data=data)

    # 0.4
    params, info = estimator.fit(jr.PRNGKey(1), data)
    losses = info.losses

Sampling returns an ``InferenceData``
-------------------------------------

``sample`` (renamed from ``sample_posterior``) takes ``params`` explicitly and
returns an arviz ``InferenceData`` directly (no diagnostics tuple).

.. code-block:: python

    # 0.3
    idata, diagnostics = model.sample_posterior(jr.PRNGKey(2), params, y_observed)

    # 0.4
    idata = estimator.sample(jr.PRNGKey(2), params, y_observed)

Sequential inference is a standalone driver
-------------------------------------------

Multi-round inference is no longer estimator state (there is no ``n_round``);
use :func:`~sbijax.run_sequential`, which simulates from the current posterior,
appends, and refits. NPE switches to its atomic proposal-posterior loss in
rounds > 0 automatically.

.. code-block:: python

    from sbijax import npe, run_sequential

    estimator = npe(prior, make_maf(2))
    params, info = run_sequential(
        jr.PRNGKey(0), estimator, prior, simulator_fn, y_observed,
        n_rounds=3, n_simulations_per_round=5_000,
    )

For truncated-prior proposals (NPSE / AiO), pass a ``proposal_fn``:

.. code-block:: python

    from sbijax.experimental import npse, make_truncated_proposal

    network = make_score_model(2)
    estimator = npse(prior, network)
    params, info = run_sequential(
        jr.PRNGKey(0), estimator, prior, simulator_fn, y_observed,
        n_rounds=3, n_simulations_per_round=5_000,
        proposal_fn=make_truncated_proposal(prior, network),
    )

ABC and summary networks
------------------------

ABC samplers take ``(prior, simulator)`` and expose only ``sample``:

.. code-block:: python

    # 0.4
    sampler = smcabc(prior, simulator_fn)
    idata = sampler.sample(jr.PRNGKey(0), y_observed, summary=summary_fn, distance=distance_fn)

Summary networks are unchanged in spirit -- ``fit`` then ``summarize``:

.. code-block:: python

    # 0.4
    sn = nass(make_nass_net(2, [64, 64]))
    params, info = sn.fit(jr.PRNGKey(0), data)
    summaries = sn.summarize(params, data["y"])

To infer with the learned summaries, chain the summary network into a
downstream estimator with :func:`~sbijax.summarized_estimator` instead of
threading ``summarize`` by hand (which risks conditioning the estimator on an
un-summarized observation):

.. code-block:: python

    from sbijax import nass, nle, summarized_estimator

    sn = nass(make_nass_net(2, [64, 64]))
    sn_params, _ = sn.fit(jr.PRNGKey(0), data)

    est = summarized_estimator(nle(prior, make_maf(2)), sn, sn_params)
    params, info = est.fit(jr.PRNGKey(1), data)            # trains on summaries
    idata = est.sample(jr.PRNGKey(2), params, y_observed)  # summarizes y_observed
