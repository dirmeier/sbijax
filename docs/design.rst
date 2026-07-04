Design philosophy
=================

``sbijax`` is written in the function-first, low-level style of
`dm-haiku <https://github.com/google-deepmind/dm-haiku>`_ and
`BlackJAX <https://github.com/blackjax-devs/blackjax>`_. There are no estimator
classes and no hidden state: every method is a **factory that returns a record
of pure functions**, and the training and sampling loops are **free functions**
that operate on those records. This page explains the design and how it is
implemented.

The one-screen version
----------------------

.. code-block:: python

    import optax
    from jax import numpy as jnp, random as jr
    from sbijax import npe, nle, train, sample, simulate
    from sbijax.mcmc import make_sampler, nuts
    from sbijax.nn import make_maf

    # a factory takes only the network
    obj = npe(make_maf(2))

    # the prior is used only to generate data
    data = simulate(jr.key(0), prior, simulator, n=10_000)

    # `train` is a free driver; the optimizer is injected here
    params, info = train(jr.key(1), obj, data, optimizer=optax.adam(3e-4))

    # `sample` is a free driver; for amortized posteriors nothing else is needed
    samples, sinfo = sample(jr.key(2), obj, params, y_obs)

    # for likelihood/ratio methods the sampler (kernel + prior) is injected here
    lik = nle(make_maf(2))
    params, _ = train(jr.key(1), lik, data, optimizer=optax.adam(3e-4))
    samples, _ = sample(
        jr.key(2), lik, params, y_obs, sampler=make_sampler(nuts, prior=prior)
    )

Principles
----------

**Factories take only the network.** ``npe(net)``, ``nle(net)``, ``nass(net)``
and friends carry no prior, no optimizer, and no sampler. Every *choice about
how* is injected at the driver that owns it. This mirrors ``hk.transform(f)``,
which knows nothing about optax or your data.

**Two free, symmetric drivers.** ``train(rng, obj, data, *, optimizer=...)`` and
``sample(rng, obj, params, observable, *, sampler=...)`` both take the objective
first and their "how" as a keyword. This is exactly BlackJAX's split: the record
carries the *bound* primitives (``SamplingAlgorithm.init``/``step``) while the
loop is a *free* driver (``run_inference_algorithm``).

**The prior enters only where a posterior is formed.** Neural posterior methods
(``npe``/``fmpe``/``npse``) learn the posterior directly -- the prior is baked
into the training data (``theta ~ prior``), so sampling just draws from the
network, prior-free. Neural likelihood/ratio methods (``nle``/``nre``) learn a
prior-*independent* object; the posterior ``p(y|theta) p(theta)`` is only formed
at sample time, so the prior travels inside the sampler. A trained likelihood
can therefore be reused under different priors without retraining -- the point
of the method.

**One generic training loop.** Because ``train`` is a single function, there is no
per-method training code and no per-method ``Info`` record. Each objective
contributes only its loss (via ``step_fn``/``eval_fn``), its parameter init, and
its ``sample_fn``.

Components
----------

A factory returns a record of bound pure functions; two free drivers operate on
it. The prior, optimizer, and sampler enter at the driver that uses each.

.. mermaid::

    flowchart TB
      subgraph factories["factories (network only)"]
        npe["npe / fmpe / npse"]
        nle["nle / snle / nre"]
        nass["nass / nasss"]
        abc["sabc / smcabc"]
      end
      subgraph records["records (bound pure fns)"]
        OF["ObjectiveFns(train, sample_fn, extra)"]
        SF["SummaryFns(train, summarize_fn)"]
        AB["ABCSampler(sample)"]
      end
      subgraph tf["TrainFns primitives (bound)"]
        initf["init_fn(optimizer, rng, batch)"]
        stepf["step_fn(optimizer, rng, state, batch)"]
        evalf["eval_fn(rng, state, batch)"]
      end
      subgraph drivers["free generic drivers"]
        fitd["train(rng, obj, data, *, optimizer)"]
        sampd["sample(rng, obj, params, y, *, sampler)"]
        seqd["run_sequential(rng, obj, prior, simulator, y)"]
      end
      mk["make_sampler(kernel, *, prior)"]

      npe --> OF
      nle --> OF
      nass --> SF
      abc --> AB
      OF --> tf
      SF --> tf
      tf --> fitd
      OF --> sampd
      mk --> sampd
      fitd --> seqd
      sampd --> seqd

The records
-----------

.. list-table::
   :header-rows: 1
   :widths: 22 26 52

   * - Record
     - Returned by
     - Fields
   * - ``TrainingState``
     - ``init_fn``
     - ``params``, ``opt_state`` -- the carry threaded through ``step_fn`` (the
       training analogue of a BlackJAX kernel state).
   * - ``TrainFns``
     - inside every trainable record
     - ``init_fn``, ``step_fn``, ``eval_fn``.
   * - ``ObjectiveFns``
     - ``npe`` / ``fmpe`` / ``npse`` / ``nle`` / ``snle`` / ``nre``
     - ``train`` (a ``TrainFns``), ``sample_fn``, ``extra``.
   * - ``SummaryFns``
     - ``nass`` / ``nasss``
     - ``train`` (a ``TrainFns``), ``summarize_fn``.
   * - ``ABCSampler``
     - ``sabc`` / ``smcabc``
     - ``sample``.
   * - ``Info``
     - ``train``
     - ``round``, ``losses`` (an ``(n_epochs, 2)`` train/validation history).

These records are return values -- you receive instances from the factories and
drivers but never construct them yourself, so they are not part of the public
namespace.

Primitive contracts
--------------------

The bound primitives on a trainable record share these signatures. The optimizer
is a *leading* argument of ``init_fn``/``step_fn``; ``train`` binds it by closure
before jitting (a ``GradientTransformation`` closed over jits fine -- only
*storing* it in the threaded ``TrainingState`` would not).

.. code-block:: text

    init_fn(optimizer, rng_key, batch)        -> TrainingState(params, opt_state)
    step_fn(optimizer, rng_key, state, batch) -> (metrics, TrainingState)   # one optim step
    eval_fn(rng_key, state, batch)            -> metrics                     # validation, no update
    sample_fn(rng_key, params, observable, *, sampler=None) -> (samples, info)
    summarize_fn(params, data)                -> summaries                   # SummaryFns only

``metrics`` is a ``dict`` with at least ``{"loss": ...}``. Amortized posterior
methods draw from the flow and ignore ``sampler``; likelihood/ratio methods
require it. Summary networks reuse the ``TrainFns`` seam and are trained by the
*same* ``train``, exposing ``summarize_fn`` instead of ``sample_fn``. ABC samplers
do no training and expose only ``sample``.

.. mermaid::

    classDiagram
      class TrainingState { +params; +opt_state }
      class TrainFns {
        +init_fn(optimizer, rng, batch) TrainingState
        +step_fn(optimizer, rng, state, batch) tuple
        +eval_fn(rng, state, batch) metrics
      }
      class ObjectiveFns { +train TrainFns; +sample_fn; +extra }
      class SummaryFns { +train TrainFns; +summarize_fn }
      class ABCSampler { +sample }
      class Info { +round int; +losses }
      TrainFns --> TrainingState
      ObjectiveFns --> TrainFns
      SummaryFns --> TrainFns

Training flow
-------------

``train`` reads ``obj.train``, builds a ``TrainingState`` with ``init_fn``, and
threads it through ``step_fn``/``eval_fn`` across epochs with early stopping and
best-parameter tracking. ``TrainingState`` never escapes ``train``; the returned
artifact is ``params``.

.. mermaid::

    sequenceDiagram
      participant U as caller
      participant F as train (free)
      participant T as obj.train
      U->>F: train(rng, obj, data, optimizer)
      F->>T: init_fn(optimizer, rng, batch)
      T-->>F: state
      loop epochs / batches
        F->>T: step_fn(optimizer, rng, state, batch)
        T-->>F: metrics, state
        F->>T: eval_fn(rng, state, val_batch)
        T-->>F: metrics
      end
      F-->>U: params, Info(round, losses)

Sampling flow
-------------

For amortized posteriors, ``sample_fn`` draws directly from the trained flow.
For likelihood/ratio methods it builds the likelihood log-density from
``(params, observable)`` and hands it to the injected sampler, which forms the
posterior target and runs the chains.

.. mermaid::

    sequenceDiagram
      participant U as caller
      participant S as sample (free)
      participant O as obj.sample_fn
      participant K as sampler = make_sampler(nuts, prior=prior)
      U->>S: sample(rng, obj, params, y, sampler=K)
      S->>O: sample_fn(rng, params, y, sampler=K)
      Note over O: loglik_fn(theta) = net.log_prob(y, theta)
      O->>K: K(rng, loglik_fn, n_chains, n_samples, n_warmup)
      Note over K: target = loglik + prior.log_prob; init ~ N(0, I)
      K-->>O: samples, MCMCSampleInfo
      O-->>U: (samples, info)

``make_sampler(kernel, *, prior)`` bundles the MCMC kernel, the prior (which
forms the target ``loglik + log p(theta)``), and ``N(0, I)`` chain
initialisation. Because the prior lives in the sampler, one trained likelihood is
reusable under different priors and kernels.

Samples and diagnostics
-----------------------

``sample`` returns ``(samples, info)`` where ``samples`` is the named prior
pytree (``{"theta": array}``, leaves of shape ``(n_chains, n_draws, dim)``) and
``info`` is a small sampling record (mean acceptance and, for multi-chain MCMC,
``rhat``/``ess``). There is no ``arviz``/``InferenceData`` and no plotting in the
library -- build figures from the returned arrays and check convergence with
:func:`sbijax.ess` / :func:`sbijax.rhat` (thin re-exports of BlackJAX
diagnostics). Calibration is available through :func:`sbijax.sbc`.

Sequential inference
--------------------

Multi-round inference is the free driver :func:`sbijax.run_sequential`, which
simulates from the current posterior each round, appends, and refits. It stays
out of the estimator: NPE switches to its atomic proposal-posterior loss in
rounds > 0 via an ``extra(prior)`` hook, and proposal-invariant methods reuse the
same objective.

.. mermaid::

    sequenceDiagram
      participant U as caller
      participant R as run_sequential
      participant F as train
      U->>R: run_sequential(rng, obj, prior, simulator, y, n_rounds, sampler)
      loop round r
        Note over R: obj_r = obj (r==0) or obj.extra(prior) (r>0, npe atomic)
        R->>R: simulate(prior, simulator, proposal)
        R->>F: train(rng, obj_r, all_data, info=info, optimizer)
        F-->>R: params, info
        Note over R: proposal := sample(rng, obj, params, y, sampler)
      end
      R-->>U: params, info

See :doc:`migration` for moving code from the class-based API.
