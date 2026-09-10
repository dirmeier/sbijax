:github_url: https://github.com/dirmeier/sbijax

👋 Welcome to ``sbijax``!
=========================

.. div:: sd-text-left sd-font-italic

    Simulation-based inference in JAX

----

``Sbijax`` is a Python library for neural simulation-based inference and
approximate Bayesian computation using `JAX <https://github.com/google/jax>`_.
It implements recent methods, such as *Simulated Annealing ABC*,
*Surjective Neural Likelihood Estimation*, *Neural Approximate Sufficient
Statistics* or *Neural Posterior Score Estimation*, as well as calibration and
convergence diagnostics.

.. caution::

    ⚠️ As per the LICENSE file, there is no warranty whatsoever for this free software tool. If you discover bugs, please report them.

Quickstart
----------

``Sbijax`` implements a low-level, functional API in the idiom of dm-haiku and
blackjax: every method is a factory returning a tuple of pure functions. 
All a user needs to define is a prior function, a simulator function
and an inferential algorithm. For example, you can define a
neural likelihood estimation method and generate posterior samples like this:

.. code-block:: python

    from jax import numpy as jnp, random as jr
    from sbijax import nle, train, sample, simulate
    from sbijax.mcmc import make_sampler, nuts
    from sbijax.nn import make_maf
    from tensorflow_probability.substrates.jax import distributions as tfd

    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.zeros(2), jnp.ones(2))
    ), batch_ndims=0)

    def simulator_fn(seed, theta):
        p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
        y = theta["theta"] + p.sample(seed=seed)
        return y

    estimator = nle(make_maf(2))

    y_observed = jnp.array([-1.0, 1.0])
    data = simulate(jr.key(1), prior, simulator_fn, n=10_000)
    params, info = train(jr.key(2), estimator, data)
    samples, _ = sample(
        jr.key(3), estimator, params, y_observed,
        sampler=make_sampler(nuts, prior=prior),
    )

Workflow
--------

Every method in ``sbijax`` takes the same two user-supplied inputs:

* a **prior**: needs a ``.sample(seed=rng_key)`` method returning a pytree, a
  batched form ``.sample(seed=rng_key, sample_shape=(n,))``, and a
  ``.log_prob(theta)`` method accepting that same pytree structure. A pytree
  here just means a (possibly nested) dict of arrays, e.g. what the
  ``prior`` in the example above returns from ``.sample(...)``:

  .. code-block:: python

      {"theta": Array([0.62, 0.84])}

  Nothing checks that the prior is literally a
  ``tensorflow_probability.substrates.jax`` (``tfd``) distribution, but every
  example uses a :code:`tfd.JointDistributionNamed`, which gives you both
  methods for free
* a **simulator**: a plain function ``(seed, theta) -> y``. Use those exact
  argument names (or ``**kwargs``) — ABC methods (``sabc``/``smcabc``) call it
  by keyword internally. ``simulate``/``run_sequential`` always call it with a
  **batched** ``theta`` (a leading axis of size ``n``), so it needs to handle
  a batch of parameter draws, not a single one

From these two, the same pipeline applies to every neural estimator (NLE, NPE,
FMPE, NPSE, NRE, SNLE):

.. mermaid::

    %%{init: {"flowchart": {"curve": "basis", "nodeSpacing": 60, "rankSpacing": 60, "htmlLabels": true, }, "themeVariables": {"fontFamily": "Helvetica, Arial, sans-serif", "fontSize": "20px", "nodePadding": "20px"}}}%%
    flowchart LR
        P([prior]) -->|simulate| D[(data)]
        M([simulator]) -->|simulate| D
        D -->|"nle/npe/fmpe/..."| O([objective])
        O -->|train| T([params])
        T -->|sample| R([posterior])

        classDef input fill:#e8eef7,stroke:#5b7fa6,stroke-width:1.5px,color:#1c2b3a,font-weight:600;
        classDef data fill:#fbf3e3,stroke:#c99a3c,stroke-width:1.5px,color:#3a2f1c,font-weight:600;
        classDef obj fill:#eaf3ea,stroke:#4c8c5a,stroke-width:1.5px,color:#1c3a22,font-weight:600;
        classDef out fill:#f6e8ee,stroke:#a65b82,stroke-width:1.5px,color:#3a1c2b,font-weight:600;
        class P,M input
        class D data
        class O obj
        class T,R out

1. ``simulate(rng_key, prior, simulator, n)`` draws ``n`` prior/simulation pairs.
2. A factory (``nle``, ``npe``, ``fmpe``, ...) wraps a neural network into an
   ``ObjectiveFns`` record of pure functions.
3. ``train(rng_key, objective, data)`` fits it, returning ``params``.
4. ``sample(rng_key, objective, params, observable)`` draws posterior samples
   (MCMC-based methods also take ``sampler=make_sampler(nuts, prior=prior)``).

Installation
------------

You can install ``sbijax`` from PyPI using:

.. code-block:: bash

    pip install sbijax

To install the latest GitHub <RELEASE>, just call the following on the command line:

.. code-block:: bash

    pip install git+https://github.com/dirmeier/sbijax@<RELEASE>

See also the installation instructions for `JAX <https://github.com/google/jax>`_, if you plan to use :code:`sbijax` on GPU/TPU.

Contributing and Support
-------------------------

If you have questions, encounter problems, or need support with this software, please use the following channels:

* **Questions & Discussions:** For general questions, usage help, or architectural discussions, please open a new thread in our `GitHub Discussions <https://github.com/dirmeier/sbijax/discussions>`_ tab.
* **Bug Reports & Feature Requests:** To report a bug, software problem, or suggest a new feature, please submit an issue via our `GitHub Issue Tracker <https://github.com/dirmeier/sbijax/issues>`_. Please check existing issues before opening a new one to ensure it hasn't already been reported.

Code contributions in the form of pull requests are more than welcome. A good way to
start is to check out issues labelled
`good first issue <https://github.com/dirmeier/sbijax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_. If you are unsure, if starting to work on a PR makes
sense, feel free to open an issue or discussion thread.

In order to contribute:

1) Clone :code:`sbijax` and install :code:`uv` from `here <https://docs.astral.sh/uv/getting-started/installation/>`_.
2) Install all dependencies using ``uv sync --all-groups``.
3) Install the Git hooks:

   .. code-block:: bash

       uv run pre-commit install -t pre-commit -t commit-msg

4) Create a new branch locally :code:`git checkout -b feature/my-new-feature` or :code:`git checkout -b issue/fixes-bug`.
5) Implement your contribution and ideally a test case.
6) Check your work (see below).
7) Submit a PR 🙂

Development commands
====================

The project uses ``uv`` for everything:

.. code-block:: bash

    uv sync --all-groups
    uv run pytest
    uv run ruff check sbijax examples
    uv run ruff check --fix sbijax examples
    uv run ruff format sbijax examples
    uv run mypy sbijax examples
    uv run pre-commit run --all-files

License
-------

:code:`sbijax` is licensed under the Apache 2.0 License.

..  toctree::
    :maxdepth: 1
    :hidden:

    🏡 Home <self>
    🧭 Design philosophy <design>
    🔀 Migration guide <migration>
    📚 References <references>

..  toctree::
    :caption:  Tutorials
    :maxdepth: 1
    :hidden:

    Getting started <notebooks/getting_started>
    A more detailed intro  <notebooks/more_detailed_intro>
    SLCP <notebooks/examples>    
    🔧 Custom loops <custom_loops>
    Self-contained examples <examples>

..  toctree::
    :caption: API
    :maxdepth: 2
    :hidden:

    api/index
