:github_url: https://github.com/dirmeier/sbijax

👋 Welcome to ``sbijax``!
=========================

.. div:: sd-text-left sd-font-italic

    Simulation-based inference in JAX

----

``Sbijax`` is a Python library for neural simulation-based inference and
approximate Bayesian computation using `JAX <https://github.com/google/jax>`_.
It implements recent methods, such as *Sequential Monte Carlo ABC*,
*Surjective Neural Likelihood Estimation*, *Neural Approximate Sufficient Statistics*
or *Consistency model posterior estimation*, as well as methods to compute model
diagnostics and for visualizing posterior distributions.

.. caution::

    ⚠️ As per the LICENSE file, there is no warranty whatsoever for this free software tool. If you discover bugs, please report them.

Example
-------

``Sbijax`` implements a fully functional API in the idiom of dm-haiku and
blackjax: every method is a factory returning a record of pure functions, with
parameters threaded explicitly. All a user needs to define is a prior, a
simulator function and an inferential algorithm. For example, you can define a
neural likelihood estimation method and generate posterior samples like this:

.. code-block:: python

    from jax import numpy as jnp, random as jr
    from sbijax import nle, simulate
    from sbijax.nn import make_maf
    from tensorflow_probability.substrates.jax import distributions as tfd

    prior = tfd.JointDistributionNamed(dict(
        theta=tfd.Normal(jnp.zeros(2), jnp.ones(2))
    ), batch_ndims=0)

    def simulator_fn(seed, theta):
        p = tfd.Normal(jnp.zeros_like(theta["theta"]), 0.1)
        y = theta["theta"] + p.sample(seed=seed)
        return y

    estimator = nle(prior, make_maf(2))

    y_observed = jnp.array([-1.0, 1.0])
    data = simulate(jr.PRNGKey(1), prior, simulator_fn, n=10_000)
    params, info = estimator.fit(jr.PRNGKey(2), data)
    posterior = estimator.sample(jr.PRNGKey(3), params, y_observed)

Migrating from the 0.3 object-oriented API? See the
:doc:`migration guide <migration>`.

Installation
------------

You can install ``sbijax`` from PyPI using:

.. code-block:: bash

    pip install sbijax

To install the latest GitHub <RELEASE>, just call the following on the command line:

.. code-block:: bash

    pip install git+https://github.com/dirmeier/sbijax@<RELEASE>

See also the installation instructions for `JAX <https://github.com/google/jax>`_, if you plan to use :code:`sbijax` on GPU/TPU.

Contributing
------------

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
`"good first issue" <https://github.com/dirmeier/sbijax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_.

In order to contribute:

1) Clone :code:`sbijax` and install :code:`uv` from `here <https://docs.astral.sh/uv/getting-started/installation/>`_,
2) install all dependencies using ```uv sync``,
3) create a new branch locally :code:`git checkout -b feature/my-new-feature` or :code:`git checkout -b issue/fixes-bug`,
4) implement your contribution and ideally a test case,
5) test it by calling ``make tests``, ``make lints`` and ``make format`` on the (Unix) command line,
6) submit a PR 🙂

License
-------

:code:`sbijax` is licensed under the Apache 2.0 License.

..  toctree::
    :maxdepth: 1
    :hidden:

    🏡 Home <self>
    🔀 Migration guide <migration>
    📚 References <references>

..  toctree::
    :caption: 🎓 Tutorials
    :maxdepth: 1
    :hidden:

    Getting started <notebooks/getting_started>
    A more detailed intro  <notebooks/more_detailed_intro>
    Examples <notebooks/examples>
    Inference using EEG data  <notebooks/eeg_data_example>

..  toctree::
    :caption: 🚀 Examples
    :maxdepth: 1
    :hidden:

    Self-contained examples <examples>

..  toctree::
    :caption: 🧱 API
    :maxdepth: 3
    :hidden:

    sbijax
    sbijax.experimental
    sbijax.mcmc
    sbijax.nn
    sbijax.simulators
    sbijax.util
