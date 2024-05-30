:github_url: https://github.com/dirmeier/sbijax

👋 Welcome to :code:`sbijax`!
=============================

.. div:: sd-text-left sd-font-italic

    Simulation-based inference in JAX

----

:code:`sbijax` implements several algorithms for simulation-based inference in
`JAX <https://github.com/google/jax>`_ using `Haiku <https://github.com/deepmind/dm-haiku>`_,
`Distrax <https://github.com/deepmind/distrax>`_ and `BlackJAX <https://github.com/blackjax-devs/blackjax>`_. Specifically, :code:`sbijax` implements

- `Sequential Monte Carlo ABC <https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9780367733728>`_ (:code:`SMCABC`)
- `Neural Likelihood Estimation <https://arxiv.org/abs/1805.07226>`_  (:code:`SNL`)
- `Surjective Neural Likelihood Estimation <https://arxiv.org/abs/2308.01054>`_ (:code:`SSNL`)
- `Neural Posterior Estimation C <https://arxiv.org/abs/1905.07488>`_ (short :code:`SNP`)
- `Contrastive Neural Ratio Estimation <https://arxiv.org/abs/2210.06170>`_ (short :code:`SNR`)
- `Neural Approximate Sufficient Statistics <https://arxiv.org/abs/2010.10079>`_ (:code:`SNASS`)
- `Neural Approximate Slice Sufficient Statistics <https://openreview.net/forum?id=jjzJ768iV1>`_ (:code:`SNASSS`)
- `Flow matching posterior estimation <https://arxiv.org/abs/2305.17161>`_ (:code:`SFMPE`)
- `Consistency model posterior estimation <https://arxiv.org/abs/2312.05440>`_ (:code:`SCMPE`)

.. caution::

    ⚠️ As per the LICENSE file, there is no warranty whatsoever for this free software tool. If you discover bugs, please report them.

Example
-------

:code:`sbijax` uses an object-oriented API with functional elements stemming from JAX. You can, for instance, define
a neural likelihood estimation method and generate posterior samples like this:

.. code-block:: python

    import distrax
    import optax
    from jax import numpy as jnp, random as jr
    from sbijax import SNL
    from sbijax.nn import make_affine_maf

    def prior_model_fns():
        p = distrax.Independent(distrax.Normal(jnp.zeros(2), jnp.ones(2)), 1)
        return p.sample, p.log_prob

    def simulator_fn(seed, theta):
        p = distrax.Normal(jnp.zeros_like(theta), 1.0)
        y = theta + p.sample(seed=seed)
        return y

    prior_simulator_fn, prior_logdensity_fn = prior_model_fns()
    fns = (prior_simulator_fn, prior_logdensity_fn), simulator_fn
    model = SNL(fns, make_affine_maf(2))

    y_observed = jnp.array([-1.0, 1.0])
    data, _ = model.simulate_data(jr.PRNGKey(0), n_simulations=5)
    params, _ = model.fit(jr.PRNGKey(1), data=data, optimizer=optax.adam(0.001))
    posterior, _ = model.sample_posterior(jr.PRNGKey(2), params, y_observed)

Installation
------------

To install from PyPI, call:

.. code-block:: bash

    pip install sbijax

To install the latest GitHub <RELEASE>, just call the following on the
command line:

.. code-block:: bash

    pip install git+https://github.com/dirmeier/sbijax@<RELEASE>

See also the installation instructions for `JAX <https://github.com/google/jax>`_, if
you plan to use :code:`sbijax` on GPU/TPU.

Contributing
------------

Contributions in the form of pull requests are more than welcome. A good way to start is to check out issues labelled
`"good first issue" <https://github.com/dirmeier/sbijax/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22>`_.

In order to contribute:

1) Clone :code:`sbijax` and install :code:`hatch` via :code:`pip install hatch`,
2) create a new branch locally :code:`git checkout -b feature/my-new-feature` or :code:`git checkout -b issue/fixes-bug`,
3) implement your contribution and ideally a test case,
4) test it by calling :code:`hatch run test` on the (Unix) command line,
5) submit a PR 🙂

Citing sbijax
-------------

.. code-block:: latex

    @article{dirmeier2024sbijax,
        author = {Simon Dirmeier and Antonietta Mira and Carlo Albert},
        title = {Simulation-based inference with the Python Package sbijax},
        year = {2024},
    }

Acknowledgements
----------------

.. note::

    📝 The API of the package is heavily inspired by the excellent Pytorch-based `sbi <https://github.com/sbi-dev/sbi>`_ package.

License
-------

:code:`sbijax` is licensed under the Apache 2.0 License.

..  toctree::
    :maxdepth: 1
    :hidden:

    🏠 Home <self>
    📰 News <news>

..  toctree::
    :caption: 🎓 Examples
    :maxdepth: 1
    :hidden:

    Introduction <notebooks/introduction>
    Self-contained scripts <examples>

..  toctree::
    :caption: 🧱 API
    :maxdepth: 2
    :hidden:

    sbijax
    sbijax.mcmc
    sbijax.nn
    sbijax.util
