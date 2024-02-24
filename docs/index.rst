:github_url: https://github.com/dirmeier/sbijax

👋 Welcome to :code:`sbijax`!
=============================

.. div:: sd-text-left sd-font-italic

    Simulation-based inference in JAX

----

:code:`sbijax` implements several algorithms for simulation-based inference in
`JAX <https://github.com/google/jax>`_ using `Haiku <https://github.com/deepmind/dm-haiku>`_,
`Distrax <https://github.com/deepmind/distrax>`_ and `BlackJAX <https://github.com/blackjax-devs/blackjax>`_. Specifically, :code:`sbijax` implements

- `Sequential Monte Carlo ABC <https://www.routledge.com/Handbook-of-Approximate-Bayesian-Computation/Sisson-Fan-Beaumont/p/book/9780367733728>`_ (:code:`SMCABC`),
- `Neural Likelihood Estimation <https://arxiv.org/abs/1805.07226>`_  (:code:`SNL`)
- `Surjective Neural Likelihood Estimation <https://arxiv.org/abs/2308.01054>`_ (:code:`SSNL`)
- `Neural Posterior Estimation C <https://arxiv.org/abs/1905.07488>`_ (short :code:`SNP`)
- `Contrastive Neural Ratio Estimation <https://arxiv.org/abs/2210.06170>`_ (short :code:`SNR`)
- `Neural Approximate Sufficient Statistics <https://arxiv.org/abs/2010.10079>`_ (:code:`SNASS`)
- `Neural Approximate Slice Sufficient Statistics <https://openreview.net/forum?id=jjzJ768iV1>`_ (:code:`SNASSS`)

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

Acknowledgements
----------------

.. note::

    📝 The package draws significant inspiration from the excellent Pytorch-based `sbi <https://github.com/sbi-dev/sbi>`_ package which is
    substantially more feature-complete and user-friendly.

License
-------

:code:`sbijax` is licensed under the Apache 2.0 License.

..  toctree::
    :maxdepth: 1
    :hidden:

    🏠 Home <self>

..  toctree::
    :caption: 🎓 Examples
    :maxdepth: 1
    :hidden:

    Self-contained scripts <examples>

..  toctree::
    :caption: 🧱 API
    :maxdepth: 1
    :hidden:

    sbijax
    sbijax.nn
