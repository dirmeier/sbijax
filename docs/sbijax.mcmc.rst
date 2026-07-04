``sbijax.mcmc``
===============

.. currentmodule:: sbijax.mcmc

``sbijax.mcmc`` builds the posterior samplers
and exposes the low-level MCMC routines they are built on.

Posterior samplers
------------------

:func:`make_sampler` bundles an MCMC kernel handle, the prior, and ``N(0, I)``
chain initialisation into a sampler that is passed to :func:`sbijax.sample`. The
available kernel handles are ``nuts``, ``mala``, ``rmh`` and ``imh``::

    from sbijax.mcmc import make_sampler, nuts

    sampler = make_sampler(nuts, prior=prior)
    samples, info = sample(key, estimator, params, y_obs, sampler=sampler)

.. autosummary::
    make_sampler

.. autofunction:: make_sampler

Low-level kernels
-----------------

The kernel-specific routines below sample a user-supplied log-density directly.

.. autosummary::
    sample_with_imh
    sample_with_mala
    sample_with_nuts
    sample_with_rmh
    sample_with_slice

.. autofunction:: sample_with_imh

.. autofunction:: sample_with_mala

.. autofunction:: sample_with_nuts

.. autofunction:: sample_with_rmh

.. autofunction:: sample_with_slice
