sbijax
======

.. currentmodule:: sbijax

The top-level module, ``sbijax``, contains all implemented methods for neural
simulation-based inference and approximate Bayesian inference as well as
diagnostics and other utility.

Every method is a **factory function** that takes only the network and returns a
record of pure functions, following the low-level functional idiom of dm-haiku
and blackjax. Training and sampling are **free driver functions** (:func:`train`,
:func:`sample`); the optimizer is injected at ``train`` and, for likelihood/ratio
methods, the sampler (which carries the prior) at ``sample``::

    est = nle(make_maf(2))
    params, info = train(key, est, data, optimizer=optax.adam(3e-4))
    samples, info = sample(
        key, est, params, y_observed, sampler=make_sampler(nuts, prior=prior)
    )

See :doc:`/design` for the full design and :doc:`/migration` for moving from the
class-based API.

.. autosummary::
    npe
    fmpe
    npse
    nle
    snle
    nre
    sabc
    smcabc
    nass
    nasss
    summarized_estimator
    train
    sample
    run_sequential
    simulate
    stack
    sbc
    rhat
    ess

Data pipeline
-------------

.. autofunction:: simulate
.. autofunction:: stack

Posterior estimation
--------------------

.. autofunction:: npe
.. autofunction:: fmpe
.. autofunction:: npse

Likelihood estimation
---------------------

.. autofunction:: nle
.. autofunction:: snle

Likelihood-ratio estimation
---------------------------

.. autofunction:: nre

Approximate Bayesian computation
--------------------------------

.. autofunction:: sabc
.. autofunction:: smcabc

Summary statistics
------------------

.. autofunction:: nass
.. autofunction:: nasss

A summary network is chained into a downstream estimator with:

.. autofunction:: summarized_estimator

Training and sampling
---------------------

Trainable objectives are trained and sampled with the two free drivers. The
sampler for likelihood/ratio methods is built with
:func:`sbijax.mcmc.make_sampler`.

.. autofunction:: train
.. autofunction:: sample

Sequential inference
--------------------

.. autofunction:: run_sequential

Diagnostics
-----------

.. autofunction:: sbc

.. autofunction:: rhat

.. autofunction:: ess
