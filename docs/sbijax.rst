``sbijax``
==========

.. currentmodule:: sbijax

The top-level module, ``sbijax``, contains all implemented methods for neural
simulation-based inference and approximate Bayesian inference as well as
diagnostics and other utility.

Every method is a **factory function** returning a record of pure functions
(:class:`Estimator`, :class:`ABCSampler` or :class:`SummaryNet`), following the
functional idiom of dm-haiku and blackjax. Parameters are threaded explicitly::

    est = nle(prior, make_maf(2))
    params, info = est.fit(key, data)
    samples, info = est.sample(key, params, y_observed)

.. autosummary::
    npe
    fmpe
    cmpe
    nle
    snle
    nre
    sabc
    smcabc
    nass
    nasss
    summarized_estimator
    run_sequential
    simulate
    stack
    sbc
    ess
    rhat
    Estimator
    ABCSampler
    SummaryNet
    MCMCSampleInfo
    DirectSampleInfo

Data pipeline
-------------

.. autofunction:: simulate
.. autofunction:: stack

Posterior estimation
--------------------

.. autofunction:: npe
.. autofunction:: fmpe
.. autofunction:: cmpe

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

Sequential inference
--------------------

.. autofunction:: run_sequential

Interfaces
----------

..  autoclass:: Estimator
    :members: fit, sample

..  autoclass:: ABCSampler
    :members: sample

..  autoclass:: SummaryNet
    :members: fit, summarize

Diagnostics
-----------

.. autofunction:: sbc
.. autofunction:: ess
.. autofunction:: rhat
.. autoclass:: MCMCSampleInfo
.. autoclass:: DirectSampleInfo
