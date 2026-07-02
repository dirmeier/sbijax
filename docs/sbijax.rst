``sbijax``
==========

.. currentmodule:: sbijax

The top-level module, ``sbijax``, contains all implemented methods for neural
simulation-based inference and approximate Bayesian inference as well as
functionality for visualization and other utility.

Every method is a **factory function** returning a record of pure functions
(:class:`Estimator`, :class:`ABCSampler` or :class:`SummaryNet`), following the
functional idiom of dm-haiku and blackjax. Parameters are threaded explicitly::

    est = nle(prior, make_maf(2))
    params, info = est.fit(key, data)
    idata = est.sample(key, params, y_observed)

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
    run_sequential
    simulate
    stack
    sbc
    Estimator
    ABCSampler
    SummaryNet
    as_inference_data
    inference_data_as_dictionary

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

Visualization
-------------

.. autofunction:: plot_ess
.. autofunction:: plot_loss_profile
.. autofunction:: plot_rank
.. autofunction:: plot_rhat_and_ress
.. autofunction:: plot_posterior
.. autofunction:: plot_trace

Utility
-------

.. autofunction:: as_inference_data
.. autofunction:: inference_data_as_dictionary
