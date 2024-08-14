``sbijax``
==========

.. currentmodule:: sbijax

The top-level module, ``sbijax``, contains all implemented methods for neural
simulation-based inference and approximate Bayesian inference as well as
functionality for visualization and other utility.

.. autosummary::
    CMPE
    FMPE
    NPE
    NLE
    SNLE
    NRE
    SMCABC
    NASS
    NASSS
    plot_ess
    plot_loss_profile
    plot_rank
    plot_rhat_and_ress
    plot_posterior
    plot_trace
    as_inference_data
    inference_data_as_dictionary


Posterior estimation
--------------------

..  autoclass:: CMPE
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior

..  autoclass:: FMPE
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior

..  autoclass:: NPE
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior

Likelihood estimation
---------------------

..  autoclass:: NLE
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior

..  autoclass:: SNLE
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior

Likelihood-ratio estimation
---------------------------

..  autoclass:: NRE
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior

Approximate Bayesian computation
--------------------------------

..  autoclass:: SMCABC
    :members: sample_posterior

Summary statistics
------------------

..  autoclass:: NASS
    :members: fit, summarize

..  autoclass:: NASSS
    :members: fit, summarize

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
