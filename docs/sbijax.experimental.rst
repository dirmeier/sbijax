``sbijax.experimental``
=======================

.. currentmodule:: sbijax.experimental

``sbijax.experimental`` contains experimental code that might get ported to the
main code base or possibly deleted again.

.. autosummary::
    AiO
    NPSE

..  autoclass:: AiO
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior

..  autoclass:: NPSE
    :members: fit, simulate_data, simulate_data_and_possibly_append, sample_posterior


.. currentmodule:: sbijax.experimental.nn

.. autosummary::
    make_score_model
    make_simformer_based_score_model
    ScoreModel

.. autofunction:: make_simformer_based_score_model

.. autofunction:: make_score_model

..  autoclass:: ScoreModel
    :members: __call__
