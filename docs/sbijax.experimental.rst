``sbijax.experimental``
=======================

.. currentmodule:: sbijax.experimental

``sbijax.experimental`` contains experimental code that might get ported to the
main code base or possibly deleted again.

``npse`` and ``aio`` are functional factories that delegate to the ``fmpe``
core; ``make_truncated_proposal`` builds the truncated-prior proposal they use
with :func:`sbijax.run_sequential`.

.. autosummary::
    npse
    aio
    make_truncated_proposal

.. autofunction:: npse

.. autofunction:: aio

.. autofunction:: make_truncated_proposal

.. currentmodule:: sbijax.experimental.nn

.. autosummary::
    make_score_model
    make_simformer_based_score_model
    ScoreModel

.. autofunction:: make_simformer_based_score_model

.. autofunction:: make_score_model

..  autoclass:: ScoreModel
    :members: __call__
