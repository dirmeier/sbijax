sbijax.experimental
===================

.. currentmodule:: sbijax.experimental

``sbijax.experimental`` contains experimental code that might get ported to the
main code base or possibly deleted again.

``cmpe`` (consistency-model posterior estimation) and ``aio`` are functional
factories; ``aio`` delegates to the ``fmpe`` core, and
``make_truncated_proposal`` builds the truncated-prior proposal used with
:func:`sbijax.run_sequential`. The score networks below are consumed by
:func:`sbijax.npse`, which now lives in the main package.

.. autosummary::
    cmpe
    aio
    make_truncated_proposal

.. autofunction:: cmpe

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
