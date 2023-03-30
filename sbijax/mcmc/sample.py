from collections import namedtuple

import blackjax as bj


def mcmc_diagnostics(samples):
    """
    Computes MCMC diagnostics.

    Compute effective sample sizes and R-hat for each parameter of a set of
    MCMC chains.

    Parameters
    ----------
    samples: jnp.ndarray
        a JAX array of dimension n_samples \times n_chains \times n_dim

    Returns
    -------
    tuple
        a tuple of jnp.ndarrays with ess and rhat estimates.
    """

    n_theta = samples.shape[-1]
    esses = [0] * n_theta
    rhats = [0] * n_theta
    for i in range(n_theta):
        posterior = samples[:, :, i].T
        esses[i] = bj.diagnostics.effective_sample_size(posterior)
        rhats[i] = bj.diagnostics.potential_scale_reduction(posterior)
    return namedtuple("diagnostics", "ess rhat")(esses, rhats)
