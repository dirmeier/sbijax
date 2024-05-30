from collections import namedtuple

import arviz as az


def mcmc_diagnostics(samples: az.InferenceData):
    MCMCDiagnostics = namedtuple("MCMCDiagnostics", "rhat ess")
    return MCMCDiagnostics(az.rhat(samples), az.ess(samples))
