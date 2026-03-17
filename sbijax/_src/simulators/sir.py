import jax
from diffrax import ODETerm, SaveAt, Tsit5, diffeqsolve
from jax import numpy as jnp
from tensorflow_probability.substrates.jax import distributions as tfd


# ruff: noqa: PLR0913, E501
def sir_model(
    population_size=1_000_000,
    binomial_count=1_000,
    initial_conditions=(1_000_000 - 1, 1, 0),
    t_end=160,
    summarize_data=False,
):
    """SIR model.

    Construct prior, simulator, and likelihood functions.

    Args:
      population_size: the size of the population for the SIR model
      binomial_count: the number of Bernoulli trials for the Binomial likelihood
      initial_conditions: tuple of three integers that should sum to
        population_size
      t_end: end time of the ODE
      summarize_data: if true returns the data from the simulator in a summarized
        version of 5 values. Otherwise returns the infection counts of the ODE.

    Returns:
      returns a tuple of three objects. The first is a
      tfd.JointDistributionNamed serving as a prior distribution. The second
      is a simulator function that can be used to generate data. The third
      is the likelihood function.

    References:
      Lueckmann, Jan-Matthis, et al., "Benchmarking Simulation-Based Inference", 2021.
    """
    dt = 0.1
    ts = jnp.linspace(0, t_end, t_end)

    def prior_fn():
        prior = tfd.JointDistributionNamed(
            dict(
                beta=tfd.LogNormal(jnp.log(jnp.array([0.4])), 0.5),
                gamma=tfd.LogNormal(jnp.log(jnp.array([0.125])), 0.2),
            )
        )
        return prior

    def _solve_ode(theta):
        def f(t, y, args):
            susceptible, infected, _ = y
            beta, gamma = jnp.squeeze(args["beta"]), jnp.squeeze(args["gamma"])
            d_s = -beta * susceptible * infected / population_size
            d_i = (
                beta * susceptible * infected / population_size
                - gamma * infected
            )
            d_r = gamma * infected
            d_y = d_s, d_i, d_r
            return d_y

        term = ODETerm(f)
        solver = Tsit5()
        saveat = SaveAt(ts=ts)

        def _solve(args):
            sol = diffeqsolve(
                term,
                solver,
                0,
                t_end,
                dt,
                initial_conditions,
                args=args,
                saveat=saveat,
            )
            return sol.ys[1]

        infections = jax.vmap(_solve)(theta)
        return infections

    def summarize(ys):
        I_max = jnp.max(ys)
        t_peak = ts[jnp.argmax(ys)]
        R_final = jnp.sum(ys)
        mean_I = jnp.mean(ys)
        var_I = jnp.var(ys)

        return jnp.array([I_max, t_peak, R_final, mean_I, var_I])

    def simulator(seed, theta):
        infections = _solve_ode(theta)
        probs = jnp.clip(infections / population_size, 0, 1)
        ys = tfd.Binomial(total_count=binomial_count, probs=probs).sample(
            seed=seed
        )
        if summarize_data:
            return jax.vmap(summarize)(ys)
        return ys

    def likelihood(y, theta):
        _, infections = _solve_ode(theta)
        probs = jnp.clip(infections / population_size, 0, 1)
        lik_fn = tfd.Independent(
            tfd.Binomial(total_count=binomial_count, probs=probs),
            reinterpreted_batch_ndims=1,
        )
        log_lik = lik_fn.log_prob(y)
        return log_lik

    return prior_fn(), simulator, likelihood
