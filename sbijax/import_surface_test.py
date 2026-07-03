import sbijax


def test_no_plotting_symbols_exported():
  assert not any(name.startswith("plot_") for name in sbijax.__all__)
  assert not hasattr(sbijax, "plot_posterior")
