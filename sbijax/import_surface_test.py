import sbijax


def test_no_plotting_symbols_exported():
  assert not any(name.startswith("plot_") for name in sbijax.__all__)
  assert not hasattr(sbijax, "plot_posterior")


def test_slim_surface_exports():
  import sbijax

  for name in ["ess", "rhat", "MCMCSampleInfo", "DirectSampleInfo", "sbc"]:
    assert name in sbijax.__all__ and hasattr(sbijax, name)
  for gone in ["as_inference_data", "inference_data_as_dictionary"]:
    assert gone not in sbijax.__all__
