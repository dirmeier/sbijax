"""Guards for the slim public surface (no plotting, no arviz helpers)."""

import sbijax


def test_no_plotting_symbols_exported():
  assert not any(name.startswith("plot_") for name in sbijax.__all__)
  assert not hasattr(sbijax, "plot_posterior")


def test_slim_surface_exports():
  for name in ["ess", "rhat", "sbc"]:
    assert name in sbijax.__all__ and hasattr(sbijax, name)
  # return-only records (constructed by the library, never by users) and the
  # dropped arviz helpers must not be part of the public surface.
  for gone in [
    "as_inference_data",
    "inference_data_as_dictionary",
    "MCMCSampleInfo",
    "DirectSampleInfo",
  ]:
    assert gone not in sbijax.__all__
