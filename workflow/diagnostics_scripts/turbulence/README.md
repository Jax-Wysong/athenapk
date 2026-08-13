# Turbulence diagnostics

This directory contains the turbulence diagnostics used by the Snakemake workflow.

`energy_transfer_analysis/` is a frozen copy of the analysis implementation from
`pgrete/energy-transfer-analysis`, branch `back-to-mpi4py-fft`. The upstream Python
files are kept unchanged except for the compatibility adjustment documented below,
so AthenaPK snapshots are processed with the same definitions used by the reference
analysis. `UPSTREAM_COMMIT` records the source revision and the included license
records the upstream licensing terms.

`wendeln_reference/` preserves the flow-analysis and transfer-analysis SLURM scripts
provided by Carolyn Wendeln. The notebook was received with its JSON content in a file
named `scales.py`; that content is preserved as `numerical_dissipation.ipynb` so that it
can be opened as the intended notebook. The site-specific SLURM scripts are reference
artifacts and are not executed directly by Snakemake.

The workflow calls the upstream `run_analysis.py` directly for each selected PHDF
snapshot. `plot_power_spectra.py` is the only new spectral-analysis script: it extracts
the final power-spectrum plotting procedure from the supplied notebook and accepts
workflow paths and labels on the command line.

The analysis Python environment must provide the dependencies imported by the
upstream package, including `mpi4py`, `mpi4py-fft`, `numpy`, `scipy`, and `h5py`.
Set `tests.turbulence.spectra.python` in the workflow configuration to that Python
executable. The plotting Python environment does not need `mpi4py-fft`.

One compatibility-only change is carried in the frozen upstream sources:
`FFTHelperFuncs.py` converts the tuple returned by NumPy 2's `meshgrid()` into a list
before rescaling its entries. Older NumPy versions returned the mutable container
expected by the original script; the numerical operations are otherwise unchanged.
In addition, the AthenaPK acceleration-field names in `IOhelperFuncs.py` use the
current Parthenon labels `acc_0`, `acc_1`, and `acc_2`; the reference script expected
the older labels `acc_Acceleration1`, `acc_Acceleration2`, and
`acc_Acceleration3`.
