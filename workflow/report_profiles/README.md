# Uniform linear-wave report profile

`linear_wave_unrefined.yaml` selects only the uniform-grid linear MHD wave convergence
study. It can be layered over the main configuration when generating a focused report:

```bash
snakemake -s workflow/Snakefile \
  --configfile workflow/config.yaml workflow/report_profiles/linear_wave_unrefined.yaml \
  --report linear-wave-report.zip
```

The fourth-order workflow intentionally contains no AMR or static-refinement profiles.
