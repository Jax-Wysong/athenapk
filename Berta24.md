# Berta24 fourth-order FV-CT-MHD branch

This experimental branch implements the Cartesian fourth-order finite-volume ideal-MHD
method based on pointwise reconstruction described by Berta et al. (2024), *A
fourth-order accurate finite volume method for ideal classical and special relativistic
MHD based on pointwise reconstructions*. It also contains fourth-order-compatible MHD
problem generators and a uniform-grid Snakemake validation workflow.

The implementation is intended for method development and comparison. It is not a
general replacement for every AthenaPK MHD path.

## Numerical method

The fourth-order path uses:

- finite-volume cell averages for the evolved hydrodynamic conserved variables;
- face-area averages for the evolved staggered magnetic field;
- fourth-order average-to-point transformations before reconstruction;
- pointwise fifth-order WENO-Z reconstruction (`wenoz_point`);
- the HLLD Riemann solver;
- pointwise UCT-HLLD edge electric fields;
- fourth-order point-to-area and point-to-line transformations for fluxes and EMFs;
- a constrained-transport curl update of the staggered magnetic field; and
- Parthenon's low-storage RK4 integrator in the supplied workflow configuration.

At every Runge--Kutta stage, the main sequence is:

1. Detect troubled cells when a detector is enabled.
2. Recover point-valued cell and face states from the evolved averages.
3. Exchange point-state boundaries and convert point conserved variables to primitives.
4. Reconstruct pointwise face states and solve HLLD Riemann problems.
5. Assemble pointwise UCT-HLLD edge EMFs.
6. Convert pointwise face fluxes and edge EMFs back to area and line averages.
7. Update cell averages by flux divergence and face averages by the CT curl.
8. Exchange the updated averaged state and refill derived primitive variables.

Separate `cons_point`, `consflux_point`, and `Bface_point` storage prevents temporary
point values from overwriting the averages advanced by the finite-volume update.

## Required runtime configuration

A normal fourth-order run uses:

```text
hydro/fluid=ucthlldmhd
hydro/riemann=hlld
hydro/reconstruction=wenoz_point
hydro/convergence_order=4
parthenon/mesh/nghost=3
parthenon/time/integrator=rk4
```

The code currently accepts second-order `ucthlldmhd` with PLM or WENO3 and fourth-order
`ucthlldmhd` with `wenoz_point`. The fourth-order path is restricted to uniform Cartesian
meshes. AthenaPK deliberately rejects mesh refinement when `convergence_order=4`.

## Discontinuity detectors and order reduction

The optional detector is selected with:

```text
hydro/discontinuity_detector=none       # or jameson or hod
hydro/discontinuity_detector_threshold=<positive value>
hydro/discontinuity_detector_epsilon=1.0e-12
hydro/discontinuity_detector_fallback=plm
```

`jameson` implements the pressure-based Jameson sensor. `hod` implements the
higher-order derivative-ratio sensor. A detected cell and its immediate neighbors are
marked in `berta24_troubled`. Reconstruction and the average/point transformations are
locally reduced to the existing second-order/PLM route around marked interfaces and
edges.

This follows Berta et al.'s fallback strategy: troubled regions use PLM reconstruction
and omit the fourth-order average/point and transverse flux/EMF integration corrections,
while retaining the HLLD and UCT-HLLD solvers. The paper does not prescribe a local
switch to first-order LLF/UCT-LLF. This branch likewise has no separate LLF positivity
fallback, so a pressure floor may still be needed for particularly severe tests. When a
detector is enabled, history output contains `maxShockIndicator` and `numTroubled`; PHDF
output can include `berta24_troubled` for visualization.

## Fourth-order-compatible problem generators

The reusable utilities in
`src/pgen/ct_glm_compatible/mhd_pgen_utils.hpp` construct the averaged initial data
expected by a fourth-order finite-volume method:

- smooth point-conserved states are converted to cell-volume averages;
- point vector potentials are converted to edge-line averages before taking a discrete
  curl, producing divergence-free face-area-averaged magnetic fields;
- nonsmooth problems use problem-specific Gaussian quadrature where required; and
- duplicate staggered faces on a single self-periodic block can be reconciled before
  centering the magnetic field.

The branch includes fourth-order initialization support for:

- circularly polarized Alfvén waves (`cpaw`);
- linear MHD waves, including directional propagation checks;
- the smooth MHD vortex;
- two- and three-dimensional field loops;
- Orszag--Tang;
- the current sheet;
- the Berta C1 MHD blast;
- the three-dimensional cloud--shock interaction; and
- driven MHD turbulence with uniform staggered magnetic fields.

The field loop, Orszag--Tang, current sheet, blast, and cloud--shock problems are useful
robustness tests but should not be expected to demonstrate fourth-order convergence
because their data contain discontinuities or nonsmooth features.

## Building

Configure builds in the usual AthenaPK fashion. For example, after an existing host
build has been configured:

```bash
cmake --build build-host -j
```

The workflow defaults refer to these repository-relative executables:

```text
build-host/bin/athenaPK
build-mpi/bin/athenaPK
build-mpi-site-hdf5/bin/athenaPK
```

Override them in a local Snakemake configuration when the build directories differ.

## Uniform-grid validation workflow

The workflow entry point is `workflow/Snakefile`, and the committed defaults are in
`workflow/config.yaml`. All expensive tests are disabled by default. The workflow has no
AMR or static-refinement rules.

Available suites include smooth convergence tests, directional wave checks, CPAW
visualization, field-loop transport and orientations, Orszag--Tang, current sheet,
blast, cloud shock, and configurable turbulence comparisons. The turbulence plotting
rules accept any case names listed under `tests/turbulence/cases`; they are not tied to a
fixed four-case ordering.

Run a parse-only dry run with:

```bash
snakemake -s workflow/Snakefile --configfile workflow/config.yaml -n
```

For actual experiments, create the ignored `workflow/config.local.yaml` overlay rather
than turning the committed defaults into a record of one machine or one production run:

```bash
snakemake -s workflow/Snakefile \
  --configfile workflow/config.yaml workflow/config.local.yaml \
  --executor slurm --jobs 8 --keep-going --latency-wait 60
```

Repository-owned inputs and plotting scripts are resolved relative to the checkout.
Outputs are placed below:

```text
<results_root>/<scheme_id>/<dimension>/...
```

The `scheme_id` namespace is important when using this workflow on multiple experimental
branches. Use a distinct value for every numerical scheme so that results cannot
overwrite one another.

No workflow job should be submitted merely by parsing the default configuration because
every test is disabled. Enable only the suites intended for a particular run.

## Scope and known limitations

- Uniform Cartesian meshes only; fourth-order AMR is intentionally unsupported.
- Ideal adiabatic MHD only for the fourth-order UCT-HLLD route developed here.
- HLLD and pointwise WENO-Z are the implemented fourth-order Riemann/reconstruction
  combination.
- Passive-scalar support remains structurally present but has not been a validation
  target.
- The shock fallback is PLM order reduction, not the complete first-order LLF strategy
  discussed in the paper.
- Parthenon's low-storage RK4 is used for comparisons; it does differ 
  slighlty from Berta24's integrator

## Branch organization

Keep the Berta-specific evolution implementation on
`experiment/fourth-order-Berta24`. The scheme-neutral pgen initialization, input decks,
and uniform-grid workflow are suitable for transfer to `experiment/fourth-order-common`.
New fourth-order method branches should start from that common branch and use a new
`scheme_id`.
