I would implement this as a **separate CT correction path**, leaving native Euler/GLM FOFC unchanged. Start with uniform-grid, source-free `ctmhd`, then add communication and refinement support after the local algorithm works.

Do this on an experimental branch rather than adding an unfinished correction mechanism to PR #195.

## 1. Define the initial supported configuration

Keep the first version deliberately narrow:

- `fluid=ctmhd`.
- Normal flux calculation remains PLM/HLLD.
- RK1/RK2/RK3 only; start testing with RK2.
- Uniform grid initially.
- No resistivity or other diffusion.
- No claim that FOFC protects subsequent source-term updates.
- Continue rejecting FOFC for `ucthlldmhd`.

Initially test with pressure/density floors disabled so they cannot hide failures.

Do **not** remove the existing CT rejection until the new task sequence is connected. Eventually, permit `ctmhd` conditionally and dispatch it separately from the native correction function.

Configuration currently lives in [`Hydro::Initialize()`](/mnt/ffs24/home/wysongj2/athenapk/src/hydro/hydro.cpp:604).

## 2. Extract reusable, non-mutating CT update arithmetic

Start from [`UpdateWithFluxEmfCurl()`](/mnt/ffs24/home/wysongj2/athenapk/src/hydro/ctmhd/ctmhd.cpp:345).

Create small device-inline helpers that return the trial value of one face component:

\[
B_{f,*}
=\gamma_0B_{f,0}+\gamma_1B_{f,1}
-\beta\Delta t\,(\nabla_h\times E)_f.
\]

Important requirements:

- Reproduce the existing signs, indexing, and RK coefficients exactly.
- Return a value without writing to live `Bface`.
- Read only active EMF components.
- In 1D, the normal field gets the RK register combination with zero curl.
- In 2D, only F1/F2 are staggered evolution components.

Eventually, use the same helpers in the production update and trial calculation so the formulas cannot drift apart. Verify that refactoring the production update alone leaves results unchanged.

**Checkpoint:** the helpers reproduce the existing face update before adding any correction logic.

## 3. Implement a trial-state positivity detector

Create a new function, for example:

```cpp
CalculateCTTrialStateAndFlags(...)
```

Names here are suggestions, not existing APIs.

For each interior cell:

1. Predict the FV update of the conserved variables using `FluxDivHelper()`.
2. Predict the bounding staggered magnetic faces using your new CT helpers.
3. Center the active trial face components.
4. Retain the FV-predicted magnetic components in inactive dimensions.
5. Evaluate density and thermal energy using the resulting CT-consistent state.

With energy correction disabled:

\[
e_{\rm th,*}
=
E_{\rm FV,*}
-\frac{|\mathbf m_*|^2}{2\rho_*}
-\frac{|\mathbf B_{\rm CT,*}|^2}{2}.
\]

With energy correction enabled, first emulate [`center_Mag_Field()`](/mnt/ffs24/home/wysongj2/athenapk/src/hydro/ctmhd/ctmhd.cpp:462):

\[
E_*
=
E_{\rm FV,*}
+\frac{|\mathbf B_{\rm CT,*}|^2-|\mathbf B_{\rm FV,*}|^2}{2}.
\]

Then use \(E_*\) and \(\mathbf B_{\rm CT,*}\) in the pressure calculation.

Check explicitly for:

- Nonfinite candidate values.
- Nonpositive density, **before dividing by density**.
- Nonpositive thermal energy.

Do not apply floors or modify the physical state here.

**Checkpoint:** with correction still disabled, the predicted state agrees with the actual source-free stage update, including both energy-correction settings.

## 4. Add the CT-compatible DC/LLF face solver

Add a dedicated header such as:

```text
src/hydro/rsolvers/ctmhd_dc_llf.hpp
```

Use these as references:

- [`glmmhd_dc_llf.hpp`](/mnt/ffs24/home/wysongj2/athenapk/src/hydro/rsolvers/glmmhd_dc_llf.hpp:46)
- [`ctmhd_hlld.hpp`](/mnt/ffs24/home/wysongj2/athenapk/src/hydro/rsolvers/ctmhd_hlld.hpp:41)

The fallback should:

1. Read adjacent cell-centered primitives directly—no PLM reconstruction.
2. Replace the normal magnetic component in both states with the same current `Bface` value.
3. Calculate LLF fluxes with the appropriate magnetosonic signal-speed estimate.
4. Omit the GLM `psi` subsystem entirely.
5. Handle passive scalars consistently.
6. Support all three face orientations.

Keep this an **internal fallback solver**. There is no need to permit LLF as a normal user-selected CT solver yet.

**Checkpoint:** identical left/right states produce the physical flux; cyclic coordinate permutations agree; both states always use one normal magnetic field.

## 5. Provide separate fallback storage

Avoid writing fallback candidates into the normal flux arrays before deciding where they are needed.

You need storage conceptually equivalent to:

| Quantity | Location | Purpose |
|---|---|---|
| Troubled-cell flag | Cell | Monotonic correction selection during a stage |
| Low-order fluxes | Face | DC/LLF candidates |
| Low-order EMFs | Edge | CT-Contact candidates built from low-order fluxes |

Trial conserved states and trial face values can remain local scratch values; a full saved trial solution is not essential.

Keep workspace derived/non-evolved. In particular, **do not register a cell-centered workspace with `WithFluxes`**: Parthenon’s FV updater selects cell variables carrying that flag, regardless of whether you intended them as scratch.

For the eventual parallel implementation, flags need communication. Candidate storage must cover the stencil required by the edge assembler, including boundary-adjacent faces.

## 6. Generate low-order CT-Contact EMFs

Refactor or overload [`Assemble_Corner_EMF()`](/mnt/ffs24/home/wysongj2/athenapk/src/hydro/ctmhd/ctmhd.cpp:20) so it can operate on a specified flux input and EMF output.

Preserve the existing normal call unchanged.

The fallback call should use:

- Low-order magnetic face fluxes.
- Low-order mass fluxes for the upwind-direction decisions.
- The unchanged current-stage primitives for the cell-centered electric fields.
- Separate low-order edge-EMF output storage.

Compute these low-order EMFs from a **complete low-order face-flux stencil**, not a partially overwritten mixture of normal and fallback inputs.

This defines your candidate fallback as **DC/LLF faces + CT-Contact edges**. Treat its positivity properties as something to establish experimentally, not something guaranteed merely by choosing LLF.

**Checkpoint:** all fallback fluxes and EMFs can be generated without modifying normal fluxes, live conserved variables, or live face fields.

## 7. Implement deterministic face and edge selection

Do not reproduce the native kernel’s concurrent “check a cell and immediately overwrite its faces” pattern.

Instead, use separate passes:

1. Detect troubled cells.
2. Derive face selections.
3. Derive edge selections.
4. Replace selected fluxes/EMFs.
5. Recheck the coupled trial update.

A practical initial rule is:

- Select a face if either adjacent cell is troubled.
- Select an edge if any face in that edge assembler’s input stencil is selected.

For a selected edge, use the precomputed **fully low-order edge candidate**. For an unselected edge, retain the original EMF.

Assign one thread to each face or edge write. Multiple cells should not independently compete to write it.

Keep selections **monotonic within a stage**: once downgraded, do not upgrade that face or edge during later correction passes.

**Checkpoint:** a forced troubled cell produces the expected face/edge footprint, and neighboring cells see the same shared values.

## 8. Connect a single-block correction loop

Add the new sequence in [`hydro_driver.cpp`](/mnt/ffs24/home/wysongj2/athenapk/src/hydro/hydro_driver.cpp:531), after normal flux and EMF assembly but before actual updates.

For the first uniform, single-block version:

```text
Normal PLM/HLLD fluxes
        ↓
Normal CT-Contact EMFs
        ↓
Trial coupled update and positivity check
        ↓
If bad cells exist:
    generate low-order candidates once
    expand correction selections
    replace selected faces/edges
    recheck
        ↓
Actual FV and CT updates
```

Keep current-stage primitives and starting conserved/face states unchanged throughout the loop. Repeated passes modify only fluxes, EMFs, and correction selections.

For efficiency, generate low-order candidates **only after the first failed trial check**, and reuse them throughout that stage.

Add a configurable iteration cap. If bad cells remain when the cap is reached—or selections stop changing—report failure clearly. Initially, **do not silently return success and rely on a floor**.

## 9. Add diagnostics before broad testing

Useful counters are:

- Initially troubled cells.
- Newly troubled cells per pass.
- Unique corrected faces and edges.
- Correction passes per stage.
- Remaining invalid cells on failure.
- Minimum trial density and pressure.

For failures, include stage, time, rank, block, and cell location.

Distinguish “cells that requested correction” from “unique corrected faces.” They are not the same count.

Do not add a shock sensor: this detector should respond to the predicted update’s physical admissibility.

## 10. Extend to multiple blocks and MPI

Only after the single-block algorithm is working:

1. Exchange troubled-cell flags before deriving boundary face/edge selections.
2. Use logical OR semantics so both representations of a shared interface make the same decision.
3. Ensure fallback candidates have valid transverse stencil data.
4. Ensure shared final face fluxes and edge EMFs agree.
5. Handle duplicate self-periodic endpoints consistently.
6. Use a globally coordinated stopping decision.

All ranks must participate in the same communication sequence, even when a particular rank has no troubled cells.

Use Parthenon’s task/reduction mechanisms for the iteration. Avoid arbitrary blocking MPI collectives inside independently scheduled per-pack tasks.

Keep mask exchanges and normal end-of-stage state exchanges properly ordered; do not assume their communication buffers can overlap safely.

**Checkpoint:** forced correction adjacent to a block interface behaves consistently on one/multiple blocks and one/two ranks.

## 11. Integrate AMR/SMR last

The existing driver performs refinement synchronization through:

```cpp
LoadAndSendFluxCorrections
ReceiveFluxCorrections
SetFluxCorrections
```

For CT correction, the final positivity decision should inspect the fluxes and EMFs **after the relevant synchronization**, because synchronization can change the candidate coarse-cell update.

The eventual iteration becomes:

```text
Select/replace fluxes and EMFs
        ↓
Synchronize shared/coarse–fine data
        ↓
Check the synchronized coupled update
        ↓
Expand selections and repeat if needed
```

Additional requirements:

- Transfer correction selections across coarse/fine interfaces using appropriate logical mappings—not ordinary interpolation of Boolean flags.
- Let established refinement machinery enforce coarse/fine flux and edge-EMF consistency.
- Reinitialize communication correctly on every iteration.
- Recheck cells on both sides of affected interfaces.
- Do not apply an unsynchronized coarse-side replacement after synchronization.

Until this works, explicitly reject CT FOFC on multilevel meshes.

## 12. Validate in increasing difficulty

I would follow this order:

1. **Smooth, no correction:** off/on results agree, corrected counts are zero, convergence is unchanged.
2. **Forced correction:** force one cell, then a patch, then all cells. Check selection footprints, conservation, and face divergence.
3. **Dimensional checks:** 1D transverse evolution, 2D Ez, then all three 3D edge orientations.
4. **Shock problems:** confirm actual positivity-triggered replacement, including cases where the fallback itself fails.
5. **Energy correction:** test off/on separately; only demand strict total-energy conservation when it is off.
6. **Block interfaces and periodic boundaries:** repeat forced and naturally triggered cases across interfaces.
7. **CPU/GPU and MPI:** compare correction counts and solutions within appropriate tolerances.
8. **SMR, then moving AMR:** inspect divergence and coarse/fine consistency during initialization and remeshing.
9. **Existing Euler/GLM regressions:** verify their untouched FOFC paths still behave as before.

Do not hide fallback failures by adjusting floors or thresholds. A failure of the fully low-order coupled update is useful evidence about the fallback, not necessarily a coding mistake.

The first substantial milestone is **single-block 2D CT-consistent trial detection plus coordinated DC/LLF face and CT-Contact edge replacement**. Get that working before tackling parallel iteration and refinement. No files were changed.