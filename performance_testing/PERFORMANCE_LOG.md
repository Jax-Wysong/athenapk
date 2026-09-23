# Teissier GPU Performance Log

This file tracks GPU performance work on the
`optimization/teissier-gpu-performance` branch. The goal is to reduce the cost
of the fourth-order Teissier paths without changing their numerical results.

## Benchmark convention

- Hardware: one NVIDIA V100S 32 GB GPU
- Problem: 3D linear MHD wave, uniform `128^3` mesh
- Duration: 100 cycles, with the first 20 excluded from timing
- Outputs: disabled
- MPI ranks: one
- Primary layout: eight `64^3` meshblocks
- Script: `run_v100_128_scheme_sweep.sh`

Use the same setup for before/after comparisons. Record the executable commit,
result directory, and any changes to the build or benchmark configuration.

## Baseline — 2026-08-26

Results: `results_v100_128_20260826_112250/summary.csv`

| Scheme | Meshblock | Blocks | Wall time (s) | Zone-cycles/s |
|---|---:|---:|---:|---:|
| UCT-HLLD 2nd, WENO3 | `64^3` | 8 | 5.02 | 3.34e7 |
| Teissier4 HLLD-point | `64^3` | 8 | 37.4 | 4.48e6 |
| Teissier4 LLF-line | `64^3` | 8 | 35.3 | 4.76e6 |
| UCT-HLLD 2nd, WENO3 | `128^3` | 1 | 5.52 | 3.04e7 |
| Teissier4 HLLD-point | `128^3` | 1 | 44.8 | 3.74e6 |
| Teissier4 LLF-line | `128^3` | 1 | 44.6 | 3.77e6 |

For the preferred `64^3` layout, HLLD-point costs about 7.45 times and
LLF-line about 7.03 times as much per cycle as second-order UCT. Accounting
for the five-stage RK4 versus three-stage RK3 integrators, the approximate
cost per RHS evaluation is 4.47 and 4.22 times higher, respectively.

Teissier et al. report an approximately 1.3-times-per-RHS cost for their
fourth-order method and approximately three times for their tenth-order
method. The tenth-order figure is therefore not the appropriate performance
target for the fourth-order implementation studied here.

## Initial profiler findings

The initial Nsight Systems profile was collected for LLF-line with eight
`64^3` blocks in the original Teissier worktree. Approximate fractions of GPU
kernel time were:

| Operation | GPU time |
|---|---:|
| Teissier face AtoP and point Riemann solve | 44.5% |
| Boundary packing/sending | 29.7% |
| Teissier face reconstruction | 18.7% |
| All remaining kernels | 7.1% |

The point-flux kernels use roughly 145–150 registers per thread, suggesting
low occupancy on V100. Kernel-launch overhead and explicit copies were small;
the principal targets are point-flux work, boundary exchanges, and face
reconstruction.

## Optimization order

1. Narrow the final full-state boundary exchange after the separate `Bface`
   exchange, while preserving custom physical boundary behavior.
2. For LLF-line, investigate computing the one required transverse layer of
   point-face data locally so the first point-face exchange can be removed.
   Retain the area-electric-field exchange: WENO5 edge reconstruction needs a
   three-cell area-average halo, whose composite point stencil reaches four
   cells and cannot be formed locally with the current three ghost zones.
3. Profile the point-flux kernel with Nsight Compute for measured occupancy,
   memory traffic, and instruction costs.
4. Reduce point-flux scratch storage and solver-specific lambda captures.
5. Optimize reconstruction only after the larger costs above are addressed.

## Correctness checks after each numerical change

- Small 2D CPAW run and representative convergence comparison
- One-block versus multiblock CPAW comparison
- `maxFaceDivB <= 1e-12`
- Small 3D linear-wave run
- Qualitative field-loop check
- Custom-boundary tests before changing boundary containers generally

## Experiment log

Add one entry per isolated change. Keep generated result directories untracked.

### 2026-08-26 — Narrow final Teissier state exchange

- Commit: uncommitted working-tree change based on `fddb789`
- Change: after the separate `Bface` exchange and fourth-order magnetic
  centering, exchange a shallow `cons` container instead of the complete
  `mu0` container.
- Build/configuration differences: no benchmark configuration changes
- Results directory: `results_v100_128_20260826_152644/`
- Results:

  | Scheme | Meshblock | Baseline (zone-cycles/s) | New (zone-cycles/s) | Change |
  |---|---:|---:|---:|---:|
  | UCT-HLLD 2nd, WENO3 | `64^3` | 3.34e7 | 3.45e7 | +3.3% |
  | Teissier4 HLLD-point | `64^3` | 4.48e6 | 5.02e6 | +12.1% |
  | Teissier4 LLF-line | `64^3` | 4.76e6 | 5.63e6 | +18.3% |
  | UCT-HLLD 2nd, WENO3 | `128^3` | 3.04e7 | 3.04e7 | 0.0% |
  | Teissier4 HLLD-point | `128^3` | 3.74e6 | 4.24e6 | +13.4% |
  | Teissier4 LLF-line | `128^3` | 3.77e6 | 4.15e6 | +10.1% |

- Correctness checks: all six cases reached cycle 100 with configurations,
  final times, and final timesteps matching the baseline. Simulation outputs
  were disabled, so final numerical states have not yet been compared.
- Decision: promising; retain pending numerical correctness checks
- Notes: the unchanged second-order control varied by +3.3% for `64^3`
  blocks and 0.0% for `128^3`, so most of the fourth-order improvement is
  larger than run-to-run variation.

#### Post-change Nsight Systems profile

Profiles: `nsys_reports/nsys_profiles_after_cons_exchange/`

| Operation | LLF-line | HLLD-point |
|---|---:|---:|
| Face AtoP and point Riemann solve | 52.3% (5.802 s) | 60.1% (7.525 s) |
| Face reconstruction | 22.0% (2.440 s) | 19.4% (2.432 s) |
| Boundary SetBounds and Send | 17.3% (1.919 s) | 13.8% (1.722 s) |
| All remaining kernels | 8.4% | 6.7% |

For LLF-line, boundary kernel time decreased from approximately 3.873 s in
the original profile to 1.919 s, while point-flux and reconstruction times
were effectively unchanged. Total profiled kernel time fell from about
13.05 s to 11.10 s. This localizes the speedup to the intended exchange
change. The next primary target is the face AtoP and point-Riemann kernel.

The HLLD point-flux kernels are approximately 30% more expensive than their
LLF counterparts and use 192--218 registers per thread versus 145--150 for
LLF, making solver-specific register and scratch use important follow-up
targets.

### 2026-08-26 — Reuse one point-state scratch array

- Commit: uncommitted experiment layered on the shallow-`cons` exchange
- Change: replace the separate `pointL` and `pointR` scratch arrays in all
  three `CalculatePointFluxesTeissier()` directional kernels with one
  sequentially reused `point` array. Scratch capacity decreased from four
  arrays to three, while `wl` and `wr` remained separate.
- Results directory: `results_v100_128_20260826_184944/`
- Correctness: fourth-order CPAW convergence passed.
- Sweep results relative to the shallow-`cons` baseline:

  | Scheme | Meshblock | Before (zone-cycles/s) | Experiment | Change |
  |---|---:|---:|---:|---:|
  | UCT-HLLD 2nd, WENO3 | `64^3` | 3.45e7 | 3.40e7 | -1.4% |
  | Teissier4 HLLD-point | `64^3` | 5.02e6 | 4.96e6 | -1.2% |
  | Teissier4 LLF-line | `64^3` | 5.63e6 | 5.61e6 | -0.4% |
  | UCT-HLLD 2nd, WENO3 | `128^3` | 3.04e7 | 3.04e7 | 0.0% |
  | Teissier4 HLLD-point | `128^3` | 4.24e6 | 4.18e6 | -1.4% |
  | Teissier4 LLF-line | `128^3` | 4.15e6 | 4.10e6 | -1.2% |

- Nsight profiles:
  `nsys_reports/nsys_profiles_after_single_point_scratch/`
- Point-flux kernel time changed from 5.802 to 5.920 seconds for LLF-line
  (+2.0%) and from the recorded 7.525 to 7.856 seconds for HLLD-point
  (+4.4%). The original HLLD report file was accidentally overwritten, but
  its aggregate values had already been recorded above. Two new HLLD profiles
  agreed closely with one another.
- Register use did not improve for LLF. HLLD register use decreased to
  approximately 188--202 registers per thread, but not enough to cross a V100
  occupancy threshold. Extra sequential barriers outweighed the smaller
  scratch footprint.
- Decision: reverted. `teissier_fluxes.cpp` was restored to the original
  four-scratch-array implementation; the shallow-`cons` optimization remains.
- Current post-revert RHS cost at the preferred `64^3` layout is approximately
  4.12 times second order for HLLD-point and 3.68 times for LLF-line, still
  substantially above the paper's approximately 1.3-times fourth-order
  result.

### 2026-08-26 — Replace the LLF point-face exchange with local halo work

- Commit: uncommitted experiment layered on the retained shallow-`cons`
  exchange, based on `fddb789`
- Change: for LLF-line only, reconstruct the additional transverse
  face-average layers and calculate the required point flux, LLF speed, and
  point electric-field halo locally. Remove the boundary exchange of
  `consflux_point`, `teissier_llf_speed`, and `teissier_eface_point`. Retain
  the later `teissier_eface_avg` exchange because edge reconstruction needs
  its wider area-average stencil. HLLD-point and second-order paths are
  unchanged.
- Results directory: `results_v100_128_20260826_200737/`
- Correctness: multiblock CPAW retained fourth-order convergence. Results are
  in `convergence_testing/results_v100_cpaw_20260826_201855/`; the LLF-line
  RMS orders were 4.8023 for `16->32` and 4.8331 for `32->64`.
- Sweep results relative to the shallow-`cons` baseline:

  | Scheme | Meshblock | Before (zone-cycles/s) | New | Change |
  |---|---:|---:|---:|---:|
  | UCT-HLLD 2nd, WENO3 | `64^3` | 3.45e7 | 3.45e7 | 0.0% |
  | Teissier4 HLLD-point | `64^3` | 5.02e6 | 5.04e6 | +0.4% |
  | Teissier4 LLF-line | `64^3` | 5.63e6 | 6.12e6 | +8.7% |
  | UCT-HLLD 2nd, WENO3 | `128^3` | 3.04e7 | 3.03e7 | -0.3% |
  | Teissier4 HLLD-point | `128^3` | 4.24e6 | 4.24e6 | 0.0% |
  | Teissier4 LLF-line | `128^3` | 4.15e6 | 4.75e6 | +14.5% |

  LLF-line wall time decreased from 29.8 to 27.4 seconds for `64^3`
  meshblocks and from 40.4 to 35.3 seconds for one `128^3` meshblock.
  Relative to the original baseline, LLF-line throughput is now 28.6% higher
  for `64^3` blocks and 26.0% higher for the single `128^3` block. At the
  preferred `64^3` layout, its approximate cost per RHS evaluation decreased
  from 4.22 times the second-order path originally, and 3.68 times after the
  shallow-`cons` change, to 3.38 times.
- Nsight profile:
  `nsys_reports/nsys_profiles_after_boundary_exchange_update/`

  | Operation | Before | New | Change |
  |---|---:|---:|---:|
  | Point AtoP and LLF solve | 5.802 s | 6.027 s | +3.9% |
  | Face reconstruction | 2.440 s | 2.561 s | +5.0% |
  | Boundary SetBounds and Send | 1.919 s | 0.728 s | -62.1% |
  | Total GPU kernel time | 11.100 s | 10.255 s | -7.6% |

  Boundary `SetBounds` and `Send` kernels each decreased from 602 to 452
  launches. The 150-launch reduction is exactly one removed exchange per
  measured RK stage. Total kernel launches decreased from 5573 to 5115.
  Profile wall time decreased from 7.41 to 6.85 seconds and throughput
  increased from 5.66e6 to 6.13e6 zone-cycles/s.
- Current LLF-line kernel-time composition: point AtoP and LLF solve 58.8%,
  face reconstruction 25.0%, boundary work 7.1%, and all remaining kernels
  9.1%. The remaining exchanges serve area electric fields, post-update
  `Bface`, and post-centering `cons`.
- Decision: retain. The intended communication cost fell substantially, the
  control paths remained stable, and fourth-order convergence passed. The
  point AtoP/LLF kernels are now the dominant target and should be examined
  with Nsight Compute before another source-code experiment.

### 2026-08-27 — Point-flux barrier and LLF electric-field fusion experiment

- Change: enforce the single-valued normal magnetic field in the face AtoP
  loop, form the one-sided LLF electric fields inside the Riemann solver, and
  remove barriers that appeared unnecessary when passive scalars were absent
  or a serial transverse loop reached its final iteration.
- Correctness: the GPU CPAW convergence test continued to show fourth-order
  convergence.
- Sweep: relative to the retained local-halo result, HLLD-point throughput
  decreased by about 4% for both `64^3` and `128^3` blocks. LLF-line decreased
  by about 1%. The second-order control was unchanged within normal variation.
- Nsight Systems: LLF point-flux kernel time increased from 6.027 to 6.129
  seconds and total GPU kernel time increased from 10.255 to 10.409 seconds.
- Nsight Compute: directional kernel runtimes changed from
  14.17/16.72/17.06 ms to 14.69/16.82/17.24 ms. Registers per thread increased
  from 145--150 to 154--158. X1 barrier stalls decreased from 25.30 to 23.04
  stalled warps per issued instruction, but X2/X3 remained near 32 and overall
  runtime did not improve. No local-memory spill was observed.
- Decision: revert. The small reduction in one barrier metric did not offset
  the extra register pressure and the experiment did not address the serial
  X2/X3 transverse-loop structure. Retain the preceding LLF local-halo
  optimization.

### 2026-08-28 — Expand the X2/X3 face-reconstruction leagues

- Baseline: the retained point-flux league expansion, in which X2 and X3
  point AtoP/Riemann work had already been changed from serial transverse
  loops to one face row per GPU team.
- Change: apply the same decomposition to
  `ReconstructFaceAveragesTeissier()`. Each X2/X3 team now owns one
  reconstruction center and writes its left state to the next face and its
  right state to the current face. Remove the rolling transverse loop and
  reduce reconstruction scratch storage from three arrays to two. X1 and all
  WENO formulas remain unchanged.
- Results directory: `results_v100_128_20260828_150602/`
- Correctness: fourth-order CPAW convergence passed for both HLLD-point and
  LLF-line.
- Sweep results relative to the retained point-flux league baseline:

  | Scheme | Meshblock | Before (zone-cycles/s) | New | Change |
  |---|---:|---:|---:|---:|
  | UCT-HLLD 2nd, WENO3 | `64^3` | 3.47e7 | 3.47e7 | 0.0% |
  | Teissier4 HLLD-point | `64^3` | 5.25e6 | 5.36e6 | +2.1% |
  | Teissier4 LLF-line | `64^3` | 6.50e6 | 6.72e6 | +3.4% |
  | UCT-HLLD 2nd, WENO3 | `128^3` | 3.03e7 | 3.04e7 | +0.3% |
  | Teissier4 HLLD-point | `128^3` | 4.39e6 | 4.76e6 | +8.4% |
  | Teissier4 LLF-line | `128^3` | 5.46e6 | 6.01e6 | +10.1% |

- Nsight Systems profiles:
  `nsys_reports/nsys_profiles_after_league_expansion_FaceAverages/`
  LLF reconstruction time decreased from 2.563 to 2.253 seconds (-12.1%)
  and HLLD reconstruction time decreased from 2.436 to 2.130 seconds
  (-12.6%). LLF total GPU kernel time decreased from 9.692 to 9.348 seconds;
  HLLD decreased from 12.034 to 11.745 seconds. X1 reconstruction was
  unchanged, while X2/X3 LLF reconstruction decreased from 0.990/1.024 to
  0.858/0.847 seconds as their leagues grew from 544 to 35,904 teams.
- Nsight Compute note: the submitted profile retained the
  `CalculatePointFluxesTeissierImpl` kernel filter, so it acted as a control
  and confirmed that point-flux performance was unchanged. Profiling the
  reconstruction kernels would require a `ReconstructFaceAveragesTeissier`
  filter.
- Decision: retain. Convergence passed, both fourth-order solvers improved,
  and profiling localized the gain to the intended X2/X3 reconstruction
  kernels.

### 2026-08-28 — Thread-per-face LLF point-flux kernel

- Change: replace the LLF portion of `CalculatePointFluxesTeissier()` with
  flat kernels in which one GPU thread performs face AtoP, point conserved-to-
  primitive conversion, the LLF solve, and one-sided electric-field creation
  for one face. Remove LLF row scratch arrays and team barriers; leave HLLD
  unchanged.
- Results directory: `results_v100_128_20260828_164107/`.
- Correctness: CPAW retained the previous printed errors exactly through
  `N=64`; the added `64->128` RMS order was 4.6718. The second-order and
  HLLD-point sweep controls were unchanged.
- Sweep results relative to the retained face-reconstruction league baseline:

  | Scheme | Meshblock | Before (zone-cycles/s) | New | Change |
  |---|---:|---:|---:|---:|
  | UCT-HLLD 2nd, WENO3 | `64^3` | 3.47e7 | 3.47e7 | 0.0% |
  | Teissier4 HLLD-point | `64^3` | 5.36e6 | 5.36e6 | 0.0% |
  | Teissier4 LLF-line | `64^3` | 6.72e6 | 1.41e7 | +109.8% |
  | UCT-HLLD 2nd, WENO3 | `128^3` | 3.04e7 | 3.04e7 | 0.0% |
  | Teissier4 HLLD-point | `128^3` | 4.76e6 | 4.76e6 | 0.0% |
  | Teissier4 LLF-line | `128^3` | 6.01e6 | 1.10e7 | +83.0% |

- With the sweep's three-stage RK3 second-order control and five-stage RK4
  fourth-order route, the stage-normalized LLF cost is approximately 1.48x
  second order for `64^3` blocks and 1.65x for one `128^3` block.
- Nsight Systems profile:
  `nsys_reports/nsys_profiles_after_LLF_flux_refactor/`. Total GPU kernel time
  fell from 9.348 to 4.443 seconds (-52.5%). The three LLF point-flux kernels
  fell from 5.399 to 0.478 seconds (-91.1%), or from approximately 35.99 to
  3.19 ms per RHS across all three directions. No major cost moved into a
  downstream operation. Reconstruction is now the largest group at 2.278
  seconds (51.3% of total), followed by boundary packing at 0.743 seconds
  (16.7%); the point-flux kernels are 10.8% of total GPU kernel time.
- Nsight Compute profile: `ncu_reports/llf_thread_per_face/`. The X1/X2/X3
  kernels use 88/94/93 registers per thread, reserve zero local memory, and
  report zero barrier stalls. Achieved occupancy is approximately 31% and is
  register-limited. DRAM throughput is 44.1/41.6/47.2% of peak, while the
  dominant remaining scheduler stall is long-scoreboard latency
  (17.66/20.82/16.32 stalled warps per issued instruction). Thus this change
  removed the intended team-scratch/barrier bottleneck; the compact kernels
  are now primarily latency-limited rather than synchronization-limited.
- Decision: retain. This is the largest successful optimization in the log,
  preserves fourth-order convergence, and brings the `64^3`-block LLF route
  to about 1.48x the second-order cost per RK stage. Future work should target
  face reconstruction and boundary packing rather than further LLF-kernel
  fusion.

### 2026-09-08 — Thread-per-face HLLD point-flux kernel

- Change: replace the Teissier HLLD-point team/scratch face kernels with flat
  kernels in which one GPU thread owns one face and performs face AtoP,
  conserved-to-primitive conversion, the HLLD solve, and UCT-auxiliary output.
  The native second-order HLLD solver and the retained LLF-line flat kernels
  are unchanged.
- Results directory: `results_v100_128_20260908_185746/`.
- Correctness: HLLD-point CPAW errors through `N=64` match the retained route
  exactly at the precision written to `cpaw-errors.dat`. The new `64->128`
  RMS order is 4.3847. LLF-line remains fourth order, with a `64->128` RMS
  order of 4.6718.
- Sweep results relative to the retained face-reconstruction league baseline:

  | Scheme | Meshblock | Before (zone-cycles/s) | New | Change |
  |---|---:|---:|---:|---:|
  | UCT-HLLD 2nd, WENO3 | `64^3` | 3.47e7 | 3.51e7 | +1.2% |
  | Teissier4 HLLD-point | `64^3` | 5.36e6 | 1.21e7 | +125.7% |
  | Teissier4 LLF-line | `64^3` | 1.41e7 | 1.43e7 | +1.4% |
  | UCT-HLLD 2nd, WENO3 | `128^3` | 3.04e7 | 3.05e7 | +0.3% |
  | Teissier4 HLLD-point | `128^3` | 4.76e6 | 9.19e6 | +93.1% |
  | Teissier4 LLF-line | `128^3` | 1.10e7 | 1.10e7 | 0.0% |

- With the sweep's three-stage RK3 second-order control and five-stage RK4
  fourth-order route, the stage-normalized HLLD-point cost is approximately
  1.74x second order for `64^3` blocks and 1.99x for one `128^3` block.
- Nsight Systems profile:
  `nsys_reports/nsys_profiles_after_HLLD_flux_refactor/`. Total GPU kernel
  time fell from 11.745 to 5.262 seconds (-55.2%). The three HLLD point-face
  kernels fell from 7.068 to 0.677 seconds (-90.4%), or from approximately
  47.12 to 4.51 ms per RHS across all three directions. The new kernel-time
  composition is face reconstruction 2.134 s (40.6%), boundary packing and
  exchange 1.607 s (30.6%), HLLD point faces 0.677 s (12.9%), edge assembly
  0.233 s (4.4%), and flux PtoA 0.230 s (4.4%).
- Nsight Compute profile: `ncu_reports/hlld_thread_per_face/`. The X1/X2/X3
  kernels use 148/148/149 registers per thread, reserve zero local memory,
  and report zero barrier stalls. Achieved occupancy is approximately 18.6%
  and is register-limited. DRAM throughput is 38.9/35.6/41.3% of peak;
  eligible warps per active cycle are 0.25/0.22/0.27, and long-scoreboard
  stalls are 9.77/12.30/8.18 stalled warps per issued instruction. Branch-
  resolving stalls are small (approximately 0.39), so the branch-selective
  HLLD implementation is not primarily limited by divergence.
- Decision: retain. The refactor preserves the measured solution while more
  than doubling HLLD-point throughput. HLLD arithmetic is no longer the main
  Teissier bottleneck; reconstruction and boundary work now dominate. LLF-line
  remains faster overall and closer to the paper's reported per-RHS ratio.

### 2026-09-08 — Reduced HLLD intermediate exchange

- Change: generate `consflux_point` and `AL/AR/DL/DR` through one transverse
  halo locally. Exchange only `Bface_point` and a new two-component
  `teissier_hlld_vbart` field instead of exchanging point fluxes, all six HLLD
  auxiliaries, and the point magnetic field. Face-average reconstruction was
  extended from one to two transverse halo layers to support the local AtoP.
- Results directory: `results_v100_128_20260908_195602/`.
- Correctness: HLLD-point CPAW errors are identical to the preceding retained
  implementation at the precision written to `cpaw-errors.dat`, including a
  `64->128` RMS order of 4.3847. LLF-line remains unchanged and fourth order.
- Sweep results relative to the thread-per-face HLLD baseline:

  | Scheme | Meshblock | Before (zone-cycles/s) | New | Change |
  |---|---:|---:|---:|---:|
  | UCT-HLLD 2nd, WENO3 | `64^3` | 3.51e7 | 3.47e7 | -1.1% |
  | Teissier4 HLLD-point | `64^3` | 1.21e7 | 1.41e7 | +16.5% |
  | Teissier4 LLF-line | `64^3` | 1.43e7 | 1.42e7 | -0.7% |
  | UCT-HLLD 2nd, WENO3 | `128^3` | 3.05e7 | 3.04e7 | -0.3% |
  | Teissier4 HLLD-point | `128^3` | 9.19e6 | 1.17e7 | +27.3% |
  | Teissier4 LLF-line | `128^3` | 1.10e7 | 1.10e7 | 0.0% |

- With the three-stage RK3 control and five-stage RK4 fourth-order route, the
  stage-normalized HLLD-point cost is now approximately 1.48x second order for
  `64^3` blocks and 1.56x for one `128^3` block. HLLD-point and LLF-line are
  effectively tied for the preferred `64^3` layout; HLLD-point is faster for
  the single `128^3` block.
- Nsight Systems profile:
  `nsys_reports/nsys_profiles_after_HLLD_flux_refactor_and_new_intermediate_exchange/`.

  | Operation | Before | New | Change |
  |---|---:|---:|---:|
  | Boundary packing and exchange | 1.607 s | 0.600 s | -62.7% |
  | Face reconstruction | 2.134 s | 2.261 s | +6.0% |
  | HLLD point faces | 0.677 s | 0.729 s | +7.6% |
  | Total GPU kernel time | 5.262 s | 4.435 s | -15.7% |

  The extra local halo work is substantially cheaper than exchanging the
  larger field set. Reconstruction is now 51.0% of total GPU kernel time,
  followed by HLLD point faces at 16.4% and boundary work at 13.5%. The total
  is effectively identical to the optimized LLF-line profile's 4.443 s.
- Nsight Compute profile:
  `ncu_reports/hlld_thread_per_face_and_new_intermediate_exchange/`. The
  directional HLLD kernels remain at 148/148/149 registers per thread with no
  local-memory allocation or barrier stalls. Occupancy remains approximately
  18.6%, and the long-scoreboard and eligible-warp metrics are essentially
  unchanged. Their modest runtime increase is therefore attributable to the
  intentionally larger face ranges rather than a new kernel-resource problem.
- Decision: retain. The numerical result is unchanged, the reduced payload
  delivers the expected boundary-time reduction, and HLLD-point now performs
  approximately on par with LLF-line.

### 2026-09-10 — Restore original face-average-to-point arithmetic

- Change: restore the original X1 neighbor summation order and multiplication
  by `1.0 / 24.0` in `FaceAverageToPoint()` for 2D and 3D. X1 in 1D and
  X2/X3 already match the baseline expressions and remain unchanged.
- Baseline: `fddb789:src/hydro/high_order/teissier_fluxes.cpp`, also matching
  the main Teissier worktree's source at inspection time.
- Scope: retain flat point-face kernels, local halos, and reduced exchanges.
  Riemann arithmetic, EOS conversion, and passive-scalar density division
  remain unchanged to isolate this experiment.
- Validation: source expressions checked against the original L/R formulas
  in every supported direction/dimension. User rebuilt and ran the one-cycle
  comparison; results are in scratch `turbulence-one-cycle-compare/`
  `results_20260910_144301/comparison.txt`, compared with
  `results_20260909_124613/comparison.txt`.
- Manifest checks: original executable, input deck, and both restart hashes
  match the baseline; the optimized executable hash changed from
  `cbca3dad3d64c1de55ab4da71149076086faf826f24d7f199d3c5a1942d4911f`
  to `4aa4ba68cf82ea0f7eebe66e071a91c4150be632c7d44a52f61b0fd6d98687da`.
  Both runs use the same host, 128^3 mesh, and 64^3 meshblocks.
- HLLD-point: unequal conserved entries decreased from 1,907,573 to
  1,322,033 (11.37% to 7.88%); primitive relative L2 decreased from
  5.566442e-17 to 4.763739e-17. Maximum primitive difference remains
  1.110223e-15.
- LLF-line: unequal conserved entries decreased from 3,857,535 to
  3,804,473 (22.99% to 22.68%); primitive relative L2 decreased from
  6.019911e-17 to 5.647203e-17. Maximum primitive difference remains
  1.554312e-15.
- Both executables retain zero numerical mismatches on repeated runs.
  Old/new Time, dt, and NCycle match. Errors remain distributed through the
  mesh, without strong boundary enrichment. The comparison uses numerical
  equality, not raw-bit comparison (signed zero is not distinguished).
- Decision: retain for the arithmetic-restoration experiment. The change
  reduces mismatch counts but does not restore exact agreement; next isolate
  original Riemann arithmetic. No performance measurement or claim yet.

### 2026-09-10 — Restore original point-Riemann arithmetic

- Change: retain scalar interfaces and flat kernels, but restore the original
  LLF cyclic component order, magnetic-pressure expressions, staged flux sums,
  separate left/right magnetic energy-flux terms, and dissipation subtraction.
  The algebra is translated from the array solver at `fddb789`.
- HLLD: restore star-density reciprocal multiplication, reciprocal reuse in
  double-star states, the original left double-star momentum dot product for
  both energy corrections, and staged jumps before flux addition. Evaluate
  fast speeds using each input normal field, matching the original solver;
  Teissier callers still enforce a shared normal field before the solve.
- Retained: branch-selective HLLD state evaluation, flat kernel launches,
  local halos, reduced exchanges, and the preceding face-to-point restoration.
  Native array HLLD, EOS conversion, and passive-scalar conversion are unchanged.
- Validation: static source comparison and diff whitespace checks only.
  Source arithmetic restoration does not guarantee compiler-level bitwise
  agreement. User will compile and run the one-cycle comparison.
- Comparison baseline: `results_20260910_144301` in the scratch
  `turbulence-one-cycle-compare` directory (face-to-point restoration only).
- Run `results_20260910_152139` reused the same optimized executable SHA-256
  (`4aa4ba68cf82ea0f7eebe66e071a91c4150be632c7d44a52f61b0fd6d98687da`)
  as the face-to-point-only baseline, and its reported numerical metrics match.
  At inspection the executable mtime was 14:41:43, preceding the Riemann header
  edits at 15:14:14--15:14:15 on September 10. This run does not test the
  restored Riemann arithmetic; rebuild `build-gpu-v100` and rerun the comparison.
- Rebuilt comparison: `results_20260910_154258`, optimized SHA-256
  `ff8cab29682fb5044f96517095aeebf1de2c2c2b8fbad39761e41be6359877bd`.
  Manifest differences from `results_20260910_144301` are only creation time
  and optimized executable hash; original executable, input, checkpoints,
  host, and mesh layout match.
- HLLD-point: unequal conserved entries decrease from 1,322,033 to
  1,305,533 (1.25% reduction). Primitive relative L2 changes from
  4.763739e-17 to 4.750516e-17; maximum primitive difference stays 1.110223e-15.
- LLF-line: unequal conserved entries decrease from 3,804,473 to
  1,443,541 (62.06% reduction). Primitive relative L2 decreases from
  5.647203e-17 to 4.835773e-17; maximum primitive difference decreases from
  1.554312e-15 to 1.221245e-15. Mismatch-count reduction is not an equivalent
  reduction in error norm.
- All reported repeatability array mismatches remain zero. Old/new Time, dt,
  and NCycle match. Neither route has strong boundary enrichment; printed
  max|divB| agrees between executables for both routes.
- Decision: retain for the controlled arithmetic-restoration experiment;
  exact agreement is not restored. Next locate the first differing
  intermediate state/flux rather than infer the remaining cause from final
  mismatch counts. Performance remains unmeasured. Builds and numerical runs
  were performed by the user, not the agent.

### YYYY-MM-DD — Short description

- Commit:
- Change:
- Build/configuration differences:
- Results directory:
- `64^3`-block wall time and zone-cycles/s:
- Change from baseline:
- Correctness checks:
- Decision: keep / revise / revert
- Notes:
