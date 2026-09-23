The cleanest roadmap is to implement sixth-order spatial support as an extension of the existing Teissier line routes, while keeping fourth order operational throughout.

The target sixth-order pipelines are:

$$
\begin{aligned}
\text{cell averages}
&\xrightarrow{\text{placeholder WENO7}}
\text{face averages}\\
&\xrightarrow{\text{AtoP6}}
\text{face-center states}\\
&\xrightarrow{\text{LLF/HLLD}}
\text{point fluxes and CT intermediates}\\
&\xrightarrow{\text{PtoA6}}
\text{area-average fluxes/intermediates}\\
&\xrightarrow{\text{placeholder WENO7}}
\text{line-average edge quantities}\\
&\rightarrow \text{FV/CT updates}.
\end{aligned}
$$


No high-order `hlld_point` path should be enabled.

## 1. Generalize the order-selection plumbing

Start in [hydro.cpp](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/hydro/hydro.cpp:409).

Allow:

```text
hydro/convergence_order = 2, 4, or 6
```

For order six, require:

```text
hydro/fluid = ucthlldmhd
hydro/reconstruction = weno7js
hydro/uct_solver = llf_line or hlld_line
hydro/riemann = llf or hlld, respectively
parthenon/mesh/nghost >= 4
```

Explicitly reject:

```text
convergence_order = 6
uct_solver = hlld_point
```

Add `weno7js` to the `Reconstruction` enum in [main.hpp](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/main.hpp:46) and to input parsing. You do not need to register it in the ordinary AthenaPK flux-function map because the Teissier driver bypasses that map.

Replace concepts such as:

```cpp
const bool teissier4 =
    fluid == Fluid::ucthlldmhd && convergence_order == 4;
```

with something like:

```cpp
const bool teissier_high_order =
    fluid == Fluid::ucthlldmhd &&
    (convergence_order == 4 || convergence_order == 6);

const bool teissier_hlld_point =
    convergence_order == 4 && uct_solver == "hlld_point";
```

Use `teissier_high_order` for common task-graph behavior and `teissier_hlld_point` only for the fourth-order point route.

The existing uniform-mesh and no-first-order-flux-correction restrictions should apply to both fourth and sixth order.

## 2. Introduce order-aware transformation helpers

Before modifying all the kernels, centralize the transformation arithmetic. A small header such as:

```text
src/hydro/high_order/teissier_transforms.hpp
```

would help prevent three or four independent copies of the sixth-order coefficients.

The most useful interfaces would be conceptually:

```cpp
template <int ORDER, int DIR, typename View>
KOKKOS_INLINE_FUNCTION
Real FaceAverageToPoint(...);

template <int ORDER, int DIR, typename View>
KOKKOS_INLINE_FUNCTION
Real PointToFaceAverage(...);
```

These should handle dimensional reduction correctly:

- 1D face average = point value;
- 2D face average is a line average;
- 3D face average is an area average.

Dispatch on `convergence_order` outside the kernels where possible:

```cpp
if (order == 4) return Implementation<4>(md);
if (order == 6) return Implementation<6>(md);
```

That lets `if constexpr` eliminate unused stencil terms on the GPU.

### Sixth-order area-average to point

In 3D, for the two-dimensional plane transverse to the face normal:

\[
q_0=
\frac{1771}{1440}\bar q_0
-\frac{23}{360}\bar q^{\Sigma10}
+\frac{3}{640}\bar q^{\Sigma20}
+\frac{1}{576}\bar q^{\Sigma11}.
\]

Here:

- \(\Sigma10\): four axial neighbors at distance one;
- \(\Sigma20\): four axial neighbors at distance two;
- \(\Sigma11\): four diagonal neighbors \((\pm1,\pm1)\).

In 2D, use the projected line-average formula:

\[
q_0=
\frac{1067}{960}\bar q_0
-\frac{29}{480}\bar q^{\Sigma1}
+\frac{3}{640}\bar q^{\Sigma2}.
\]

Do not obtain the 2D expression by simply removing one direction from the 3D expression; the coefficients are different.

### Sixth-order point to area-average

In 3D:

\[
\bar q_0=
\frac{1159}{1440}q_0
+\frac{1}{20}q^{\Sigma10}
-\frac{17}{5760}q^{\Sigma20}
+\frac{1}{576}q^{\Sigma11}.
\]

In 2D, point to line average:

\[
\bar q_0=
\frac{863}{960}q_0
+\frac{77}{1440}q^{\Sigma1}
-\frac{17}{5760}q^{\Sigma2}.
\]

Use these same point-to-face-average helpers for:

- numerical fluxes;
- LLF point electric fields;
- HLLD point \(v_tB_n\) products.

Preserve the current fourth-order expression ordering in the `<4>` specialization to minimize unnecessary regression differences.

## 3. Increase the locally valid stencil regions

This is the most important implementation detail.

Define:

```cpp
const int point_halo =
    convergence_order == 4 ? 1 : 2;

const int reconstructed_face_halo =
    convergence_order == 4 ? 2 : 4;
```

The reasoning is:

- PtoA4 needs point values through transverse offset one.
- PtoA6 needs axial offsets two and diagonal offsets \((\pm1,\pm1)\).
- AtoP6 around those halo-two point locations itself reaches another two cells.
- Therefore locally reconstructed face averages must extend to transverse halo four.

The intended valid regions are:

| Quantity | Fourth order | Sixth order |
|---|---:|---:|
| Reconstructed face averages | transverse halo 2 | transverse halo 4 |
| Point fluxes | transverse halo 1 | transverse halo 2 |
| Point LLF electric fields | transverse halo 1 | transverse halo 2 |
| Point HLLD \(v_tB_n\) products | transverse halo 1 | transverse halo 2 |
| Point HLLD \(a,d\) coefficients | currently halo 1 | generated through halo 2 with the point solve |
| Communicated area intermediates | global ghost width 3 | global ghost width 4 |

Update [ReconstructFaceAveragesTeissier](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/hydro/high_order/teissier_fluxes.cpp:147) first.

For an \(x\)-normal face in 3D, the reconstructed region must cover the complete transverse box:

```text
j = js-4 ... je+4
k = ks-4 ... ke+4
```

and similarly under direction permutation. It must be a box, not only axial strips, because AtoP6 uses diagonal transverse values.

Then update the loops in:

- `CalculatePointFluxesTeissierLLF`
- `CalculatePointFluxesTeissierHLLDLine`

from transverse halo one to transverse halo two.

Leave `CalculatePointFluxesTeissierHLLDPoint` unchanged.

### Ghost-zone conclusion

Use `nghost=4` for the sixth-order route. It covers:

- WENO7 reconstruction accesses;
- transverse face reconstruction out to four layers;
- the eight-value union used by paired WENO7 edge reconstructions;
- sixth-order magnetic centering.

The existing intermediate-field exchanges in [hydro_driver.cpp](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/hydro/hydro_driver.cpp:619) should remain sufficient. No fundamentally new exchange stage is needed:

1. generate point quantities locally;
2. convert interior point quantities to area averages;
3. exchange the area-average intermediate field;
4. reconstruct it to edges.

The critical check is that Parthenon supplies valid corner ghost data for `cons` and `Bface` at depth four.

## 4. Add placeholder WENO7 interfaces

Use a stable intended API now so the actual WENO7 implementation can be inserted later.

For cell-average-to-face reconstruction:

```cpp
Reconstruct<Reconstruction::weno7js, DIR>(...);
```

For direct scalar reconstruction:

```cpp
WENO7JS(q_im3, q_im2, q_im1, q_i,
        q_ip1, q_ip2, q_ip3,
        ql_ip1, qr_i);
```

For reconstructing the two states at one edge, create a shared helper analogous to `ReconstructWENO5Interface`:

```cpp
ReconstructWENO7Interface(
    q_m4, q_m3, q_m2, q_m1,
    q_0, q_p1, q_p2, q_p3,
    q_from_lower, q_from_upper);
```

Its eventual structure should be:

```cpp
WENO7JS(q_m4, q_m3, q_m2, q_m1,
        q_0, q_p1, q_p2,
        q_from_lower, unused);

WENO7JS(q_m3, q_m2, q_m1, q_0,
        q_p1, q_p2, q_p3,
        unused, q_from_upper);
```

This gives the union stencil \([-4,+3]\) around the edge.

I would isolate the placeholder behind these two APIs instead of placing raw provisional calls everywhere. Also, avoid silently labeling a WENO5 fallback as sixth order. If you temporarily use WENO5 to exercise task plumbing, emit a conspicuous warning or keep the order-six input disabled for scientific runs.

## 5. Upgrade the common finite-volume flux conversion

Modify [PointToAreaAverage](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/hydro/high_order/conversions.cpp:31) to dispatch between fourth- and sixth-order implementations.

The output range remains only the physical faces needed by the finite-volume update. The expanded point-solver loops provide the input halo.

For each face:

- 1D: copy point flux;
- 2D: apply sixth-order point-to-line;
- 3D: apply sixth-order point-to-area, including diagonals.

This part is identical for `llf_line` and `hlld_line`.

A useful intermediate checkpoint is to test this conversion on constants and analytic polynomials before touching CT.

## 6. Upgrade `llf_line`

The LLF route is the most direct Teissier implementation and should come first.

### Point electric fields

Modify `PointElectricFieldsToAreaAverage` to use the common sixth-order PtoA helper. Its mathematical sequence remains:

```text
point states
→ point v×B
→ sixth-order face average of v×B
→ exchange face averages
```

### Edge reconstruction

In [uctllfmhd.cpp](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/hydro/ctmhd/uctllfmhd.cpp:16):

- dispatch to `ReconstructWENO5Interface` for order four;
- dispatch to placeholder `ReconstructWENO7Interface` for order six;
- apply the same selection to both the electric fields and transverse `Bface` values.

Do not change the final LLF edge formula:

\[
E_z=
\frac14(E_z^{SW}+E_z^{SE}+E_z^{NW}+E_z^{NE})
+\frac{S}{2}(B_y^E-B_y^W)
-\frac{S}{2}(B_x^N-B_x^S).
\]

Also leave

\[
S=\max(a_x^S,a_x^N,a_y^W,a_y^E)
\]

unchanged. `S` remains an edge dissipation coefficient; it is not an averaged physical field that needs AtoP or PtoA conversion.

## 7. Upgrade `hlld_line`

After `llf_line` is structurally working, apply the same changes to `hlld_line`.

### Point products

`CalculateHLLDLineFace` remains conceptually unchanged:

```cpp
vbartB(face, 0, ...) = result.aux.vT1 * bnormal;
vbartB(face, 1, ...) = result.aux.vT2 * bnormal;
```

These are properly collocated point products.

Modify `PointHLLDProductsToAreaAverage` to use PtoA6, including the diagonal transverse terms.

### Edge reconstruction

In [Assemble_HLLD_Line_Edge_EMF](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/hydro/ctmhd/ucthlldmhd.cpp:595), replace the repeated WENO5 pairs with an order-aware interface helper. Apply it to:

- area-averaged \(v_xB_y\), \(v_yB_x\), and their permutations;
- staggered area-averaged transverse magnetic components.

The final HLLD edge formula remains unchanged.

### Leave \(a\) and \(d\) alone initially

Continue forming edge coefficients using the current adjacent-face arithmetic averages:

```cpp
0.5 * (coefficient_lower_face + coefficient_upper_face)
```

Do not apply AtoP/PtoA transforms to `a` or `d`.

This is the one part of sixth-order `hlld_line` not prescribed directly by Teissier. The working hypothesis remains that errors in these upwind coefficients multiply increasingly small reconstructed jumps and therefore do not necessarily limit the overall smooth convergence order. Test that hypothesis with CPAW; only revisit the coefficient construction if `llf_line` reaches sixth order and `hlld_line` does not.

## 8. Upgrade staggered magnetic-field centering

Modify [center_Mag_Field](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/hydro/ctmhd/ctmhd.cpp:468).

Generalize:

```cpp
convergence_order == 4 ? IndexDomain::interior : ...
```

to all orders above two, because the high-order interpolation should continue to operate only on interior cells after the dedicated `Bface` exchange.

For sixth order use Teissier equation (22):

\[
\langle B_x\rangle_i =
\frac{
802(B_{x,i-1/2}+B_{x,i+1/2})
-93(B_{x,i-3/2}+B_{x,i+3/2})
+11(B_{x,i-5/2}+B_{x,i+5/2})
}{1440}.
\]

With the current indexing that corresponds to:

```text
nearest: Bface(i),   Bface(i+1)
next:    Bface(i-1), Bface(i+2)
outer:   Bface(i-2), Bface(i+3)
```

and direction permutations for \(B_y\) and \(B_z\).

The existing stage-end sequence remains correct:

```text
exchange Bface
→ center interior B into cons
→ exchange cons
```

`ReconcileSelfPeriodicFluxesAndEmfs` is order-independent and should not need numerical changes.

## 9. Upgrade smooth high-order initialization

Without sixth-order initial averages, a CPAW convergence study can be capped at fourth order even if the evolution operator is correct.

Generalize [mhd_pgen_utils.hpp](/mnt/ffs24/home/wysongj2/athenapk-fourth-Teissier24-performance/src/pgen/ct_glm_compatible/mhd_pgen_utils.hpp:20) from “fourth order” to order-aware initialization.

### Point conserved state to cell-volume average

For a 3D sixth-order volume average:

\[
\bar q_0 =
\frac{689}{960}q_0
+\frac{67}{1440}q^{\Sigma100}
-\frac{17}{5760}q^{\Sigma200}
+\frac{1}{576}q^{\Sigma110}.
\]

Here:

- \(\Sigma100\): six axial distance-one neighbors;
- \(\Sigma200\): six axial distance-two neighbors;
- \(\Sigma110\): twelve two-direction diagonal neighbors.

Use the appropriate projected line/area formulas in 1D and 2D.

### Vector potential to face-average magnetic field

Convert pointwise vector potential to sixth-order line averages using:

\[
\bar A_0=
\frac{863}{960}A_0
+\frac{77}{1440}A^{\Sigma1}
-\frac{17}{5760}A^{\Sigma2}.
\]

Then take the same discrete curl of those line averages. This retains roundoff-level discrete divergence.

Initially, upgrading CPAW and the smooth-vortex/linear-wave shared initialization is enough. Nonsmooth Gaussian-quadrature initializers can be addressed later; a three-point Gauss rule in each active integration direction is the natural sixth-order choice.

## 10. Recommended implementation checkpoints

Make the changes in this order:

1. Generalize order selection and metadata without changing fourth-order numerics.
2. Add the reusable AtoP/PtoA helper functions.
3. Implement sixth-order transformation formulas.
4. Expand face-average reconstruction to halo four.
5. Expand line-route point solves to halo two.
6. Upgrade common point-flux-to-area conversion.
7. Upgrade `llf_line` point electric-field conversion.
8. Add placeholder WENO7 edge calls to `llf_line`.
9. Add sixth-order magnetic centering.
10. Upgrade smooth initialization.
11. Validate `llf_line`.
12. Upgrade HLLD product conversion and edge reconstruction.
13. Validate `hlld_line`.

This keeps the paper-prescribed LLF path as the reference implementation before introducing the more experimental HLLD edge coefficients.

## Validation criteria

Before claiming sixth-order evolution:

- Constants must be preserved exactly by every transformation.
- Polynomial/manufactured samples should verify the AtoP6 and PtoA6 formulas independently.
- Fourth-order `llf_line` and `hlld_line` results should remain unchanged within expected roundoff.
- Sixth-order CPAW should converge near order six when RK4 temporal error is suppressed.
- Single-block and multiblock CPAW should agree consistently.
- `maxFaceDivB` should remain at roundoff.
- `llf_line` should be validated beforke diagnosing `hlld_line`.
- Shock tests should remain stable, but should not be used to measure formal sixth-order convergence.

The most likely implementation error will be an insufficient valid region rather than a wrong coefficient. In particular, inspect every sixth-order diagonal access near a MeshBlock corner before interpreting a failed multiblock convergence test.



## Adjustments along the way
- For weno7, use epsilon = 1e-26 and and exponent of 4.
- For weno5, switch also to epsilon = 1e-26 but leave exponent of 2
  - This is done on purpose so as to make one-to-one comparisons to results from Wendeln's WENO-based FD Spartha code