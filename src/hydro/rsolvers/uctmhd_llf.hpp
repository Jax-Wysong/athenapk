//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file uctmhd_llf.hpp
//! \brief Scalar LLF Riemann solver for the Teissier UCT-LLF path.

#ifndef RSOLVERS_UCTMHD_LLF_HPP_
#define RSOLVERS_UCTMHD_LLF_HPP_

#include <algorithm>
#include <cmath>

#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"
#include "rsolvers.hpp"
#include "uctmhd_point_state.hpp"

struct UCTLLFCons1D {
  Real d, mx, my, mz, e, by, bz;
};

template <>
struct Riemann<Fluid::ucthlldmhd, RiemannSolver::llf> {
  // Solve one face in physical component ordering. DIR is compile-time so the
  // normal/tangential component selection introduces no runtime branch.
  template <int DIR>
  static KOKKOS_INLINE_FUNCTION UCTMHDPointFlux
  SolveFace(const UCTMHDPointPrimitive &wl, const UCTMHDPointPrimitive &wr,
            Real &llf_speed, const AdiabaticCTMHDEOS &eos) {
    static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR,
                  "UCT-LLF face direction must be X1DIR, X2DIR, or X3DIR");

    // Use the original cyclic ordering and staged flux sums to preserve
    // source-level floating-point evaluation while keeping scalar inputs.
    UCTMHDPointPrimitive l = wl, r = wr;
    if constexpr (DIR == X2DIR) {
      l.v1 = wl.v2;
      l.v2 = wl.v3;
      l.v3 = wl.v1;
      l.b1 = wl.b2;
      l.b2 = wl.b3;
      l.b3 = wl.b1;
      r.v1 = wr.v2;
      r.v2 = wr.v3;
      r.v3 = wr.v1;
      r.b1 = wr.b2;
      r.b2 = wr.b3;
      r.b3 = wr.b1;
    } else if constexpr (DIR == X3DIR) {
      l.v1 = wl.v3;
      l.v2 = wl.v1;
      l.v3 = wl.v2;
      l.b1 = wl.b3;
      l.b2 = wl.b1;
      l.b3 = wl.b2;
      r.v1 = wr.v3;
      r.v2 = wr.v1;
      r.v3 = wr.v2;
      r.b1 = wr.b3;
      r.b2 = wr.b1;
      r.b3 = wr.b2;
    }
    const Real igm1 = 1.0 / (eos.GetGamma() - 1.0);

    const Real bxi = 0.5 * (l.b1 + r.b1);

    Real qa = l.d * l.v1;
    Real qb = r.d * r.v1;
    const Real qc =
        0.5 * (SQR(l.b2) + SQR(l.b3) - SQR(bxi));
    const Real qd =
        0.5 * (SQR(r.b2) + SQR(r.b3) - SQR(bxi));

    UCTLLFCons1D fsum;
    fsum.d = qa + qb;
    fsum.mx = qa * l.v1 + qb * r.v1 + qc + qd;
    fsum.my = qa * l.v2 + qb * r.v2 -
              bxi * (l.b2 + r.b2);
    fsum.mz = qa * l.v3 + qb * r.v3 -
              bxi * (l.b3 + r.b3);
    fsum.by = l.b2 * l.v1 + r.b2 * r.v1 -
              bxi * (l.v2 + r.v2);
    fsum.bz = l.b3 * l.v1 + r.b3 * r.v1 -
              bxi * (l.v3 + r.v3);

    const Real el =
        l.p * igm1 +
        0.5 * l.d *
            (SQR(l.v1) + SQR(l.v2) + SQR(l.v3)) +
        qc + SQR(bxi);
    const Real er =
        r.p * igm1 +
        0.5 * r.d *
            (SQR(r.v1) + SQR(r.v2) + SQR(r.v3)) +
        qd + SQR(bxi);
    fsum.mx += l.p + r.p;
    fsum.e = (el + l.p + qc) * l.v1 +
             (er + r.p + qd) * r.v1 -
             bxi * (l.b2 * l.v2 + l.b3 * l.v3) -
             bxi * (r.b2 * r.v2 + r.b3 * r.v3);

    qa = eos.FastMagnetosonicSpeed(l.d, l.p, l.b1,
                                   l.b2, l.b3);
    qb = eos.FastMagnetosonicSpeed(r.d, r.p, r.b1,
                                   r.b2, r.b3);
    const Real a =
        fmax(fabs(l.v1) + qa, fabs(r.v1) + qb);

    UCTLLFCons1D du;
    du.d = a * (r.d - l.d);
    du.mx = a * (r.d * r.v1 - l.d * l.v1);
    du.my = a * (r.d * r.v2 - l.d * l.v2);
    du.mz = a * (r.d * r.v3 - l.d * l.v3);
    du.e = a * (er - el);
    du.by = a * (r.b2 - l.b2);
    du.bz = a * (r.b3 - l.b3);

    llf_speed = a;
    UCTMHDPointFlux f;
    f.d = 0.5 * (fsum.d - du.d);
    f.e = 0.5 * (fsum.e - du.e);
    if constexpr (DIR == X1DIR) {
      f.m1 = 0.5 * (fsum.mx - du.mx);
      f.m2 = 0.5 * (fsum.my - du.my);
      f.m3 = 0.5 * (fsum.mz - du.mz);
      f.b1 = 0.0;
      f.b2 = 0.5 * (fsum.by - du.by);
      f.b3 = 0.5 * (fsum.bz - du.bz);
    } else if constexpr (DIR == X2DIR) {
      f.m2 = 0.5 * (fsum.mx - du.mx);
      f.m3 = 0.5 * (fsum.my - du.my);
      f.m1 = 0.5 * (fsum.mz - du.mz);
      f.b2 = 0.0;
      f.b3 = 0.5 * (fsum.by - du.by);
      f.b1 = 0.5 * (fsum.bz - du.bz);
    } else if constexpr (DIR == X3DIR) {
      f.m3 = 0.5 * (fsum.mx - du.mx);
      f.m1 = 0.5 * (fsum.my - du.my);
      f.m2 = 0.5 * (fsum.mz - du.mz);
      f.b3 = 0.0;
      f.b1 = 0.5 * (fsum.by - du.by);
      f.b2 = 0.5 * (fsum.bz - du.bz);
    }
    return f;
  }
};

#endif // RSOLVERS_UCTMHD_LLF_HPP_
