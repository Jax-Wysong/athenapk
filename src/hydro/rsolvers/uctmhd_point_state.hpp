//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file uctmhd_point_state.hpp
//! \brief Compact point-state helpers shared by Teissier CT-MHD Riemann solvers.

#ifndef RSOLVERS_UCTMHD_POINT_STATE_HPP_
#define RSOLVERS_UCTMHD_POINT_STATE_HPP_

#include <cmath>

#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"

struct UCTMHDPointConserved {
  Real d, m1, m2, m3, e, b1, b2, b3;
};

struct UCTMHDPointPrimitive {
  Real d, v1, v2, v3, p, b1, b2, b3;
};

struct UCTMHDPointFlux {
  Real d, m1, m2, m3, e, b1, b2, b3;
};

// Scalar counterpart of AdiabaticCTMHDEOS::ConsToPrim(). The conserved state is
// mutable because floors and ceilings apply the normal EOS corrections.
KOKKOS_INLINE_FUNCTION void
UCTMHDConsToPrimPoint(UCTMHDPointConserved &u, UCTMHDPointPrimitive &w,
                      const AdiabaticCTMHDEOS &eos) {
  const Real gm1 = eos.GetGamma() - 1.0;
  const Real density_floor = eos.GetDensityFloor();
  const Real pressure_floor = eos.GetPressureFloor();
  const Real e_floor = eos.GetInternalEFloor();
  const Real velocity_ceiling = eos.GetVelocityCeiling();
  const Real e_ceiling = eos.GetInternalECeiling();

  PARTHENON_REQUIRE(u.d > 0.0 || density_floor > 0.0,
                    "Got negative density. Consider enabling first-order flux "
                    "correction or setting a reasonble density floor.");
  u.d = (u.d > density_floor) ? u.d : density_floor;
  w.d = u.d;

  const Real di = 1.0 / u.d;
  w.v1 = u.m1 * di;
  w.v2 = u.m2 * di;
  w.v3 = u.m3 * di;
  w.b1 = u.b1;
  w.b2 = u.b2;
  w.b3 = u.b3;

  Real e_k = 0.5 * di * (SQR(u.m1) + SQR(u.m2) + SQR(u.m3));
  const Real e_B = 0.5 * (SQR(u.b1) + SQR(u.b2) + SQR(u.b3));
  w.p = gm1 * (u.e - e_k - e_B);

  const Real v2 = SQR(w.v1) + SQR(w.v2) + SQR(w.v3);
  if (v2 > SQR(velocity_ceiling)) {
    const Real v = sqrt(v2);
    const Real fac = velocity_ceiling / v;
    w.v1 *= fac;
    w.v2 *= fac;
    w.v3 *= fac;
    u.m1 *= fac;
    u.m2 *= fac;
    u.m3 *= fac;
    const Real e_k_new = 0.5 * u.d * SQR(velocity_ceiling);
    u.e -= e_k - e_k_new;
    e_k = e_k_new;
  }

  PARTHENON_REQUIRE(w.p > 0.0 || pressure_floor > 0.0 || e_floor > 0.0,
                    "Got negative pressure. Consider enabling first-order flux "
                    "correction or setting a reasonable pressure or temperature floor.");

  if ((pressure_floor > 0.0) && (w.p < pressure_floor)) {
    u.e = (pressure_floor / gm1) + e_k + e_B;
    w.p = pressure_floor;
  }

  const Real effective_pressure_floor = gm1 * u.d * e_floor;
  if (w.p < effective_pressure_floor) {
    u.e = (u.d * e_floor) + e_k + e_B;
    w.p = effective_pressure_floor;
  }

  const Real effective_pressure_ceiling = gm1 * u.d * e_ceiling;
  if (w.p > effective_pressure_ceiling) {
    u.e = (u.d * e_ceiling) + e_k + e_B;
    w.p = effective_pressure_ceiling;
  }
}

#endif // RSOLVERS_UCTMHD_POINT_STATE_HPP_
