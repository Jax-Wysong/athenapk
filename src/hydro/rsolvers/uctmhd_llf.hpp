//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file uctmhd_llf.hpp
//! \brief LLF Riemann solver for the Teissier UCT-LLF path.

#ifndef RSOLVERS_UCTMHD_LLF_HPP_
#define RSOLVERS_UCTMHD_LLF_HPP_

#include <algorithm>
#include <cmath>

#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "rsolvers.hpp"

using TE = parthenon::TopologicalElement;

struct UCTLLFCons1D {
  Real d, mx, my, mz, e, by, bz;
};

template <>
struct Riemann<Fluid::ucthlldmhd, RiemannSolver::llf> {
  template <bool point_flux, typename FluxPack>
  static KOKKOS_INLINE_FUNCTION void
  Solve(const parthenon::team_mbr_t &member, const int k, const int j,
        const int il, const int iu, const int ivx,
        const parthenon::ScratchPad2D<Real> &wl,
        const parthenon::ScratchPad2D<Real> &wr, FluxPack &flux_out,
        parthenon::VariablePack<Real> &llf_speed,
        const AdiabaticCTMHDEOS &eos, const Real /*c_h*/) {
    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = NHYDRO + ivx - IV1;
    const int iBy = NHYDRO + ivy - IV1;
    const int iBz = NHYDRO + ivz - IV1;
    const Real igm1 = 1.0 / (eos.GetGamma() - 1.0);
    constexpr int NCTMHD = 8;

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      Real wli[NCTMHD], wri[NCTMHD];

      wli[IDN] = wl(IDN, i);
      wli[IV1] = wl(ivx, i);
      wli[IV2] = wl(ivy, i);
      wli[IV3] = wl(ivz, i);
      wli[IPR] = wl(IPR, i);
      wli[IB1] = wl(iBx, i);
      wli[IB2] = wl(iBy, i);
      wli[IB3] = wl(iBz, i);

      wri[IDN] = wr(IDN, i);
      wri[IV1] = wr(ivx, i);
      wri[IV2] = wr(ivy, i);
      wri[IV3] = wr(ivz, i);
      wri[IPR] = wr(IPR, i);
      wri[IB1] = wr(iBx, i);
      wri[IB2] = wr(iBy, i);
      wri[IB3] = wr(iBz, i);

      const Real bxi = 0.5 * (wli[IB1] + wri[IB1]);

      Real qa = wli[IDN] * wli[IV1];
      Real qb = wri[IDN] * wri[IV1];
      const Real qc =
          0.5 * (SQR(wli[IB2]) + SQR(wli[IB3]) - SQR(bxi));
      const Real qd =
          0.5 * (SQR(wri[IB2]) + SQR(wri[IB3]) - SQR(bxi));

      UCTLLFCons1D fsum;
      fsum.d = qa + qb;
      fsum.mx = qa * wli[IV1] + qb * wri[IV1] + qc + qd;
      fsum.my = qa * wli[IV2] + qb * wri[IV2] -
                bxi * (wli[IB2] + wri[IB2]);
      fsum.mz = qa * wli[IV3] + qb * wri[IV3] -
                bxi * (wli[IB3] + wri[IB3]);
      fsum.by = wli[IB2] * wli[IV1] + wri[IB2] * wri[IV1] -
                bxi * (wli[IV2] + wri[IV2]);
      fsum.bz = wli[IB3] * wli[IV1] + wri[IB3] * wri[IV1] -
                bxi * (wli[IV3] + wri[IV3]);

      const Real el =
          wli[IPR] * igm1 +
          0.5 * wli[IDN] *
              (SQR(wli[IV1]) + SQR(wli[IV2]) + SQR(wli[IV3])) +
          qc + SQR(bxi);
      const Real er =
          wri[IPR] * igm1 +
          0.5 * wri[IDN] *
              (SQR(wri[IV1]) + SQR(wri[IV2]) + SQR(wri[IV3])) +
          qd + SQR(bxi);
      fsum.mx += wli[IPR] + wri[IPR];
      fsum.e = (el + wli[IPR] + qc) * wli[IV1] +
               (er + wri[IPR] + qd) * wri[IV1] -
               bxi * (wli[IB2] * wli[IV2] + wli[IB3] * wli[IV3]) -
               bxi * (wri[IB2] * wri[IV2] + wri[IB3] * wri[IV3]);

      qa = eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1],
                                     wli[IB2], wli[IB3]);
      qb = eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1],
                                     wri[IB2], wri[IB3]);
      const Real a =
          fmax(fabs(wli[IV1]) + qa, fabs(wri[IV1]) + qb);

      UCTLLFCons1D du;
      du.d = a * (wri[IDN] - wli[IDN]);
      du.mx = a * (wri[IDN] * wri[IV1] - wli[IDN] * wli[IV1]);
      du.my = a * (wri[IDN] * wri[IV2] - wli[IDN] * wli[IV2]);
      du.mz = a * (wri[IDN] * wri[IV3] - wli[IDN] * wli[IV3]);
      du.e = a * (er - el);
      du.by = a * (wri[IB2] - wli[IB2]);
      du.bz = a * (wri[IB3] - wli[IB3]);

      const auto face =
          ivx == IV1 ? TE::F1 : (ivx == IV2 ? TE::F2 : TE::F3);
      if constexpr (point_flux) {
        flux_out(face, IDN, k, j, i) = 0.5 * (fsum.d - du.d);
        flux_out(face, ivx, k, j, i) = 0.5 * (fsum.mx - du.mx);
        flux_out(face, ivy, k, j, i) = 0.5 * (fsum.my - du.my);
        flux_out(face, ivz, k, j, i) = 0.5 * (fsum.mz - du.mz);
        flux_out(face, IEN, k, j, i) = 0.5 * (fsum.e - du.e);
        flux_out(face, iBx, k, j, i) = 0.0;
        flux_out(face, iBy, k, j, i) = 0.5 * (fsum.by - du.by);
        flux_out(face, iBz, k, j, i) = 0.5 * (fsum.bz - du.bz);
      } else {
        flux_out.flux(ivx, IDN, k, j, i) = 0.5 * (fsum.d - du.d);
        flux_out.flux(ivx, ivx, k, j, i) = 0.5 * (fsum.mx - du.mx);
        flux_out.flux(ivx, ivy, k, j, i) = 0.5 * (fsum.my - du.my);
        flux_out.flux(ivx, ivz, k, j, i) = 0.5 * (fsum.mz - du.mz);
        flux_out.flux(ivx, IEN, k, j, i) = 0.5 * (fsum.e - du.e);
        flux_out.flux(ivx, iBx, k, j, i) = 0.0;
        flux_out.flux(ivx, iBy, k, j, i) = 0.5 * (fsum.by - du.by);
        flux_out.flux(ivx, iBz, k, j, i) = 0.5 * (fsum.bz - du.bz);
      }
      llf_speed(face, 0, k, j, i) = a;
    });
  }
};

#endif // RSOLVERS_UCTMHD_LLF_HPP_
