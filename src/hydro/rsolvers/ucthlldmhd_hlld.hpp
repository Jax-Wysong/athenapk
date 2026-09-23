//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file hlld.cpp
//! \brief UCT-HLLD Riemann solver for adiabatic MHD.
//!
//! REFERENCES:
//! - T. Miyoshi & K. Kusano, "A multi-state HLL approximate Riemann solver for ideal
//!   MHD", JCP, 208, 315 (2005)


#ifndef RSOLVERS_UCTHLLDMHD_HLLD_HPP_
#define RSOLVERS_UCTHLLDMHD_HLLD_HPP_

// C++ headers
#include <algorithm> // max(), min()
#include <cmath>     // sqrt()

// Athena headers
#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"
#include "interface/variable_pack.hpp"
#include "rsolvers.hpp"
#include "uctmhd_point_state.hpp"

using TE = parthenon::TopologicalElement;


// container to store (density, momentum, total energy, tranverse magnetic field)
// minimizes changes required to adopt athena4.2 version of this solver

struct UCTHLLDCons1D {
  Real d, mx, my, mz, e, by, bz;
};

struct UCTHLLDPrim1D {
  Real d, vx, vy, vz, p, bx, by, bz;
};

struct UCTHLLDStar1D {
  UCTHLLDCons1D u;
  Real vb;
};

struct UCTHLLDPointAux {
  Real aL, aR, dL, dR, vT1, vT2;
};

struct UCTHLLDPointResult {
  UCTMHDPointFlux flux;
  UCTHLLDPointAux aux;
};

#define SMALL_NUMBER 1.0e-8

template <>
struct Riemann<Fluid::ucthlldmhd, RiemannSolver::hlld> {
  static KOKKOS_INLINE_FUNCTION UCTHLLDCons1D
  PointConserved1D(const UCTHLLDPrim1D &w, const Real igm1,
                   const Real magnetic_pressure) {
    UCTHLLDCons1D u;
    u.d = w.d;
    u.mx = w.vx * u.d;
    u.my = w.vy * u.d;
    u.mz = w.vz * u.d;
    u.e = w.p * igm1 +
          0.5 * w.d * (SQR(w.vx) + (SQR(w.vy) + SQR(w.vz))) +
          magnetic_pressure;
    u.by = w.by;
    u.bz = w.bz;
    return u;
  }

  static KOKKOS_INLINE_FUNCTION UCTHLLDCons1D
  PhysicalFlux1D(const UCTHLLDPrim1D &w, const UCTHLLDCons1D &u,
                 const Real total_pressure, const Real bxsq) {
    UCTHLLDCons1D f;
    f.d = u.mx;
    f.mx = u.mx * w.vx + total_pressure - bxsq;
    f.my = u.my * w.vx - w.bx * u.by;
    f.mz = u.mz * w.vx - w.bx * u.bz;
    f.e = w.vx * (u.e + total_pressure - bxsq) -
          w.bx * (w.vy * u.by + w.vz * u.bz);
    f.by = u.by * w.vx - w.bx * w.vy;
    f.bz = u.bz * w.vx - w.bx * w.vz;
    return f;
  }

  static KOKKOS_INLINE_FUNCTION UCTHLLDStar1D
  StarState1D(const UCTHLLDPrim1D &w, const UCTHLLDCons1D &u,
              const Real sd, const Real sdm, const Real sm,
              const Real total_pressure, const Real star_total_pressure,
              const Real bxsq) {
    UCTHLLDStar1D star;
    const Real sdm_inv = 1.0 / sdm;
    star.u.d = u.d * sd * sdm_inv;
    const Real star_d_inv = 1.0 / star.u.d;
    star.u.mx = star.u.d * sm;

    const Real denom = u.d * sd * sdm - bxsq;
    if (std::abs(denom) < SMALL_NUMBER * star_total_pressure) {
      star.u.my = star.u.d * w.vy;
      star.u.mz = star.u.d * w.vz;
      star.u.by = u.by;
      star.u.bz = u.bz;
    } else {
      Real tmp = w.bx * (sd - sdm) / denom;
      star.u.my = star.u.d * (w.vy - u.by * tmp);
      star.u.mz = star.u.d * (w.vz - u.bz * tmp);
      tmp = (u.d * SQR(sd) - bxsq) / denom;
      star.u.by = u.by * tmp;
      star.u.bz = u.bz * tmp;
    }

    star.vb = (star.u.mx * w.bx +
               (star.u.my * star.u.by + star.u.mz * star.u.bz)) *
              star_d_inv;
    star.u.e =
        (sd * u.e - total_pressure * w.vx + star_total_pressure * sm +
         w.bx * (w.vx * w.bx + (w.vy * u.by + w.vz * u.bz) - star.vb)) *
        sdm_inv;
    return star;
  }

  static KOKKOS_INLINE_FUNCTION void
  AddJump1D(UCTHLLDCons1D &f, const Real speed, const UCTHLLDCons1D &to,
            const UCTHLLDCons1D &from) {
    // Stage the jump before flux addition, as in the original solver.
    UCTHLLDCons1D jump;
    jump.d = speed * (to.d - from.d);
    jump.mx = speed * (to.mx - from.mx);
    jump.my = speed * (to.my - from.my);
    jump.mz = speed * (to.mz - from.mz);
    jump.e = speed * (to.e - from.e);
    jump.by = speed * (to.by - from.by);
    jump.bz = speed * (to.bz - from.bz);
    f.d += jump.d;
    f.mx += jump.mx;
    f.my += jump.my;
    f.mz += jump.mz;
    f.e += jump.e;
    f.by += jump.by;
    f.bz += jump.bz;
  }

  // Scalar HLLD solve for one Teissier point face. Inputs and outputs retain
  // physical component ordering; the local HLLD algebra uses cyclic
  // normal/tangential ordering matching the original team solver.
  template <int DIR>
  static KOKKOS_INLINE_FUNCTION UCTHLLDPointResult
  SolvePointFace(const UCTMHDPointPrimitive &wl,
                 const UCTMHDPointPrimitive &wr,
                 const AdiabaticCTMHDEOS &eos) {
    static_assert(DIR == X1DIR || DIR == X2DIR || DIR == X3DIR,
                  "UCT-HLLD face direction must be X1DIR, X2DIR, or X3DIR");

    UCTHLLDPrim1D l, r;
    l.d = wl.d;
    l.p = wl.p;
    r.d = wr.d;
    r.p = wr.p;
    if constexpr (DIR == X1DIR) {
      l.vx = wl.v1;
      l.vy = wl.v2;
      l.vz = wl.v3;
      r.vx = wr.v1;
      r.vy = wr.v2;
      r.vz = wr.v3;
      l.bx = wl.b1;
      l.by = wl.b2;
      l.bz = wl.b3;
      r.bx = wr.b1;
      r.by = wr.b2;
      r.bz = wr.b3;
    } else if constexpr (DIR == X2DIR) {
      l.vx = wl.v2;
      l.vy = wl.v3;
      l.vz = wl.v1;
      r.vx = wr.v2;
      r.vy = wr.v3;
      r.vz = wr.v1;
      l.bx = wl.b2;
      l.by = wl.b3;
      l.bz = wl.b1;
      r.bx = wr.b2;
      r.by = wr.b3;
      r.bz = wr.b1;
    } else {
      l.vx = wl.v3;
      l.vy = wl.v1;
      l.vz = wl.v2;
      r.vx = wr.v3;
      r.vy = wr.v1;
      r.vz = wr.v2;
      l.bx = wl.b3;
      l.by = wl.b1;
      l.bz = wl.b2;
      r.bx = wr.b3;
      r.by = wr.b1;
      r.bz = wr.b2;
    }

    const Real igm1 = 1.0 / (eos.GetGamma() - 1.0);
    const Real bx = 0.5 * (l.bx + r.bx);
    // The original solver evaluates wave speeds with each input normal field.
    const Real cfl = eos.FastMagnetosonicSpeed(l.d, l.p, l.bx, l.by, l.bz);
    const Real cfr = eos.FastMagnetosonicSpeed(r.d, r.p, r.bx, r.by, r.bz);
    l.bx = bx;
    r.bx = bx;
    const Real bxsq = SQR(bx);
    const Real pbl = 0.5 * (bxsq + (SQR(l.by) + SQR(l.bz)));
    const Real pbr = 0.5 * (bxsq + (SQR(r.by) + SQR(r.bz)));
    const Real ptl = l.p + pbl;
    const Real ptr = r.p + pbr;
    const UCTHLLDCons1D ul = PointConserved1D(l, igm1, pbl);
    const UCTHLLDCons1D ur = PointConserved1D(r, igm1, pbr);

    const Real sl = std::min(l.vx - cfl, r.vx - cfr);
    const Real sr = std::max(l.vx + cfl, r.vx + cfr);
    const Real sdl = sl - l.vx;
    const Real sdr = sr - r.vx;
    const Real sm =
        (sdr * ur.mx - sdl * ul.mx + (ptl - ptr)) /
        (sdr * ur.d - sdl * ul.d);
    const Real sdml = sl - sm;
    const Real sdmr = sr - sm;
    // Match the reciprocal multiplication used by the original star states.
    const Real sdml_inv = 1.0 / sdml;
    const Real sdmr_inv = 1.0 / sdmr;
    const Real dlstar = ul.d * sdl * sdml_inv;
    const Real drstar = ur.d * sdr * sdmr_inv;
    const Real sqrtdl = std::sqrt(dlstar);
    const Real sqrtdr = std::sqrt(drstar);
    const Real slstar = sm - std::abs(bx) / sqrtdl;
    const Real srstar = sm + std::abs(bx) / sqrtdr;

    UCTHLLDPointResult result;
    {
      constexpr Real uct_eps = 1.0e-9;
      Real nuS = 0.0;
      if (std::abs(srstar - slstar) > uct_eps * std::abs(sr - sl)) {
        nuS = (std::abs(srstar) - std::abs(slstar)) / (srstar - slstar);
      }
      const Real nuL = (slstar + sl) / (std::abs(slstar) + std::abs(sl));
      const Real nuR = (srstar + sr) / (std::abs(srstar) + std::abs(sr));
      const Real tchiL =
          ((l.vx - sm) * (sl - sm)) / (slstar + sl - 2.0 * sm);
      const Real tchiR =
          ((r.vx - sm) * (sr - sm)) / (srstar + sr - 2.0 * sm);
      const Real alphaP = std::max(0.0, sr);
      const Real alphaM = -std::min(0.0, sl);
      result.aux.aL = 0.5 * (1.0 + nuS);
      result.aux.aR = 0.5 * (1.0 - nuS);
      result.aux.dL =
          0.5 * (nuL - nuS) * tchiL +
          0.5 * (std::abs(slstar) - nuS * slstar);
      result.aux.dR =
          0.5 * (nuR - nuS) * tchiR +
          0.5 * (std::abs(srstar) - nuS * srstar);
      result.aux.vT1 = (alphaP * l.vy + alphaM * r.vy) / (alphaP + alphaM);
      result.aux.vT2 = (alphaP * l.vz + alphaM * r.vz) / (alphaP + alphaM);
    }

    UCTHLLDCons1D flux;
    if (sl >= 0.0) {
      flux = PhysicalFlux1D(l, ul, ptl, bxsq);
    } else if (sr <= 0.0) {
      flux = PhysicalFlux1D(r, ur, ptr, bxsq);
    } else {
      const Real ptstl = ptl + ul.d * sdl * (sm - l.vx);
      const Real ptstr = ptr + ur.d * sdr * (sm - r.vx);
      const Real ptst = 0.5 * (ptstr + ptstl);
      if (slstar >= 0.0) {
        const auto lst = StarState1D(l, ul, sdl, sdml, sm, ptl, ptst, bxsq);
        flux = PhysicalFlux1D(l, ul, ptl, bxsq);
        AddJump1D(flux, sl, lst.u, ul);
      } else if (srstar <= 0.0) {
        const auto rst = StarState1D(r, ur, sdr, sdmr, sm, ptr, ptst, bxsq);
        flux = PhysicalFlux1D(r, ur, ptr, bxsq);
        AddJump1D(flux, sr, rst.u, ur);
      } else {
        const auto lst = StarState1D(l, ul, sdl, sdml, sm, ptl, ptst, bxsq);
        const auto rst = StarState1D(r, ur, sdr, sdmr, sm, ptr, ptst, bxsq);
        const Real invsumd = 1.0 / (sqrtdl + sqrtdr);
        const Real bxsig = (bx > 0.0 ? 1.0 : -1.0);
        const Real lst_d_inv = 1.0 / lst.u.d;
        const Real rst_d_inv = 1.0 / rst.u.d;
        const Real vdst_y =
            invsumd * (sqrtdl * (lst.u.my * lst_d_inv) +
                       sqrtdr * (rst.u.my * rst_d_inv) +
                       bxsig * (rst.u.by - lst.u.by));
        const Real vdst_z =
            invsumd * (sqrtdl * (lst.u.mz * lst_d_inv) +
                       sqrtdr * (rst.u.mz * rst_d_inv) +
                       bxsig * (rst.u.bz - lst.u.bz));
        const Real bdst_y =
            invsumd * (sqrtdl * rst.u.by + sqrtdr * lst.u.by +
                       bxsig * sqrtdl * sqrtdr *
                           ((rst.u.my * rst_d_inv) - (lst.u.my * lst_d_inv)));
        const Real bdst_z =
            invsumd * (sqrtdl * rst.u.bz + sqrtdr * lst.u.bz +
                       bxsig * sqrtdl * sqrtdr *
                           ((rst.u.mz * rst_d_inv) - (lst.u.mz * lst_d_inv)));
        // Preserve the original left double-star momentum dot product, even
        // when selecting the right flux. Cancelling density changes rounding.
        const Real ldst_my = lst.u.d * vdst_y;
        const Real ldst_mz = lst.u.d * vdst_z;
        const Real vbdst =
            sm * bx + (ldst_my * bdst_y + ldst_mz * bdst_z) / lst.u.d;

        if (sm >= 0.0) {
          UCTHLLDCons1D ldst;
          ldst.d = lst.u.d;
          ldst.mx = lst.u.mx;
          ldst.my = ldst_my;
          ldst.mz = ldst_mz;
          ldst.by = bdst_y;
          ldst.bz = bdst_z;
          ldst.e = lst.u.e - sqrtdl * bxsig * (lst.vb - vbdst);
          flux = PhysicalFlux1D(l, ul, ptl, bxsq);
          AddJump1D(flux, sl, lst.u, ul);
          AddJump1D(flux, slstar, ldst, lst.u);
        } else {
          UCTHLLDCons1D rdst;
          rdst.d = rst.u.d;
          rdst.mx = rst.u.mx;
          rdst.my = rst.u.d * vdst_y;
          rdst.mz = rst.u.d * vdst_z;
          rdst.by = bdst_y;
          rdst.bz = bdst_z;
          rdst.e = rst.u.e + sqrtdr * bxsig * (rst.vb - vbdst);
          flux = PhysicalFlux1D(r, ur, ptr, bxsq);
          AddJump1D(flux, sr, rst.u, ur);
          AddJump1D(flux, srstar, rdst, rst.u);
        }
      }
    }

    result.flux.d = flux.d;
    result.flux.e = flux.e;
    if constexpr (DIR == X1DIR) {
      result.flux.m1 = flux.mx;
      result.flux.m2 = flux.my;
      result.flux.m3 = flux.mz;
      result.flux.b1 = 0.0;
      result.flux.b2 = flux.by;
      result.flux.b3 = flux.bz;
    } else if constexpr (DIR == X2DIR) {
      result.flux.m2 = flux.mx;
      result.flux.m3 = flux.my;
      result.flux.m1 = flux.mz;
      result.flux.b2 = 0.0;
      result.flux.b3 = flux.by;
      result.flux.b1 = flux.bz;
    } else {
      result.flux.m3 = flux.mx;
      result.flux.m1 = flux.my;
      result.flux.m2 = flux.mz;
      result.flux.b3 = 0.0;
      result.flux.b1 = flux.by;
      result.flux.b2 = flux.bz;
    }
    return result;
  }

  template <bool point_flux, typename FluxPack>
  static KOKKOS_INLINE_FUNCTION void
  Solve(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
        const int iu, const int ivx, const ScratchPad2D<Real> &wl,
        const ScratchPad2D<Real> &wr, 
        FluxPack &flux_out,
        VariablePack<Real> &uct_hlld,
        const AdiabaticCTMHDEOS &eos, const Real /*c_h*/) {
    const int ivy = IV1 + ((ivx - IV1) + 1) % 3;
    const int ivz = IV1 + ((ivx - IV1) + 2) % 3;
    const int iBx = ivx - 1 + NHYDRO;
    const int iBy = ivy - 1 + NHYDRO;
    const int iBz = ivz - 1 + NHYDRO;

    const auto gamma = eos.GetGamma();
    const auto gm1 = gamma - 1.0;
    const auto igm1 = 1.0 / gm1;

    // TODO(pgrete) move to a more central center and add logic
    constexpr int NCTMHD = 8;

    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      Real wli[NCTMHD], wri[NCTMHD], flxi[NCTMHD];
      Real spd[5];                     // signal speeds, left to right
      UCTHLLDCons1D ul, ur;                   // L/R states, conserved variables (computed)
      UCTHLLDCons1D ulst, uldst, urdst, urst; // Conserved variable for all states
      UCTHLLDCons1D fl, fr;                   // Fluxes for left & right states

      //--- Step 1.  Load L/R states into local variables

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

      flxi[IB1] = 0.0; // normal component of flux is 0.0
      Real bxi = 0.5 * (wli[IB1] + wri[IB1]);
      
      // Compute L/R states for selected conserved variables
      Real bxsq = bxi * bxi;
      // (KGF): group transverse vector components for floating-point associativity
      // symmetry
      Real pbl = 0.5 * (bxsq + (SQR(wli[IB2]) + SQR(wli[IB3]))); // magnetic pressure (l/r)
      Real pbr = 0.5 * (bxsq + (SQR(wri[IB2]) + SQR(wri[IB3])));
      Real kel = 0.5 * wli[IDN] * (SQR(wli[IV1]) + (SQR(wli[IV2]) + SQR(wli[IV3])));
      Real ker = 0.5 * wri[IDN] * (SQR(wri[IV1]) + (SQR(wri[IV2]) + SQR(wri[IV3])));

      ul.d = wli[IDN];
      ul.mx = wli[IV1] * ul.d;
      ul.my = wli[IV2] * ul.d;
      ul.mz = wli[IV3] * ul.d;
      ul.e = wli[IPR] * igm1 + kel + pbl;
      ul.by = wli[IB2];
      ul.bz = wli[IB3];

      ur.d = wri[IDN];
      ur.mx = wri[IV1] * ur.d;
      ur.my = wri[IV2] * ur.d;
      ur.mz = wri[IV3] * ur.d;
      ur.e = wri[IPR] * igm1 + ker + pbr;
      ur.by = wri[IB2];
      ur.bz = wri[IB3];

      //--- Step 2.  Compute L & R wave speeds according to Miyoshi & Kusano, eqn. (67)

      const auto cfl =
          eos.FastMagnetosonicSpeed(wli[IDN], wli[IPR], wli[IB1], wli[IB2], wli[IB3]);
      const auto cfr =
          eos.FastMagnetosonicSpeed(wri[IDN], wri[IPR], wri[IB1], wri[IB2], wri[IB3]);

      spd[0] = std::min(wli[IV1] - cfl, wri[IV1] - cfr); // lambda_L
      spd[4] = std::max(wli[IV1] + cfl, wri[IV1] + cfr); // lambda_R

      // Real cfmax = std::max(cfl,cfr);
      // if (wli[IV1] <= wri[IV1]) {
      //   spd[0] = wli[IV1] - cfmax;
      //   spd[4] = wri[IV1] + cfmax;
      // } else {
      //   spd[0] = wri[IV1] - cfmax;
      //   spd[4] = wli[IV1] + cfmax;
      // }

      //--- Step 3.  Compute L/R fluxes

      Real ptl = wli[IPR] + pbl; // total pressures L,R
      Real ptr = wri[IPR] + pbr;

      fl.d = ul.mx;
      fl.mx = ul.mx * wli[IV1] + ptl - bxsq;
      fl.my = ul.my * wli[IV1] - bxi * ul.by;
      fl.mz = ul.mz * wli[IV1] - bxi * ul.bz;
      fl.e = wli[IV1] * (ul.e + ptl - bxsq) - bxi * (wli[IV2] * ul.by + wli[IV3] * ul.bz);
      fl.by = ul.by * wli[IV1] - bxi * wli[IV2];
      fl.bz = ul.bz * wli[IV1] - bxi * wli[IV3];

      fr.d = ur.mx;
      fr.mx = ur.mx * wri[IV1] + ptr - bxsq;
      fr.my = ur.my * wri[IV1] - bxi * ur.by;
      fr.mz = ur.mz * wri[IV1] - bxi * ur.bz;
      fr.e = wri[IV1] * (ur.e + ptr - bxsq) - bxi * (wri[IV2] * ur.by + wri[IV3] * ur.bz);
      fr.by = ur.by * wri[IV1] - bxi * wri[IV2];
      fr.bz = ur.bz * wri[IV1] - bxi * wri[IV3];

      //--- Step 4.  Compute middle and Alfven wave speeds

      Real sdl = spd[0] - wli[IV1]; // S_i-u_i (i=L or R)
      Real sdr = spd[4] - wri[IV1];

      // S_M: eqn (38) of Miyoshi & Kusano
      // (KGF): group ptl, ptr terms for floating-point associativity symmetry
      spd[2] = (sdr * ur.mx - sdl * ul.mx + (ptl - ptr)) / (sdr * ur.d - sdl * ul.d); // lambda

      Real sdml = spd[0] - spd[2]; // S_i-S_M (i=L or R)
      Real sdmr = spd[4] - spd[2];
      Real sdml_inv = 1.0 / sdml;
      Real sdmr_inv = 1.0 / sdmr;
      // eqn (43) of Miyoshi & Kusano
      ulst.d = ul.d * sdl * sdml_inv;
      urst.d = ur.d * sdr * sdmr_inv;
      Real ulst_d_inv = 1.0 / ulst.d;
      Real urst_d_inv = 1.0 / urst.d;
      Real sqrtdl = std::sqrt(ulst.d);
      Real sqrtdr = std::sqrt(urst.d);

      // eqn (51) of Miyoshi & Kusano
      spd[1] = spd[2] - std::abs(bxi) / sqrtdl; // lambda*_L
      spd[3] = spd[2] + std::abs(bxi) / sqrtdr; // lambda*_R

      //--- Step 5.  Compute intermediate states
      // eqn (23) explicitly becomes eq (41) of Miyoshi & Kusano
      // TODO(felker): place an assertion that ptstl==ptstr
      Real ptstl = ptl + ul.d * sdl * (spd[2] - wli[IV1]);
      Real ptstr = ptr + ur.d * sdr * (spd[2] - wri[IV1]);
      // Real ptstl = ptl + ul.d*sdl*(sdl-sdml); // these equations had issues when
      // averaged Real ptstr = ptr + ur.d*sdr*(sdr-sdmr);
      Real ptst = 0.5 * (ptstr + ptstl); // total pressure (star state)

      // ul* - eqn (39) of M&K
      ulst.mx = ulst.d * spd[2];
      if (std::abs(ul.d * sdl * sdml - bxsq) < (SMALL_NUMBER)*ptst) {
        // Degenerate case
        ulst.my = ulst.d * wli[IV2];
        ulst.mz = ulst.d * wli[IV3];

        ulst.by = ul.by;
        ulst.bz = ul.bz;
      } else {
        // eqns (44) and (46) of M&K
        Real tmp = bxi * (sdl - sdml) / (ul.d * sdl * sdml - bxsq);
        ulst.my = ulst.d * (wli[IV2] - ul.by * tmp);
        ulst.mz = ulst.d * (wli[IV3] - ul.bz * tmp);

        // eqns (45) and (47) of M&K
        tmp = (ul.d * SQR(sdl) - bxsq) / (ul.d * sdl * sdml - bxsq);
        ulst.by = ul.by * tmp;
        ulst.bz = ul.bz * tmp;
      }
      // v_i* dot B_i*
      // (KGF): group transverse momenta terms for floating-point associativity symmetry
      Real vbstl = (ulst.mx * bxi + (ulst.my * ulst.by + ulst.mz * ulst.bz)) * ulst_d_inv;
      // eqn (48) of M&K
      // (KGF): group transverse by, bz terms for floating-point associativity symmetry
      ulst.e = (sdl * ul.e - ptl * wli[IV1] + ptst * spd[2] +
                bxi * (wli[IV1] * bxi + (wli[IV2] * ul.by + wli[IV3] * ul.bz) - vbstl)) *
               sdml_inv;

      // ur* - eqn (39) of M&K
      urst.mx = urst.d * spd[2];
      if (std::abs(ur.d * sdr * sdmr - bxsq) < (SMALL_NUMBER)*ptst) {
        // Degenerate case
        urst.my = urst.d * wri[IV2];
        urst.mz = urst.d * wri[IV3];

        urst.by = ur.by;
        urst.bz = ur.bz;
      } else {
        // eqns (44) and (46) of M&K
        Real tmp = bxi * (sdr - sdmr) / (ur.d * sdr * sdmr - bxsq);
        urst.my = urst.d * (wri[IV2] - ur.by * tmp);
        urst.mz = urst.d * (wri[IV3] - ur.bz * tmp);

        // eqns (45) and (47) of M&K
        tmp = (ur.d * SQR(sdr) - bxsq) / (ur.d * sdr * sdmr - bxsq);
        urst.by = ur.by * tmp;
        urst.bz = ur.bz * tmp;
      }
      // v_i* dot B_i*
      // (KGF): group transverse momenta terms for floating-point associativity symmetry
      Real vbstr = (urst.mx * bxi + (urst.my * urst.by + urst.mz * urst.bz)) * urst_d_inv;
      // eqn (48) of M&K
      // (KGF): group transverse by, bz terms for floating-point associativity symmetry
      urst.e = (sdr * ur.e - ptr * wri[IV1] + ptst * spd[2] +
                bxi * (wri[IV1] * bxi + (wri[IV2] * ur.by + wri[IV3] * ur.bz) - vbstr)) *
               sdmr_inv;
      // ul** and ur** - if Bx is near zero, same as *-states
      Real invsumd = 1.0 / (sqrtdl + sqrtdr);
      Real bxsig = (bxi > 0.0 ? 1.0 : -1.0);

      uldst.d = ulst.d;
      urdst.d = urst.d;

      uldst.mx = ulst.mx;
      urdst.mx = urst.mx;

      // eqn (59) of M&K
      Real tmp =
          invsumd * (sqrtdl * (ulst.my * ulst_d_inv) + sqrtdr * (urst.my * urst_d_inv) +
                     bxsig * (urst.by - ulst.by));
      uldst.my = uldst.d * tmp;
      urdst.my = urdst.d * tmp;

      // eqn (60) of M&K
      tmp = invsumd * (sqrtdl * (ulst.mz * ulst_d_inv) + sqrtdr * (urst.mz * urst_d_inv) +
                       bxsig * (urst.bz - ulst.bz));
      uldst.mz = uldst.d * tmp;
      urdst.mz = urdst.d * tmp;

      // eqn (61) of M&K
      tmp = invsumd *
            (sqrtdl * urst.by + sqrtdr * ulst.by +
             bxsig * sqrtdl * sqrtdr * ((urst.my * urst_d_inv) - (ulst.my * ulst_d_inv)));
      uldst.by = urdst.by = tmp;

      // eqn (62) of M&K
      tmp = invsumd *
            (sqrtdl * urst.bz + sqrtdr * ulst.bz +
             bxsig * sqrtdl * sqrtdr * ((urst.mz * urst_d_inv) - (ulst.mz * ulst_d_inv)));
      uldst.bz = urdst.bz = tmp;

      // eqn (63) of M&K
      tmp = spd[2] * bxi + (uldst.my * uldst.by + uldst.mz * uldst.bz) / uldst.d;
      uldst.e = ulst.e - sqrtdl * bxsig * (vbstl - tmp);
      urdst.e = urst.e + sqrtdr * bxsig * (vbstr - tmp);

      //--- Step 6.  Compute flux
      uldst.d = spd[1] * (uldst.d - ulst.d);
      uldst.mx = spd[1] * (uldst.mx - ulst.mx);
      uldst.my = spd[1] * (uldst.my - ulst.my);
      uldst.mz = spd[1] * (uldst.mz - ulst.mz);
      uldst.e = spd[1] * (uldst.e - ulst.e);
      uldst.by = spd[1] * (uldst.by - ulst.by);
      uldst.bz = spd[1] * (uldst.bz - ulst.bz);

      ulst.d = spd[0] * (ulst.d - ul.d);
      ulst.mx = spd[0] * (ulst.mx - ul.mx);
      ulst.my = spd[0] * (ulst.my - ul.my);
      ulst.mz = spd[0] * (ulst.mz - ul.mz);
      ulst.e = spd[0] * (ulst.e - ul.e);
      ulst.by = spd[0] * (ulst.by - ul.by);
      ulst.bz = spd[0] * (ulst.bz - ul.bz);

      urdst.d = spd[3] * (urdst.d - urst.d);
      urdst.mx = spd[3] * (urdst.mx - urst.mx);
      urdst.my = spd[3] * (urdst.my - urst.my);
      urdst.mz = spd[3] * (urdst.mz - urst.mz);
      urdst.e = spd[3] * (urdst.e - urst.e);
      urdst.by = spd[3] * (urdst.by - urst.by);
      urdst.bz = spd[3] * (urdst.bz - urst.bz);

      urst.d = spd[4] * (urst.d - ur.d);
      urst.mx = spd[4] * (urst.mx - ur.mx);
      urst.my = spd[4] * (urst.my - ur.my);
      urst.mz = spd[4] * (urst.mz - ur.mz);
      urst.e = spd[4] * (urst.e - ur.e);
      urst.by = spd[4] * (urst.by - ur.by);
      urst.bz = spd[4] * (urst.bz - ur.bz);

      if (spd[0] >= 0.0) {
        // return Fl if flow is supersonic
        flxi[IDN] = fl.d;
        flxi[IV1] = fl.mx;
        flxi[IV2] = fl.my;
        flxi[IV3] = fl.mz;
        flxi[IEN] = fl.e;
        flxi[IB2] = fl.by;
        flxi[IB3] = fl.bz;
      } else if (spd[4] <= 0.0) {
        // return Fr if flow is supersonic
        flxi[IDN] = fr.d;
        flxi[IV1] = fr.mx;
        flxi[IV2] = fr.my;
        flxi[IV3] = fr.mz;
        flxi[IEN] = fr.e;
        flxi[IB2] = fr.by;
        flxi[IB3] = fr.bz;
      } else if (spd[1] >= 0.0) {
        // return Fl*
        flxi[IDN] = fl.d + ulst.d;
        flxi[IV1] = fl.mx + ulst.mx;
        flxi[IV2] = fl.my + ulst.my;
        flxi[IV3] = fl.mz + ulst.mz;
        flxi[IEN] = fl.e + ulst.e;
        flxi[IB2] = fl.by + ulst.by;
        flxi[IB3] = fl.bz + ulst.bz;
      } else if (spd[3] <= 0.0) {
        // return Fr*
        flxi[IDN] = fr.d + urst.d;
        flxi[IV1] = fr.mx + urst.mx;
        flxi[IV2] = fr.my + urst.my;
        flxi[IV3] = fr.mz + urst.mz;
        flxi[IEN] = fr.e + urst.e;
        flxi[IB2] = fr.by + urst.by;
        flxi[IB3] = fr.bz + urst.bz;

      } else if (spd[2] >= 0.0) {
        // return Fl**
        flxi[IDN] = fl.d + ulst.d + uldst.d;
        flxi[IV1] = fl.mx + ulst.mx + uldst.mx;
        flxi[IV2] = fl.my + ulst.my + uldst.my;
        flxi[IV3] = fl.mz + ulst.mz + uldst.mz;
        flxi[IEN] = fl.e + ulst.e + uldst.e;
        flxi[IB2] = fl.by + ulst.by + uldst.by;
        flxi[IB3] = fl.bz + ulst.bz + uldst.bz;
      } else {
        // return Fr**
        flxi[IDN] = fr.d + urst.d + urdst.d;
        flxi[IV1] = fr.mx + urst.mx + urdst.mx;
        flxi[IV2] = fr.my + urst.my + urdst.my;
        flxi[IV3] = fr.mz + urst.mz + urdst.mz;
        flxi[IEN] = fr.e + urst.e + urdst.e;
        flxi[IB2] = fr.by + urst.by + urdst.by;
        flxi[IB3] = fr.bz + urst.bz + urdst.bz;
      }

      if constexpr (point_flux) {
        if (ivx == IV1) {
          flux_out(TE::F1, IDN, k, j, i) = flxi[IDN];
          flux_out(TE::F1, ivx, k, j, i) = flxi[IV1];
          flux_out(TE::F1, ivy, k, j, i) = flxi[IV2];
          flux_out(TE::F1, ivz, k, j, i) = flxi[IV3];
          flux_out(TE::F1, IEN, k, j, i) = flxi[IEN];
          flux_out(TE::F1, iBx, k, j, i) = flxi[IB1];
          flux_out(TE::F1, iBy, k, j, i) = flxi[IB2];
          flux_out(TE::F1, iBz, k, j, i) = flxi[IB3];
        } else if (ivx == IV2) {
          flux_out(TE::F2, IDN, k, j, i) = flxi[IDN];
          flux_out(TE::F2, ivx, k, j, i) = flxi[IV1];
          flux_out(TE::F2, ivy, k, j, i) = flxi[IV2];
          flux_out(TE::F2, ivz, k, j, i) = flxi[IV3];
          flux_out(TE::F2, IEN, k, j, i) = flxi[IEN];
          flux_out(TE::F2, iBx, k, j, i) = flxi[IB1];
          flux_out(TE::F2, iBy, k, j, i) = flxi[IB2];
          flux_out(TE::F2, iBz, k, j, i) = flxi[IB3];
        } else {
          flux_out(TE::F3, IDN, k, j, i) = flxi[IDN];
          flux_out(TE::F3, ivx, k, j, i) = flxi[IV1];
          flux_out(TE::F3, ivy, k, j, i) = flxi[IV2];
          flux_out(TE::F3, ivz, k, j, i) = flxi[IV3];
          flux_out(TE::F3, IEN, k, j, i) = flxi[IEN];
          flux_out(TE::F3, iBx, k, j, i) = flxi[IB1];
          flux_out(TE::F3, iBy, k, j, i) = flxi[IB2];
          flux_out(TE::F3, iBz, k, j, i) = flxi[IB3];
        }
      } else {
        flux_out.flux(ivx, IDN, k, j, i) = flxi[IDN];
        flux_out.flux(ivx, ivx, k, j, i) = flxi[IV1];
        flux_out.flux(ivx, ivy, k, j, i) = flxi[IV2];
        flux_out.flux(ivx, ivz, k, j, i) = flxi[IV3];
        flux_out.flux(ivx, IEN, k, j, i) = flxi[IEN];
        flux_out.flux(ivx, iBx, k, j, i) = flxi[IB1];
        flux_out.flux(ivx, iBy, k, j, i) = flxi[IB2];
        flux_out.flux(ivx, iBz, k, j, i) = flxi[IB3];
      }


      // following Mignone/Del Zanna (2021) [MDZ21]systematic construction of
      // UCT schemes for the UCT-HLLD scheme

      // Renaming to follow [MDZ21] notation more clearly (S stands for star *)
      Real lam_L = spd[0];
      Real lam_R = spd[4];
      Real lamS  = spd[2];
      Real lamS_L= spd[1];
      Real lamS_R= spd[3];

      // [MDZ21] equations (45) with eq. (46) degeneracy check for vS
      Real uct_eps = 1.0e-9;
      Real nuS;
      if (std::abs(lamS_R - lamS_L) > uct_eps * std::abs(lam_R - lam_L)){
        nuS = (std::abs(lamS_R) - std::abs(lamS_L)) / (lamS_R - lamS_L);
      } else {
        nuS = 0.0;
      }

      // [MDZ21] equation (45) for nuL and nuR
      Real nuL = (lamS_L + lam_L) / (std::abs(lamS_L) + std::abs(lam_L)); 
      Real nuR = (lamS_R + lam_R) / (std::abs(lamS_R) + std::abs(lam_R)); 

      // [MDZ21] equation for \tilde \chi L and R (above eq. 45)
      Real tchiL = ((wli[IV1] - lamS) * (lam_L - lamS)) / (lamS_L + lam_L - 2.0*lamS);
      Real tchiR = ((wri[IV1] - lamS) * (lam_R - lamS)) / (lamS_R + lam_R - 2.0*lamS);


      Real alphaP = std::max(0.0, lam_R);
      Real alphaM = -std::min(0.0,lam_L);
      
      Real aL = 0.5 * (1.0 + nuS);
      Real aR = 0.5 * (1.0 - nuS);
      Real dL = 0.5 * (nuL - nuS) * tchiL + 0.5 * (std::abs(lamS_L) -  nuS * lamS_L);
      Real dR = 0.5 * (nuR - nuS) * tchiR + 0.5 * (std::abs(lamS_R) -  nuS * lamS_R);
      Real vT1= (alphaP * wli[IV2] + alphaM * wri[IV2]) / (alphaP + alphaM);
      Real vT2= (alphaP * wli[IV3] + alphaM * wri[IV3]) / (alphaP + alphaM);

      if (ivx == 1){
        uct_hlld(TE::F1, AL, k, j, i) = aL;
        uct_hlld(TE::F1, AR, k, j, i) = aR;
        uct_hlld(TE::F1, DL, k, j, i) = dL;
        uct_hlld(TE::F1, DR, k, j, i) = dR;
        uct_hlld(TE::F1, VBART1, k, j, i) = vT1;
        uct_hlld(TE::F1, VBART2, k, j, i) = vT2;
      } else if (ivx == 2){
        uct_hlld(TE::F2, AL, k, j, i) = aL;
        uct_hlld(TE::F2, AR, k, j, i) = aR;
        uct_hlld(TE::F2, DL, k, j, i) = dL;
        uct_hlld(TE::F2, DR, k, j, i) = dR;
        uct_hlld(TE::F2, VBART1, k, j, i) = vT1;
        uct_hlld(TE::F2, VBART2, k, j, i) = vT2;
      } else if (ivx == 3){
        uct_hlld(TE::F3, AL, k, j, i) = aL;
        uct_hlld(TE::F3, AR, k, j, i) = aR;
        uct_hlld(TE::F3, DL, k, j, i) = dL;
        uct_hlld(TE::F3, DR, k, j, i) = dR;
        uct_hlld(TE::F3, VBART1, k, j, i) = vT1;
        uct_hlld(TE::F3, VBART2, k, j, i) = vT2;
      }



      

    });
  }
};
#endif // RSOLVERS_UCTHLLDMHD_HLLD_HPP_
