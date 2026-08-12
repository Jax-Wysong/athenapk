//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
// Athena++ astrophysical MHD code
// Copyright(C) 2014 James M. Stone <jmstone@princeton.edu> and other code contributors
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file field_loop.cpp
//! \brief Problem generator for advection of a field loop test.
//!
//! Can only be run in 2D or 3D.  Input parameters are:
//!  -  problem/rad   = radius of field loop
//!  -  problem/amp   = amplitude of vector potential (and therefore B)
//!  -  problem/vflow  = legacy flow velocity used for every coordinate direction
//!  -  problem/vflow1 = optional x1 flow velocity (defaults to vflow)
//!  -  problem/vflow2 = optional x2 flow velocity (defaults to vflow)
//!  -  problem/vflow3 = optional x3 flow velocity (defaults to vflow)
//!  -  problem/drat  = density ratio in loop, to test density advection and conduction
//! Each flow component is multiplied by the corresponding global domain size, so equal
//! component values give equal domain-crossing rates.
//!
//! Various test cases are possible:
//!  - (iprob=1): field loop in x1-x2 plane (cylinder in 3D)
//!  - (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
//!  - (iprob=3): field loop in x3-x1 plane (cylinder in 3D)
//!  - (iprob=4): rotated cylindrical field loop in 3D.
//!  - (iprob=5): spherical field loop in rotated plane
//!
//! REFERENCE: T. Gardiner & J.M. Stone, "An unsplit Godunov method for ideal MHD via
//! constrined transport", JCP, 205, 509 (2005)
//========================================================================================

// C headers

// C++ headers
#include <algorithm> // min, max
#include <array>
#include <cmath>     // sqrt()
#include <cstdio>    // fopen(), fprintf(), freopen()
#include <iostream>  // endl
#include <sstream>   // stringstream
#include <stdexcept> // runtime_error
#include <string>    // c_str()

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// Athena headers
#include "../../main.hpp"
#include "mhd_pgen_utils.hpp"
#include "outputs/outputs.hpp"

namespace field_loop {
using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface, ParameterInput *pin);

struct FieldLoopParameters {
  Real rad, amp, drat, gm1;
  Real v1, v2, v3;
  Real cos_a2, sin_a2, lambda;
  int iprob;
};

std::array<Real, 3> EvaluateVectorPotential(const Real x1, const Real x2,
                                            const Real x3,
                                            const FieldLoopParameters &params);
std::array<Real, IB3 + 1> EvaluatePointConserved(
    const Real x1, const Real x2, const Real x3,
    const FieldLoopParameters &params);

template <typename ConsHost, typename BfaceHost, typename EdgeHost>
void CurlEdgePotentialToFaceField(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
                                  EdgeHost &ax, EdgeHost &ay, EdgeHost &az);

template <typename ConsHost, typename BfaceHost>
void InitializeFourthOrderFieldLoop(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
                                    const FieldLoopParameters &params);

Real B0_ = 0.0;

// Relative divergence of B error, i.e., L * |div(B)| / |B_0|
// This is different from the standard package one because it uses
// a fixed B0, which is required in this pgen to get sensible results
// as some fraction of the domain has |B| = 0
Real RelDivBHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  auto B0 = B0_;

  pmb->par_reduce(
      "RelDivBHst", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &cons = cons_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        Real divb =
            (cons(IB1, k, j, i + 1) - cons(IB1, k, j, i - 1)) / coords.Dxc<1>(k, j, i) +
            (cons(IB2, k, j + 1, i) - cons(IB2, k, j - 1, i)) / coords.Dxc<2>(k, j, i);
        if (three_d) {
          divb +=
              (cons(IB3, k + 1, j, i) - cons(IB3, k - 1, j, i)) / coords.Dxc<3>(k, j, i);
        }
        lsum += 0.5 *
                (std::sqrt(SQR(coords.Dxc<1>(k, j, i)) + SQR(coords.Dxc<2>(k, j, i)) +
                           SQR(coords.Dxc<3>(k, j, i)))) *
                std::abs(divb) / B0 * coords.CellVolume(k, j, i);
      },
      sum);

  return sum;
}

// If using CT, this will be the relevant metric to check
Real FaceDivBHst(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");

  const auto &Bface_pack = md->PackVariables(std::vector<std::string>{"Bface"});
  const auto &cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  const bool three_d = cons_pack.GetNdim() == 3;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  Real sum = 0.0;
  auto B0 = B0_;

  pmb->par_reduce(
      "FaceDivBHst", 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lsum) {
        const auto &Bface = Bface_pack(b);
        const auto &coords = cons_pack.GetCoords(b);

        Real facedivb =
            (Bface(TE::F1,  0, k, j, i + 1) - Bface(TE::F1,  0, k, j, i)) / coords.Dxc<1>(k, j, i) +
            (Bface(TE::F2,  0, k, j + 1, i) - Bface(TE::F2,  0, k, j, i)) / coords.Dxc<2>(k, j, i);
        if (three_d) {
          facedivb +=
              (Bface(TE::F3,  0, k + 1, j, i) - Bface(TE::F3,  0, k, j, i)) / coords.Dxc<3>(k, j, i);
        }
        lsum += 0.5 *
                (std::sqrt(SQR(coords.Dxc<1>(k, j, i)) + SQR(coords.Dxc<2>(k, j, i)) +
                           SQR(coords.Dxc<3>(k, j, i)))) *
                std::abs(facedivb) / B0 * coords.CellVolume(k, j, i);
      },
      sum);

  return sum;
}

Real MaxFaceDivBHst(MeshData<Real> *md) {
  const auto &Bface_pack = md->PackVariables(std::vector<std::string>{"Bface"});
  const bool three_d = Bface_pack.GetNdim() == 3;

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  const Real B0 = B0_;
  Real max_facedivb = 0.0;

  Kokkos::parallel_reduce(
      "CPAWMaxFaceDivBHst",
      Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
          parthenon::DevExecSpace(), {0, kb.s, jb.s, ib.s},
          {Bface_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
          {1, 1, 1, ib.e + 1 - ib.s}),
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, Real &lmax) {
        const auto &Bface = Bface_pack(b);
        const auto &coords = Bface_pack.GetCoords(b);

        Real facedivb =
            (Bface(TE::F1, 0, k, j, i + 1) - Bface(TE::F1, 0, k, j, i)) /
                coords.Dxc<1>(k, j, i) +
            (Bface(TE::F2, 0, k, j + 1, i) - Bface(TE::F2, 0, k, j, i)) /
                coords.Dxc<2>(k, j, i);
        Real length_sq = SQR(coords.Dxc<1>(k, j, i)) +
                         SQR(coords.Dxc<2>(k, j, i));
        if (three_d) {
          facedivb +=
              (Bface(TE::F3, 0, k + 1, j, i) - Bface(TE::F3, 0, k, j, i)) /
              coords.Dxc<3>(k, j, i);
          length_sq += SQR(coords.Dxc<3>(k, j, i));
        }

        const Real normalized_divb = std::sqrt(length_sq) * std::abs(facedivb) / B0;
        lmax = Kokkos::fmax(lmax, normalized_divb);
      },
      Kokkos::Max<Real>(max_facedivb));

  return max_facedivb;
}

void ProblemInitPackageData(ParameterInput *pin, parthenon::StateDescriptor *pkg) {
  // gives us ctmhd ucthlldmhd or glmmhd
  const auto fluid = pkg->Param<Fluid>("fluid");
  auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
  if (fluid == Fluid::glmmhd){
    hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                      RelDivBHst, "UserRelDivB"));
    pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
    }
  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd){
    hst_vars.emplace_back(parthenon::HistoryOutputVar(parthenon::UserHistoryOperation::sum,
                                                    FaceDivBHst, "UserFaceDivB"));
    pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
    auto hst_vars = pkg->Param<parthenon::HstVar_list>(parthenon::hist_param_key);
    hst_vars.emplace_back(parthenon::HistoryOutputVar(
        parthenon::UserHistoryOperation::max, MaxFaceDivBHst, "MaxFaceDivB"));
    pkg->UpdateParam(parthenon::hist_param_key, hst_vars);
  } 
}

//----------------------------------------------------------------------------------------
//! \brief Evaluate the analytic vector potential for the CT field-loop variants.

std::array<Real, 3> EvaluateVectorPotential(
    const Real x1, const Real x2, const Real x3,
    const FieldLoopParameters &params) {
  if (params.iprob == 1) {
    const Real r = std::sqrt(x1 * x1 + x2 * x2);
    const Real potential = r < params.rad ? params.amp * (params.rad - r) : 0.0;
    return {0.0, 0.0, potential};
  }

  Real x = x1 * params.cos_a2 + x3 * params.sin_a2;
  while (x > 0.5 * params.lambda)
    x -= params.lambda;
  while (x < -0.5 * params.lambda)
    x += params.lambda;

  const Real r = std::sqrt(x * x + x2 * x2);
  const Real potential = r < params.rad ? params.amp * (params.rad - r) : 0.0;
  return {-params.sin_a2 * potential, 0.0,
          params.cos_a2 * potential};
}

//----------------------------------------------------------------------------------------
//! \brief Evaluate the complete analytic pointwise field-loop conserved state.

std::array<Real, IB3 + 1> EvaluatePointConserved(
    const Real x1, const Real x2, const Real x3,
    const FieldLoopParameters &params) {
  Real bx = 0.0;
  Real by = 0.0;
  Real bz = 0.0;

  Real x = x1;
  if (params.iprob == 4) {
    x = x1 * params.cos_a2 + x3 * params.sin_a2;
    while (x > 0.5 * params.lambda)
      x -= params.lambda;
    while (x < -0.5 * params.lambda)
      x += params.lambda;
  }

  const Real r = std::sqrt(x * x + x2 * x2);
  if (r > 0.0 && r < params.rad) {
    const Real bx_rot = -params.amp * x2 / r;
    by = params.amp * x / r;
    if (params.iprob == 1) {
      bx = bx_rot;
    } else {
      bx = bx_rot * params.cos_a2;
      bz = bx_rot * params.sin_a2;
    }
  }

  const Real r_density_sq = x1 * x1 + x2 * x2 + x3 * x3;
  const Real rho = r_density_sq < params.rad * params.rad ? params.drat : 1.0;

  std::array<Real, IB3 + 1> u_point{};
  u_point[IDN] = rho;
  u_point[IM1] = rho * params.v1;
  u_point[IM2] = rho * params.v2;
  u_point[IM3] = rho * params.v3;
  u_point[IEN] =
      1.0 / params.gm1 +
      0.5 * rho * (params.v1 * params.v1 + params.v2 * params.v2 +
                   params.v3 * params.v3) +
      0.5 * (bx * bx + by * by + bz * bz);
  u_point[IB1] = bx;
  u_point[IB2] = by;
  u_point[IB3] = bz;
  return u_point;
}

//----------------------------------------------------------------------------------------
//! \brief Take the discrete curl of line-averaged A and center the resulting B.

template <typename ConsHost, typename BfaceHost, typename EdgeHost>
void CurlEdgePotentialToFaceField(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
                                  EdgeHost &ax, EdgeHost &ay, EdgeHost &az) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = pmb->pmy_mesh->ndim < 3;
  auto &coords = pmb->coords;

  auto Bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto By_face = Bface.Get(IBF2, 0, 0, 0);
  auto Bz_face = Bface.Get(IBF3, 0, 0, 0);

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e + 1; ++i) {
        const Real da2_dx3 =
            two_d ? 0.0 : (ay(k + 1, j, i) - ay(k, j, i)) / coords.Dxc<3>(k);
        Bx_face(k, j, i) =
            (az(k, j + 1, i) - az(k, j, i)) / coords.Dxc<2>(j) - da2_dx3;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e + 1; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real da1_dx3 =
            two_d ? 0.0 : (ax(k + 1, j, i) - ax(k, j, i)) / coords.Dxc<3>(k);
        By_face(k, j, i) =
            da1_dx3 - (az(k, j, i + 1) - az(k, j, i)) / coords.Dxc<1>(i);
      }
    }
  }

  const int ku = two_d ? kb.e : kb.e + 1;
  for (int k = kb.s; k <= ku; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        Bz_face(k, j, i) =
            two_d ? 0.0
                  : (ay(k, j, i + 1) - ay(k, j, i)) / coords.Dxc<1>(i) -
                        (ax(k, j + 1, i) - ax(k, j, i)) / coords.Dxc<2>(j);
      }
    }
  }

  mhd_pgen_utils::ReconcileSelfPeriodicFaces(pmb, Bface);

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        u(IB1, k, j, i) =
            0.5 * (Bx_face(k, j, i) + Bx_face(k, j, i + 1));
        u(IB2, k, j, i) =
            0.5 * (By_face(k, j, i) + By_face(k, j + 1, i));
        u(IB3, k, j, i) =
            two_d ? 0.0
                  : 0.5 * (Bz_face(k, j, i) + Bz_face(k + 1, j, i));
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Initialize the nonsmooth field loop with fourth-order Gaussian quadrature.

template <typename ConsHost, typename BfaceHost>
void InitializeFourthOrderFieldLoop(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
                                    const FieldLoopParameters &params) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const bool two_d = pmb->pmy_mesh->ndim < 3;
  auto &coords = pmb->coords;

  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ax(
      "field loop line-averaged A1", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ay(
      "field loop line-averaged A2", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "field loop line-averaged A3", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  const Real gauss_offset = 0.5 / std::sqrt(3.0);
  const int kl = two_d ? kb.s : kb.s - 1;
  const int ku = two_d ? kb.e : kb.e + 1;
  for (int k = kl; k <= ku; ++k) {
    for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
      for (int i = ib.s - 1; i <= ib.e + 1; ++i) {
        const Real x1_e1 = coords.X<1, TE::E1>(k, j, i);
        const Real x2_e1 = coords.X<2, TE::E1>(k, j, i);
        const Real x3_e1 = coords.X<3, TE::E1>(k, j, i);
        const Real dx1 = gauss_offset * coords.Dxf<1>(i);
        const auto a1_m =
            EvaluateVectorPotential(x1_e1 - dx1, x2_e1, x3_e1, params);
        const auto a1_p =
            EvaluateVectorPotential(x1_e1 + dx1, x2_e1, x3_e1, params);
        ax(k, j, i) = 0.5 * (a1_m[0] + a1_p[0]);

        const Real x1_e2 = coords.X<1, TE::E2>(k, j, i);
        const Real x2_e2 = coords.X<2, TE::E2>(k, j, i);
        const Real x3_e2 = coords.X<3, TE::E2>(k, j, i);
        const Real dx2 = gauss_offset * coords.Dxf<2>(j);
        const auto a2_m =
            EvaluateVectorPotential(x1_e2, x2_e2 - dx2, x3_e2, params);
        const auto a2_p =
            EvaluateVectorPotential(x1_e2, x2_e2 + dx2, x3_e2, params);
        ay(k, j, i) = 0.5 * (a2_m[1] + a2_p[1]);

        const Real x1_e3 = coords.X<1, TE::E3>(k, j, i);
        const Real x2_e3 = coords.X<2, TE::E3>(k, j, i);
        const Real x3_e3 = coords.X<3, TE::E3>(k, j, i);
        if (two_d) {
          az(k, j, i) =
              EvaluateVectorPotential(x1_e3, x2_e3, x3_e3, params)[2];
        } else {
          const Real dx3 = gauss_offset * coords.Dxf<3>(k);
          const auto a3_m =
              EvaluateVectorPotential(x1_e3, x2_e3, x3_e3 - dx3, params);
          const auto a3_p =
              EvaluateVectorPotential(x1_e3, x2_e3, x3_e3 + dx3, params);
          az(k, j, i) = 0.5 * (a3_m[2] + a3_p[2]);
        }
      }
    }
  }

  const int nq3 = two_d ? 1 : 2;
  const Real weight = 1.0 / static_cast<Real>(4 * nq3);
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        std::array<Real, IB3 + 1> u_average{};
        for (int qk = 0; qk < nq3; ++qk) {
          const Real zq =
              two_d ? coords.Xc<3>(k)
                    : coords.Xc<3>(k) +
                          (qk == 0 ? -1.0 : 1.0) * gauss_offset *
                              coords.Dxf<3>(k);
          for (int qj = 0; qj < 2; ++qj) {
            const Real yq =
                coords.Xc<2>(j) + (qj == 0 ? -1.0 : 1.0) * gauss_offset *
                                          coords.Dxf<2>(j);
            for (int qi = 0; qi < 2; ++qi) {
              const Real xq =
                  coords.Xc<1>(i) + (qi == 0 ? -1.0 : 1.0) * gauss_offset *
                                              coords.Dxf<1>(i);
              const auto u_point = EvaluatePointConserved(xq, yq, zq, params);
              for (int n = IDN; n <= IEN; ++n) {
                u_average[n] += weight * u_point[n];
              }
            }
          }
        }
        for (int n = IDN; n <= IEN; ++n) {
          u(n, k, j, i) = u_average[n];
        }
      }
    }
  }

  CurlEdgePotentialToFaceField(pmb, u, Bface, ax, ay, az);
}

//========================================================================================
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//! \brief field loop advection problem generator for 2D/3D problems.
//========================================================================================

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  // gives us ctmhd ucthlldmhd or glmmhd
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const bool berta4 = mhd_pgen_utils::UseFourthOrderInitialization(pmb);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ax(
      "ax", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ay(
      "ay", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;

  // Read initial conditions, diffusion coefficients (if needed)
  Real rad = pin->GetReal("problem/field_loop", "rad");
  Real amp = pin->GetReal("problem/field_loop", "amp");
  B0_ = amp;
  Real vflow = pin->GetReal("problem/field_loop", "vflow");
  Real vflow1 = pin->GetOrAddReal("problem/field_loop", "vflow1", vflow);
  Real vflow2 = pin->GetOrAddReal("problem/field_loop", "vflow2", vflow);
  Real vflow3 = pin->GetOrAddReal("problem/field_loop", "vflow3", vflow);
  Real drat = pin->GetOrAddReal("problem/field_loop", "drat", 1.0);
  int iprob = pin->GetInteger("problem/field_loop", "iprob");
  Real ang_2, cos_a2(0.0), sin_a2(0.0), lambda(0.0);

  Real x1size =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  Real x2size =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);

  const bool two_d = pmb->pmy_mesh->ndim < 3;

  // for 2D sim set x3size to zero so that v_z is 0 below
  Real x3size =
      two_d ? 0
            : pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);

  // For (iprob=4) -- rotated cylinder in 3D -- set up rotation angle and wavelength
  if (iprob == 4) {

    // We put 1 wavelength in each direction.  Hence the wavelength
    //     lambda = x1size*cos_a;
    //     AND   lambda = x3size*sin_a;  are both satisfied.

    if (x1size == x3size) {
      // ang_2 = PI/4.0;  // unused variable
      cos_a2 = sin_a2 = std::sqrt(0.5);
    } else {
      ang_2 = std::atan(x1size / x3size);
      sin_a2 = std::sin(ang_2);
      cos_a2 = std::cos(ang_2);
    }
    // Use the larger angle to determine the wavelength
    if (cos_a2 >= sin_a2) {
      lambda = x1size * cos_a2;
    } else {
      lambda = x3size * sin_a2;
    }
  }

  const FieldLoopParameters params{
      rad, amp, drat, gm1,
      vflow1 * x1size, vflow2 * x2size, vflow3 * x3size,
      cos_a2, sin_a2, lambda, iprob};

  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  // manually defining loop bounds here to make vector potential work
  int kl, ku;
  if (two_d) {
    kl = 0;
    ku = 0;
  } else {
    kl = kb.s - 1;
    ku = kb.e + 1;
  }
  // if using ctmhd or ucthlldmhd, build az with the helper function
  // otherwise, if using glmmhd, build az like this
  if (fluid == Fluid::glmmhd){
  for (int k = kl; k <= ku; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        // (iprob=1): field loop in x1-x2 plane (cylinder in 3D) */
        if (iprob == 1) {
          ax(k, j, i) = 0.0;
          ay(k, j, i) = 0.0;
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j))) < rad * rad) {
            az(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j))));
          } else {
            az(k, j, i) = 0.0;
          }
        }

        // (iprob=2): field loop in x2-x3 plane (cylinder in 3D)
        if (iprob == 2) {
          if ((SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) < rad * rad) {
            ax(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))));
          } else {
            ax(k, j, i) = 0.0;
          }
          ay(k, j, i) = 0.0;
          az(k, j, i) = 0.0;
        }

        // (iprob=3): field loop in x3-x1 plane (cylinder in 3D)
        if (iprob == 3) {
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))) < rad * rad) {
            ay(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<3>(k))));
          } else {
            ay(k, j, i) = 0.0;
          }
          ax(k, j, i) = 0.0;
          az(k, j, i) = 0.0;
        }

        // (iprob=4): rotated cylindrical field loop in 3D.  Similar to iprob=1 with a
        // rotation about the x2-axis.  Define coordinate systems (x1,x2,x3) and (x,y,z)
        // with the following transformation rules:
        //    x =  x1*std::cos(ang_2) + x3*std::sin(ang_2)
        //    y =  x2
        //    z = -x1*std::sin(ang_2) + x3*std::cos(ang_2)
        // This inverts to:
        //    x1  = x*std::cos(ang_2) - z*std::sin(ang_2)
        //    x2  = y
        //    x3  = x*std::sin(ang_2) + z*std::cos(ang_2)

        if (iprob == 4) {
          Real x = coords.Xc<1>(i) * cos_a2 + coords.Xc<3>(k) * sin_a2;
          Real y = coords.Xc<2>(j);
          // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
          while (x > 0.5 * lambda)
            x -= lambda;
          while (x < -0.5 * lambda)
            x += lambda;
          if ((x * x + y * y) < rad * rad) {
            ax(k, j, i) = amp * (rad - std::sqrt(x * x + y * y)) * (-sin_a2);
          } else {
            ax(k, j, i) = 0.0;
          }
          ay(k, j, i) = 0.0;

          x = coords.Xc<1>(i) * cos_a2 + coords.Xc<3>(k) * sin_a2;
          y = coords.Xc<2>(j);
          // shift x back to the domain -0.5*lambda <= x <= 0.5*lambda
          while (x > 0.5 * lambda)
            x -= lambda;
          while (x < -0.5 * lambda)
            x += lambda;
          if ((x * x + y * y) < rad * rad) {
            az(k, j, i) = amp * (rad - std::sqrt(x * x + y * y)) * (cos_a2);
          } else {
            az(k, j, i) = 0.0;
          }
        }

        // (iprob=5): spherical field loop in rotated plane
        if (iprob == 5) {
          ax(k, j, i) = 0.0;
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) <
              rad * rad) {
            ay(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) +
                                       SQR(coords.Xc<3>(k))));
          } else {
            ay(k, j, i) = 0.0;
          }
          if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) <
              rad * rad) {
            az(k, j, i) =
                amp * (rad - std::sqrt(SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) +
                                       SQR(coords.Xc<3>(k))));
          } else {
            az(k, j, i) = 0.0;
          }
        }
      }
    }
  }
}

  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd){
    // fills u_cons() with the cell-averaged b values from
    // the face centered values made via the discrete
    // curl of the vector potential
    // Also deep copy the Bface vector for later evolution
    auto &u_dev_face = mbd->Get("Bface").data;
    auto Bface = u_dev_face.GetHostMirrorAndCopy();

    if (berta4) {
      PARTHENON_REQUIRE_THROWS(
          iprob == 1 || iprob == 4,
          "Fourth-order field-loop initialization currently supports iprob=1 and iprob=4");
      InitializeFourthOrderFieldLoop(pmb, u, Bface, params);
    } else {
      Bface_Fill_Cons(pmb, u, Bface, pin);
    }
    u_dev_face.DeepCopy(Bface);
  }

  // Initialize density and momenta.  If drat != 1, then density and temperature will be
  // different inside loop than background values
  if (!berta4) {
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
        u(IDN, k, j, i) = 1.0;
        if ((SQR(coords.Xc<1>(i)) + SQR(coords.Xc<2>(j)) + SQR(coords.Xc<3>(k))) <
            rad * rad) {
          u(IDN, k, j, i) = drat;
        }
        u(IM1, k, j, i) = u(IDN, k, j, i) * vflow1 * x1size;
        u(IM2, k, j, i) = u(IDN, k, j, i) * vflow2 * x2size;
        u(IM3, k, j, i) = u(IDN, k, j, i) * vflow3 * x3size;
        
        if (fluid == Fluid::glmmhd){
          Real aydz =
              two_d ? 0.0 : (ay(k + 1, j, i) - ay(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
          Real axdz =
              two_d ? 0.0 : (ax(k + 1, j, i) - ax(k - 1, j, i)) / coords.Dxc<3>(k) / 2.0;
          
          u(IB1, k, j, i) =
              (az(k, j + 1, i) - az(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0 - aydz;
          u(IB2, k, j, i) =
              axdz - (az(k, j, i + 1) - az(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0;
          u(IB3, k, j, i) = (ay(k, j, i + 1) - ay(k, j, i - 1)) / coords.Dxc<1>(i) / 2.0 -
                            (ax(k, j + 1, i) - ax(k, j - 1, i)) / coords.Dxc<2>(j) / 2.0;
          u(IPS, k, j, i) = 0.0;
        }

        // this will fill properly since it is done after 
        // B field initialization regardless of glmmhd or ct
        u(IEN, k, j, i) =
            1.0 / gm1 +
            0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) + SQR(u(IB3, k, j, i))) +
            0.5 * (SQR(u(IM1, k, j, i)) + SQR(u(IM2, k, j, i)) + SQR(u(IM3, k, j, i))) /
                u(IDN, k, j, i);
        }
      }
    }
  }
  // copy initialized vars to device
  u_dev.DeepCopy(u);
}

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface, ParameterInput *pin) {

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  // Athena++'s pbval->nblevel equivalent. Multiple finer neighbors can occupy the same
  // offset, but they are all at the same level on a properly nested Parthenon mesh.
  const int level = pmb->loc.level();
  std::array<int, 27> neighbor_levels;
  neighbor_levels.fill(level);
  for (const auto &neighbor : pmb->GetNeighbors()) {
    const int ox1 = neighbor.offsets(X1DIR);
    const int ox2 = neighbor.offsets(X2DIR);
    const int ox3 = neighbor.offsets(X3DIR);
    const int idx = (ox1 + 1) + 3 * ((ox2 + 1) + 3 * (ox3 + 1));
    neighbor_levels[idx] = std::max(neighbor_levels[idx], neighbor.loc.level());
  }
 
  // always needed
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ax(
      "ax", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ay(
      "ay", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  Real rad = pin->GetReal("problem/field_loop", "rad");
  Real amp = pin->GetReal("problem/field_loop", "amp");
  int iprob = pin->GetInteger("problem/field_loop", "iprob");

  Real ang_2, cos_a2(0.0), sin_a2(0.0), lambda(0.0);

  Real x1size =
      pmb->pmy_mesh->mesh_size.xmax(X1DIR) - pmb->pmy_mesh->mesh_size.xmin(X1DIR);
  Real x2size =
      pmb->pmy_mesh->mesh_size.xmax(X2DIR) - pmb->pmy_mesh->mesh_size.xmin(X2DIR);

  const bool two_d = pmb->pmy_mesh->ndim < 3;

  auto finer_neighbor = [&](const int ox1, const int ox2, const int ox3) {
    const int idx = (ox1 + 1) + 3 * ((ox2 + 1) + 3 * (ox3 + 1));
    return neighbor_levels[idx] > level;
  };

  // Return true when an edge lies on a coarse/fine interface. The neighbor offset along
  // the edge must be zero, while each nonzero transverse offset must coincide with the
  // corresponding lower or upper block boundary.
  auto edge_touches_finer_neighbor = [&](const int edge_dir, const int k, const int j,
                                         const int i) {
    if (two_d) return false;

    const std::array<int, 3> edge_idx{i, j, k};
    const std::array<int, 3> lower{ib.s, jb.s, kb.s};
    const std::array<int, 3> upper{ib.e + 1, jb.e + 1, kb.e + 1};

    for (int ox3 = -1; ox3 <= 1; ++ox3) {
      for (int ox2 = -1; ox2 <= 1; ++ox2) {
        for (int ox1 = -1; ox1 <= 1; ++ox1) {
          const std::array<int, 3> offset{ox1, ox2, ox3};
          if (offset[edge_dir - 1] != 0) continue;
          if (ox1 == 0 && ox2 == 0 && ox3 == 0) continue;
          if (!finer_neighbor(ox1, ox2, ox3)) continue;

          bool touches = true;
          for (int dir = 0; dir < 3; ++dir) {
            if (dir == edge_dir - 1 || offset[dir] == 0) continue;
            if ((offset[dir] < 0 && edge_idx[dir] != lower[dir]) ||
                (offset[dir] > 0 && edge_idx[dir] != upper[dir])) {
              touches = false;
              break;
            }
          }
          if (touches) return true;
        }
      }
    }
    return false;
  };

  // for 2D sim set x3size to zero so that v_z is 0 below
  Real x3size =
      two_d ? 0
            : pmb->pmy_mesh->mesh_size.xmax(X3DIR) - pmb->pmy_mesh->mesh_size.xmin(X3DIR);

  // For (iprob=4) -- rotated cylinder in 3D -- set up rotation angle and wavelength
  if (iprob == 4) {

    // We put 1 wavelength in each direction.  Hence the wavelength
    //     lambda = x1size*cos_a;
    //     AND   lambda = x3size*sin_a;  are both satisfied.

    if (x1size == x3size) {
      // ang_2 = PI/4.0;  // unused variable
      cos_a2 = sin_a2 = std::sqrt(0.5);
    } else {
      ang_2 = std::atan(x1size / x3size);
      sin_a2 = std::sin(ang_2);
      cos_a2 = std::cos(ang_2);
    }
    // Use the larger angle to determine the wavelength
    if (cos_a2 >= sin_a2) {
      lambda = x1size * cos_a2;
    } else {
      lambda = x3size * sin_a2;
    }
  }

  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  auto rotated_loop_potential = [&](const Real x1, const Real x2, const Real x3,
                                    const Real component_factor) {
    Real x = x1 * cos_a2 + x3 * sin_a2;
    while (x > 0.5 * lambda)
      x -= lambda;
    while (x < -0.5 * lambda)
      x += lambda;

    const Real r2 = x * x + x2 * x2;
    return (r2 < rad * rad) ? amp * (rad - std::sqrt(r2)) * component_factor : 0.0;
  };

  // create corner valued vector potential
  int kl, ku;
  if (two_d) {
    kl = kb.s;
    ku = kb.e;
  } else {
    kl = kb.s - 1;
    ku = kb.e + 1;
  }
  for (int k = kl; k <= ku; k++) {
    for (int j = jb.s - 1; j <= jb.e + 1; j++) {
      for (int i = ib.s - 1; i <= ib.e + 1; i++) {
        // define az on the proper edges
        Real x = coords.X<1, TE::E3>(k,j,i);
        Real y = coords.X<2, TE::E3>(k,j,i);
        // (iprob=1): field loop in x1-x2 plane (cylinder in 3D) */
        if (iprob == 1) {
          ax(k, j, i) = 0.0;
          ay(k, j, i) = 0.0;
          Real R2 = SQR(x) + SQR(y);
            if (R2 < rad * rad) {
              az(k, j, i) =
                  amp * (rad - std::sqrt(R2));
            } else {
              az(k, j, i) = 0.0;
            }
          }

        // (iprob=4): rotated cylindrical field loop in 3D.  Similar to iprob=1 with a
        // rotation about the x2-axis.  Define coordinate systems (x1,x2,x3) and (x,y,z)
        // with the following transformation rules:
        //    x =  x1*std::cos(ang_2) + x3*std::sin(ang_2)
        //    y =  x2
        //    z = -x1*std::sin(ang_2) + x3*std::cos(ang_2)
        // This inverts to:
        //    x1  = x*std::cos(ang_2) - z*std::sin(ang_2)
        //    x2  = y
        //    x3  = x*std::sin(ang_2) + z*std::cos(ang_2)

        if (iprob == 4) {
          // Ax component lives on E1
          Real x1_ax = coords.X<1, TE::E1>(k,j,i);
          Real x2_ax = coords.X<2, TE::E1>(k,j,i);
          Real x3_ax = coords.X<3, TE::E1>(k,j,i);

          // Az component lives on E3
          Real x1_az = coords.X<1, TE::E3>(k,j,i);
          Real x2_az = coords.X<2, TE::E3>(k,j,i);
          Real x3_az = coords.X<3, TE::E3>(k,j,i);

          if (edge_touches_finer_neighbor(X1DIR, k, j, i)) {
            const Real quarter_dx1 = 0.25 * coords.Dxf<1>(i);
            ax(k, j, i) =
                0.5 *
                (rotated_loop_potential(x1_ax - quarter_dx1, x2_ax, x3_ax, -sin_a2) +
                 rotated_loop_potential(x1_ax + quarter_dx1, x2_ax, x3_ax, -sin_a2));
          } else {
            ax(k, j, i) = rotated_loop_potential(x1_ax, x2_ax, x3_ax, -sin_a2);
          }
          ay(k, j, i) = 0.0;

          if (edge_touches_finer_neighbor(X3DIR, k, j, i)) {
            const Real quarter_dx3 = 0.25 * coords.Dxf<3>(k);
            az(k, j, i) =
                0.5 *
                (rotated_loop_potential(x1_az, x2_az, x3_az - quarter_dx3, cos_a2) +
                 rotated_loop_potential(x1_az, x2_az, x3_az + quarter_dx3, cos_a2));
          } else {
            az(k, j, i) = rotated_loop_potential(x1_az, x2_az, x3_az, cos_a2);
          }
        }

        }
      }
    }

  CurlEdgePotentialToFaceField(pmb, u, Bface, ax, ay, az);
}

} // namespace field_loop
