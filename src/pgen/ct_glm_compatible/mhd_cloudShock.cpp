//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the 3-clause BSD License, see LICENSE file for details
//========================================================================================
//! \file mhd_cloudShock.cpp
//! \brief Problem generator for 3D cloud-shock interaction following Berta et al. (2024)
//!

// C++ headers
#include <algorithm> // min, max
#include <cmath>     // log
#include <cstring>   // strcmp()
#include <memory>

// Parthenon headers
#include "bvals/boundary_conditions_generic.hpp"
#include "mesh/mesh.hpp"
#include <basic_types.hpp>
#include <iomanip>
#include <ios>
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>
#include <random>
#include <sstream>

// AthenaPK headers
#include "../../main.hpp"
#include "mhd_pgen_utils.hpp"

namespace mhd_cloudShock {
using namespace parthenon::driver::prelude;
using TE = parthenon::TopologicalElement;

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface, ParameterInput *pin);

std::array<Real, IB3 + 1> EvaluatePointConserved(
    const Real x1, const Real x2, const Real x3, const Real gm1);

template <typename ConsHost, typename BfaceHost>
void InitializeFourthOrderCloudShock(MeshBlock *pmb, ParameterInput *pin, ConsHost &u, BfaceHost &Bface);

// x < 0.6
Real rhoL = 3.86859;
Real vxL  = 0.0;
Real vyL  = 0.0;
Real vzL  = 0.0;
Real pL   = 167.345; 
Real BxL  = 0.0;
Real ByL  = 2.1826182;
Real BzL  = -2.1826182;

// x > 0.6
Real rhoR = 1.0;
Real vxR  = -11.2536;
Real vyR  = 0.0;
Real vzR  = 0.0;
Real pR   = 1.0;
Real BxR  = 0.0;
Real ByR  = 0.56418958;
Real BzR  = 0.56418958;

Real rho_OD = 10.0;
Real cloud_r= 0.15;

bool IsInsideCloud(const Real x1, const Real x2, const Real x3) {
  const Real radius_sq = SQR(x1 - 0.8) + SQR(x2 - 0.5) + SQR(x3 - 0.5);
  return radius_sq <= SQR(cloud_r);
}

Real PlanarFaceAverage(const Real left_state, const Real right_state,
                       const Real x_left, const Real x_right) {
  constexpr Real shock_x = 0.6;
  const Real left_fraction =
      std::clamp((shock_x - x_left) / (x_right - x_left), 0.0, 1.0);
  return left_fraction * left_state + (1.0 - left_fraction) * right_state;
}

void FixedInnerX1(std::shared_ptr<MeshBlockData<Real>> &mbd, bool coarse) {
  using parthenon::BoundaryFunction::BCSide;
  using parthenon::BoundaryFunction::BCType;

  // This callback is also invoked for the detector-only shallow container. Give every
  // field a valid zero-gradient ghost state first, then replace the cloud-shock state
  // fields when they are present in the current container.
  parthenon::BoundaryFunction::GenericBC<
      parthenon::X1DIR, BCSide::Inner, BCType::Outflow,
      parthenon::variable_names::any>(mbd, coarse);

  auto pmb = mbd->GetBlockPointer();
  const Real gm1 = pmb->packages.Get("Hydro")->Param<Real>("AdiabaticIndex") - 1.0;
  const Real rho_left = rhoL;
  const Real vx_left = vxL;
  const Real vy_left = vyL;
  const Real vz_left = vzL;
  const Real bx_left = BxL;
  const Real by_left = ByL;
  const Real bz_left = BzL;
  const Real energy_left =
      pL / gm1 +
      0.5 * rho_left *
          (vx_left * vx_left + vy_left * vy_left + vz_left * vz_left) +
      0.5 * (bx_left * bx_left + by_left * by_left + bz_left * bz_left);
  const bool fine = false;
  const auto nb = IndexRange{0, 0};

  const auto impose_cons_state = [&](const std::string &name) {
    if (!mbd->HasVariable(name)) return;

    auto cons = mbd->PackVariables(std::vector<std::string>{name}, coarse);
    pmb->par_for_bndry(
        "CloudShockFixedInnerX1_" + name, nb, IndexDomain::inner_x1, TE::CC,
        coarse, fine,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          cons(IDN, k, j, i) = rho_left;
          cons(IM1, k, j, i) = rho_left * vx_left;
          cons(IM2, k, j, i) = rho_left * vy_left;
          cons(IM3, k, j, i) = rho_left * vz_left;
          cons(IEN, k, j, i) = energy_left;
          cons(IB1, k, j, i) = bx_left;
          cons(IB2, k, j, i) = by_left;
          cons(IB3, k, j, i) = bz_left;
        });
  };

  const auto impose_face_state = [&](const std::string &name) {
    if (!mbd->HasVariable(name)) return;

    auto bface = mbd->PackVariables(std::vector<std::string>{name}, coarse);
    pmb->par_for_bndry(
        "CloudShockFixedInnerX1_" + name + "_B1", nb, IndexDomain::inner_x1,
        TE::F1, coarse, fine,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          bface(TE::F1, 0, k, j, i) = bx_left;
        });
    pmb->par_for_bndry(
        "CloudShockFixedInnerX1_" + name + "_B2", nb, IndexDomain::inner_x1,
        TE::F2, coarse, fine,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          bface(TE::F2, 0, k, j, i) = by_left;
        });
    pmb->par_for_bndry(
        "CloudShockFixedInnerX1_" + name + "_B3", nb, IndexDomain::inner_x1,
        TE::F3, coarse, fine,
        KOKKOS_LAMBDA(const int &, const int &k, const int &j, const int &i) {
          bface(TE::F3, 0, k, j, i) = bz_left;
        });
  };

  impose_cons_state("cons");
  impose_cons_state("cons_point");
  impose_face_state("Bface");
  impose_face_state("Bface_point");
}


//----------------------------------------------------------------------------------------
//! \brief Initialize the nonsmooth cloud-shock problem with Gaussian quadrature.

template <typename ConsHost, typename BfaceHost>
void InitializeFourthOrderCloudShock(MeshBlock *pmb, ParameterInput *pin, ConsHost &u, BfaceHost &Bface) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &coords = pmb->coords;

  auto gamma = pin->GetReal("hydro", "gamma");
  auto gm1 = (gamma - 1.0);

  const Real gauss_offset = 0.5 / std::sqrt(3.0);
  const int nq3 = 2;
  const Real weight = 1.0 / static_cast<Real>(4 * nq3);
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        std::array<Real, IB3 + 1> u_average{};
        for (int qk = 0; qk < nq3; ++qk) {
          const Real zq =
              coords.Xc<3>(k) +
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
              const auto u_point = EvaluatePointConserved(xq, yq, zq, gm1);
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

  // The magnetic field only varies across the planar discontinuity at x = 0.6.
  // B1 is constant on each x-face, while the B2 and B3 face areas span one x-cell.
  // Integrate that piecewise-constant profile exactly instead of sampling it with
  // Gaussian quadrature.
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e + 1; ++i) {
        const Real x_face = coords.Xf<1>(i);
        Bface(IBF1, k, j, i) = x_face < 0.6 ? BxL : BxR;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e + 1; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        Bface(IBF2, k, j, i) =
            PlanarFaceAverage(ByL, ByR, coords.Xf<1>(i), coords.Xf<1>(i + 1));
      }
    }
  }

  for (int k = kb.s; k <= kb.e + 1; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        Bface(IBF3, k, j, i) =
            PlanarFaceAverage(BzL, BzR, coords.Xf<1>(i), coords.Xf<1>(i + 1));
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        u(IB1, k, j, i) =
            0.5 * (Bface(IBF1, k, j, i) + Bface(IBF1, k, j, i + 1));
        u(IB2, k, j, i) =
            0.5 * (Bface(IBF2, k, j, i) + Bface(IBF2, k, j + 1, i));
        u(IB3, k, j, i) =
            0.5 * (Bface(IBF3, k, j, i) + Bface(IBF3, k + 1, j, i));
      }
    }
  }




}

//----------------------------------------------------------------------------------------
//! \brief Evaluate the complete analytic pointwise field-loop conserved state.

std::array<Real, IB3 + 1> EvaluatePointConserved(
    const Real x1, const Real x2, const Real x3, const Real gm1) {

    const bool over_dense_region = IsInsideCloud(x1, x2, x3);

    std::array<Real, IB3 + 1> u_point{};
    Real rho;
    if (over_dense_region){
      rho = rho_OD;
    } else {
      rho = x1 < 0.6 ? rhoL : rhoR;
    }

    Real vx = x1 < 0.6 ? vxL : vxR;
    Real vy = x1 < 0.6 ? vyL : vyR;
    Real vz = x1 < 0.6 ? vzL : vzR;

    u_point[IDN] = rho;
    u_point[IM1] = vx*rho;
    u_point[IM2] = vy*rho;
    u_point[IM3] = vz*rho;

    Real p = x1 < 0.6 ? pL : pR;
    Real E = p/(gm1) + 0.5*rho*(vx*vx + vy*vy + vz*vz);
    u_point[IEN] = E;

    Real bx = x1 < 0.6 ? BxL : BxR;
    Real by = x1 < 0.6 ? ByL : ByR;
    Real bz = x1 < 0.6 ? BzL : BzR;

    u_point[IEN] += 0.5 * (bx*bx + by*by + bz*bz);


  return u_point;
}

//----------------------------------------------------------------------------------------
//! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
//  \brief Problem Generator for the cloud in wind setup

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  auto ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  auto jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  auto kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

 const bool berta4 = mhd_pgen_utils::UseFourthOrderInitialization(pmb);


  auto gamma = pin->GetReal("hydro", "gamma");
  auto gm1 = (gamma - 1.0);

  // initialize conserved variables
  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  auto &coords = pmb->coords;
  // initializing on host
  auto u = u_dev.GetHostMirrorAndCopy();

  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd){
    auto &u_dev_face = mbd->Get("Bface").data;
    auto Bface = u_dev_face.GetHostMirrorAndCopy();

    if (berta4) {
      InitializeFourthOrderCloudShock(pmb, pin, u, Bface);
    } else {
      Bface_Fill_Cons(pmb, u, Bface, pin);
    }
    
    u_dev_face.DeepCopy(Bface);
  }


  // The fourth-order branch has already supplied cell-volume averages. Preserve
  // those values instead of replacing them with samples at cell centers.
  if (!berta4) {
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          const Real x = coords.Xc<1>(i);
          const Real y = coords.Xc<2>(j);
          const Real z = coords.Xc<3>(k);

          const bool over_dense_region = IsInsideCloud(x, y, z);

          Real rho;
          if (over_dense_region){
            rho = rho_OD;
          } else {
            rho = x < 0.6 ? rhoL : rhoR;
          }

          Real vx = x < 0.6 ? vxL : vxR;
          Real vy = x < 0.6 ? vyL : vyR;
          Real vz = x < 0.6 ? vzL : vzR;

          u(IDN, k, j, i) = rho;
          u(IM1, k, j, i) = vx*rho;
          u(IM2, k, j, i) = vy*rho;
          u(IM3, k, j, i) = vz*rho;

          Real p = x < 0.6 ? pL : pR;
          Real E = p/(gm1) + 0.5*rho*(vx*vx + vy*vy + vz*vz);
          u(IEN, k, j, i) = E;

          if (fluid == Fluid::glmmhd){
            u(IB1, k, j, i) = x < 0.6 ? BxL : BxR;
            u(IB2, k, j, i) = x < 0.6 ? ByL : ByR;
            u(IB3, k, j, i) = x < 0.6 ? BzL : BzR;
          }
          // else its ct and u(IB1:IB3) already filled up
          u(IEN, k, j, i) +=
              0.5 * (u(IB1, k, j, i) * u(IB1, k, j, i) +
                     u(IB2, k, j, i) * u(IB2, k, j, i) +
                     u(IB3, k, j, i) * u(IB3, k, j, i));
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

  // Use vector potential to initialize field loop
  auto &coords = pmb->coords;

  auto Bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto By_face = Bface.Get(IBF2, 0, 0, 0);
  auto Bz_face = Bface.Get(IBF3, 0, 0, 0);

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e + 1; ++i) {
        const Real x = coords.Xc<1>(i);
        Bx_face(k, j, i) = x < 0.6 ? BxL : BxR;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e + 1; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real x = coords.Xc<1>(i);
        By_face(k, j, i) = x < 0.6 ? ByL : ByR;
      }
    }
  }

  const int ku = kb.e + 1;
  for (int k = kb.s; k <= ku; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real x = coords.Xc<1>(i);
        Bz_face(k, j, i) = x < 0.6 ? BzL : BzR;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        u(IB1, k, j, i) =
            0.5 * (Bx_face(k, j, i) + Bx_face(k, j, i + 1));
        u(IB2, k, j, i) =
            0.5 * (By_face(k, j, i) + By_face(k, j + 1, i));
        u(IB3, k, j, i) =
            0.5 * (Bz_face(k, j, i) + Bz_face(k + 1, j, i));
      }
    }
  }

}


} // namespace mhd_cloudShock
