//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file smooth_mhd_vortex.cpp
//! \brief Problem generator for (currently 2D) smooth mhd vortex convergence test
// 
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

namespace smooth_mhd_vortex {
  using namespace parthenon::package::prelude;
  using TE = parthenon::TopologicalElement;

  template <typename ConsHost, typename BfaceHost>
  void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface);

  std::array<Real, IB3 + 1> EvaluatePointConserved(
      const Real x1, const Real x2, const Real x3, const Real gm1);
  std::array<Real, 3> EvaluateVectorPotential(
      const Real x1, const Real x2, const Real x3);


Real B0_ = 0.0;

Real VortexAmplitude() { return 1.0 / (2.0 * M_PI); }

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
  } 
}

void UserWorkAfterLoop(Mesh *mesh, ParameterInput *pin, parthenon::SimTime &tm) {
  if (!pin->GetOrAddBoolean("problem/smooth_mhd_vortex", "compute_error", false)) return;

  constexpr int NMHD = 8;

  // Initialize errors to zero
  Real l1_err[NMHD]{}, max_err[NMHD]{};

  for (auto &pmb : mesh->block_list) {
    const auto hydro_pkg = pmb->packages.Get("Hydro");
    const auto fluid = hydro_pkg->Param<Fluid>("fluid");
    const bool fourth_order_init =
        mhd_pgen_utils::UseFourthOrderFVInitialization(pmb.get());

    IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    const Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
    // Save analytic solution of conserved variables in 4D scratch array on host
    Kokkos::View<Real ****, parthenon::LayoutWrapper, parthenon::HostMemSpace> cons_(
        "cons scratch", NMHD, pmb->cellbounds.ncellsk(IndexDomain::entire),
        pmb->cellbounds.ncellsj(IndexDomain::entire),
        pmb->cellbounds.ncellsi(IndexDomain::entire));

    B0_ = VortexAmplitude();
    auto &mbd = pmb->meshblock_data.Get();
    if (fourth_order_init) {
      auto &u_dev_face = mbd->Get("Bface").data;
      auto Bface_ref = u_dev_face.GetHostMirrorAndCopy();
      const auto evaluate_point_cons =
          [gm1](const Real x1, const Real x2, const Real x3) {
            return EvaluatePointConserved(x1, x2, x3, gm1);
          };
      mhd_pgen_utils::InitializeFourthOrderSmoothMHD(
          pmb.get(), cons_, Bface_ref, evaluate_point_cons,
          EvaluateVectorPotential);
    } else {
      if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd) {
        // Construct the legacy second-order CT magnetic reference.
        auto &u_dev_face = mbd->Get("Bface").data;
        auto Bface_ref = u_dev_face.GetHostMirrorAndCopy();
        Bface_Fill_Cons(pmb.get(), cons_, Bface_ref);
      }

      auto &coords = pmb->coords;
      for (int k = kb.s; k <= kb.e; k++) {
        for (int j = jb.s; j <= jb.e; j++) {
          for (int i = ib.s; i <= ib.e; i++) {
            const auto point_cons =
                EvaluatePointConserved(coords.Xc<1>(i), coords.Xc<2>(j),
                                       coords.Xc<3>(k), gm1);
            for (int n = IDN; n <= IEN; ++n) {
              cons_(n, k, j, i) = point_cons[n];
            }

            if (fluid == Fluid::glmmhd) {
              cons_(IB1, k, j, i) = point_cons[IB1];
              cons_(IB2, k, j, i) = point_cons[IB2];
              cons_(IB3, k, j, i) = point_cons[IB3];
            } else {
              // Preserve the legacy CT reference energy, which is constructed from
              // its discrete cell-centered magnetic representation.
              cons_(IEN, k, j, i) =
                  point_cons[IEN] -
                  0.5 * (SQR(point_cons[IB1]) + SQR(point_cons[IB2]) +
                         SQR(point_cons[IB3])) +
                  0.5 * (SQR(cons_(IB1, k, j, i)) +
                         SQR(cons_(IB2, k, j, i)) +
                         SQR(cons_(IB3, k, j, i)));
            }
          }
        }
      }
    }

    auto u = mbd->Get("cons").data.GetHostMirrorAndCopy();
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          // Load cell-averaged <U>, either midpoint approx. or fourth-order approx
          Real d1 = cons_(IDN, k, j, i);
          Real m1 = cons_(IM1, k, j, i);
          Real m2 = cons_(IM2, k, j, i);
          Real m3 = cons_(IM3, k, j, i);
          // Weight l1 error by cell volume
          Real vol = pmb->coords.CellVolume(k, j, i);

          l1_err[IDN] += std::abs(d1 - u(IDN, k, j, i)) * vol;
          max_err[IDN] =
              std::max(static_cast<Real>(std::abs(d1 - u(IDN, k, j, i))), max_err[IDN]);
          l1_err[IM1] += std::abs(m1 - u(IM1, k, j, i)) * vol;
          l1_err[IM2] += std::abs(m2 - u(IM2, k, j, i)) * vol;
          l1_err[IM3] += std::abs(m3 - u(IM3, k, j, i)) * vol;
          max_err[IM1] =
              std::max(static_cast<Real>(std::abs(m1 - u(IM1, k, j, i))), max_err[IM1]);
          max_err[IM2] =
              std::max(static_cast<Real>(std::abs(m2 - u(IM2, k, j, i))), max_err[IM2]);
          max_err[IM3] =
              std::max(static_cast<Real>(std::abs(m3 - u(IM3, k, j, i))), max_err[IM3]);

          Real e0 = cons_(IEN, k, j, i);
          l1_err[IEN] += std::abs(e0 - u(IEN, k, j, i)) * vol;
          max_err[IEN] =
              std::max(static_cast<Real>(std::abs(e0 - u(IEN, k, j, i))), max_err[IEN]);

          Real b1 = cons_(IB1, k, j, i);
          Real b2 = cons_(IB2, k, j, i);
          Real b3 = cons_(IB3, k, j, i);
          Real db1 = std::abs(b1 - u(IB1, k, j, i));
          Real db2 = std::abs(b2 - u(IB2, k, j, i));
          Real db3 = std::abs(b3 - u(IB3, k, j, i));

          l1_err[IB1] += db1 * vol;
          l1_err[IB2] += db2 * vol;
          l1_err[IB3] += db3 * vol;
          max_err[IB1] = std::max(db1, max_err[IB1]);
          max_err[IB2] = std::max(db2, max_err[IB2]);
          max_err[IB3] = std::max(db3, max_err[IB3]);
        }
      }
    }
  }
  Real rms_err = 0.0, max_max_over_l1 = 0.0;

#ifdef MPI_PARALLEL
  if (parthenon::Globals::my_rank == 0) {
    MPI_Reduce(MPI_IN_PLACE, &l1_err, (NMHD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(MPI_IN_PLACE, &max_err, (NMHD), MPI_PARTHENON_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  } else {
    MPI_Reduce(&l1_err, &l1_err, (NMHD), MPI_PARTHENON_REAL, MPI_SUM, 0,
               MPI_COMM_WORLD);
    MPI_Reduce(&max_err, &max_err, (NMHD), MPI_PARTHENON_REAL, MPI_MAX, 0,
               MPI_COMM_WORLD);
  }
#endif

  // only the root process outputs the data
  if (parthenon::Globals::my_rank == 0) {
    // normalize errors by number of cells
    const auto mesh_size = mesh->mesh_size;
    const auto vol = (mesh_size.xmax(X1DIR) - mesh_size.xmin(X1DIR)) *
                     (mesh_size.xmax(X2DIR) - mesh_size.xmin(X2DIR)) *
                     (mesh_size.xmax(X3DIR) - mesh_size.xmin(X3DIR));
    for (int i = 0; i < (NMHD); ++i)
      l1_err[i] = l1_err[i] / vol;
    // compute rms error
    for (int i = 0; i < (NMHD); ++i) {
      rms_err += SQR(l1_err[i]);
      if (l1_err[i] > 0.0){
      max_max_over_l1 = std::max(max_max_over_l1, (max_err[i] / l1_err[i]));
      }
    }
    rms_err = std::sqrt(rms_err);

    // open output file and write out errors
    std::string fname;
    fname.assign("smoothVortexMHD-errors.dat");
    std::stringstream msg;
    FILE *pfile;

    // The file exists -- reopen the file in append mode
    if ((pfile = std::fopen(fname.c_str(), "r")) != nullptr) {
      if ((pfile = std::freopen(fname.c_str(), "a", pfile)) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg);
      }

      // The file does not exist -- open the file in write mode and add headers
    } else {
      if ((pfile = std::fopen(fname.c_str(), "w")) == nullptr) {
        msg << "### FATAL ERROR in function Mesh::UserWorkAfterLoop" << std::endl
            << "Error output file could not be opened" << std::endl;
        PARTHENON_FAIL(msg);
      }
      std::fprintf(pfile, "# Nx1  Nx2  Nx3  Ncycle  ");
      std::fprintf(pfile, "RMS-L1-Error  d_L1  M1_L1  M2_L1  M3_L1  E_L1 ");
      std::fprintf(pfile, "  B1c_L1  B2c_L1  B3c_L1");
      std::fprintf(pfile, "  Largest-Max/L1  d_max  M1_max  M2_max  M3_max  E_max ");
      std::fprintf(pfile, "  B1c_max  B2c_max  B3c_max");
      std::fprintf(pfile, "\n");
    }

    // write errors
    std::fprintf(pfile, "%d  %d", mesh_size.nx(X1DIR), mesh_size.nx(X2DIR));
    std::fprintf(pfile, "  %d  %d", mesh_size.nx(X3DIR), tm.ncycle);
    std::fprintf(pfile, "  %e  %e", rms_err, l1_err[IDN]);
    std::fprintf(pfile, "  %e  %e  %e", l1_err[IM1], l1_err[IM2], l1_err[IM3]);
    std::fprintf(pfile, "  %e", l1_err[IEN]);
    std::fprintf(pfile, "  %e", l1_err[IB1]);
    std::fprintf(pfile, "  %e", l1_err[IB2]);
    std::fprintf(pfile, "  %e", l1_err[IB3]);
    std::fprintf(pfile, "  %e  %e  ", max_max_over_l1, max_err[IDN]);
    std::fprintf(pfile, "%e  %e  %e", max_err[IM1], max_err[IM2], max_err[IM3]);
    std::fprintf(pfile, "  %e", max_err[IEN]);
    std::fprintf(pfile, "  %e", max_err[IB1]);
    std::fprintf(pfile, "  %e", max_err[IB2]);
    std::fprintf(pfile, "  %e", max_err[IB3]);
    std::fprintf(pfile, "\n");
    std::fclose(pfile);
  }
}

  //========================================================================================
  //! \fn void MeshBlock::ProblemGenerator(ParameterInput *pin)
  //! \brief smooth mhd vortex problem generator for 2D CT convergence testing.
  //========================================================================================

  void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");
  const bool fourth_order_init =
      mhd_pgen_utils::UseFourthOrderFVInitialization(pmb);

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const Real gm1 = pin->GetReal("hydro", "gamma") - 1.0;
  auto &coords = pmb->coords;

  B0_ = VortexAmplitude();

  // Initialize density and momenta

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

    if (fourth_order_init) {
      const auto evaluate_point_cons =
          [gm1](const Real x1, const Real x2, const Real x3) {
            return EvaluatePointConserved(x1, x2, x3, gm1);
          };
      mhd_pgen_utils::InitializeFourthOrderSmoothMHD(
          pmb, u, Bface, evaluate_point_cons, EvaluateVectorPotential);
    } else {
      Bface_Fill_Cons(pmb, u, Bface);
    }
    u_dev_face.DeepCopy(Bface);
  }

  if (!fourth_order_init) {
    for (int k = kb.s; k <= kb.e; k++) {
      for (int j = jb.s; j <= jb.e; j++) {
        for (int i = ib.s; i <= ib.e; i++) {
          const auto point_cons =
              EvaluatePointConserved(coords.Xc<1>(i), coords.Xc<2>(j),
                                     coords.Xc<3>(k), gm1);
          for (int n = IDN; n <= IEN; ++n) {
            u(n, k, j, i) = point_cons[n];
          }

          if (fluid == Fluid::glmmhd) {
            u(IB1, k, j, i) = point_cons[IB1];
            u(IB2, k, j, i) = point_cons[IB2];
            u(IB3, k, j, i) = point_cons[IB3];
            u(IPS, k, j, i) = 0.0;
          } else {
            // Preserve the legacy CT energy based on its discrete centered B.
            u(IEN, k, j, i) =
                point_cons[IEN] -
                0.5 * (SQR(point_cons[IB1]) + SQR(point_cons[IB2]) +
                       SQR(point_cons[IB3])) +
                0.5 * (SQR(u(IB1, k, j, i)) + SQR(u(IB2, k, j, i)) +
                       SQR(u(IB3, k, j, i)));
          }
        }
      }
    }
  }
  u_dev.DeepCopy(u);
}

//----------------------------------------------------------------------------------------
//! \brief Evaluate the complete analytic pointwise smooth-vortex conserved state.

std::array<Real, IB3 + 1> EvaluatePointConserved(
    const Real x1, const Real x2, const Real /*x3*/, const Real gm1) {
  const Real amp = VortexAmplitude();
  const Real r2 = SQR(x1) + SQR(x2);
  const Real g = std::exp(0.5 * (1.0 - r2));

  std::array<Real, IB3 + 1> u_point{};
  u_point[IDN] = 1.0;
  u_point[IM1] = 1.0 - amp * x2 * g;
  u_point[IM2] = 1.0 + amp * x1 * g;
  u_point[IM3] = 0.0;
  u_point[IB1] = -amp * x2 * g;
  u_point[IB2] = amp * x1 * g;
  u_point[IB3] = 0.0;

  const Real pressure =
      1.0 - r2 * std::exp(1.0 - r2) / (8.0 * SQR(M_PI));
  u_point[IEN] =
      pressure / gm1 +
      0.5 * (SQR(u_point[IM1]) + SQR(u_point[IM2]) +
             SQR(u_point[IM3])) /
          u_point[IDN] +
      0.5 * (SQR(u_point[IB1]) + SQR(u_point[IB2]) +
             SQR(u_point[IB3]));

  return u_point;
}

//----------------------------------------------------------------------------------------
//! \brief Evaluate the analytic smooth-vortex vector potential.

std::array<Real, 3> EvaluateVectorPotential(
    const Real x1, const Real x2, const Real /*x3*/) {
  const Real r2 = SQR(x1) + SQR(x2);
  return {0.0, 0.0,
          VortexAmplitude() * std::exp(0.5 * (1.0 - r2))};
}

template <typename ConsHost, typename BfaceHost>
void Bface_Fill_Cons(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface) {

  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
 
  // always needed
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "az", pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  const Real A0_amp = VortexAmplitude();

  const bool two_d = pmb->pmy_mesh->ndim < 3;

  // Use the vector potential to initialize the smooth vortex.
  auto &coords = pmb->coords;

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
        // define az at the cell-corners or nodes
        Real x_node = coords.X<1, parthenon::TopologicalElement::NN>(k, j, i);
        Real y_node = coords.X<2, parthenon::TopologicalElement::NN>(k, j, i);
        Real R2 = SQR(x_node) + SQR(y_node);
        az(k, j, i) =
            A0_amp * std::exp(0.5*(1-R2));
        }
      }
  }

  // Initialize density and momenta

  auto &mbd = pmb->meshblock_data.Get();
  // initializing on host
  auto Bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto By_face = Bface.Get(IBF2, 0, 0, 0);
  auto Bz_face = Bface.Get(IBF3, 0, 0, 0);

  // fill Bface first (cell faces so +1 on the bounds)

  // x-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e+1; i++) { // +1 here
        // create face-fields with cell-corner defined az
        Bx_face(k, j, i) =
            (az(k, j + 1, i) - az(k, j, i)) / coords.Dxc<2>(j);
      }
    }
  }
  // y-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e+1; j++) { // +1 here
      for (int i = ib.s; i <= ib.e; i++) { 
        // create face-fields with cell-corner defined az
        By_face(k, j, i) =
        - (az(k, j, i + 1) - az(k, j, i)) / coords.Dxc<1>(i);
      }
    }
  }
  // z-face
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e; i++) { 
        Bz_face(k, j, i) = 0.0;
      }
    }
  }
  // now good to fill up the Bx/By cons vector
  for (int k = kb.s; k <= kb.e; k++) {
    for (int j = jb.s; j <= jb.e; j++) { 
      for (int i = ib.s; i <= ib.e; i++) {
        // create cell-centered B from face-centered average
        u(IB1, k, j, i) =
            0.5 * (Bx_face(k, j, i) + Bx_face(k, j, i + 1));
        u(IB2, k, j, i) =
            0.5 * (By_face(k, j, i) + By_face(k, j + 1, i));
        u(IB3, k, j, i) = 0.0; // for 2D testing, fill this outside of here    
      }
    }
  }
}

} // namespace smooth_mhd_vortex
