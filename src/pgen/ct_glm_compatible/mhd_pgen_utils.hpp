//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file mhd_pgen_utils.hpp
//! \brief Shared host-side initialization utilities for smooth fourth-order MHD pgens.

#ifndef PGEN_CT_GLM_COMPATIBLE_MHD_PGEN_UTILS_HPP_
#define PGEN_CT_GLM_COMPATIBLE_MHD_PGEN_UTILS_HPP_

#include <array>
#include <utility>

#include "mesh/mesh.hpp"

#include "../../main.hpp"

namespace mhd_pgen_utils {

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

// These transformations implement the Cartesian, smooth-data formulas used by the
// fourth-order Berta24 path. They intentionally do not handle refinement interfaces or
// nonsmooth problem data.

inline bool UseFourthOrderInitialization(MeshBlock *pmb) {
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  return hydro_pkg->Param<Fluid>("fluid") == Fluid::ucthlldmhd &&
         hydro_pkg->Param<int>("convergence_order") == 4;
}

//----------------------------------------------------------------------------------------
//! \brief Give a self-periodic block one canonical copy of each normal boundary face.
//!
//! A face-centered field stores both endpoint faces of a block. If a block spans an
//! entire periodic direction, those endpoints represent the same physical face but are
//! initialized independently. Copying the lower endpoint to the upper endpoint makes
//! the duplicate representation bitwise identical. On a multiblock periodic direction,
//! no block has periodic flags on both sides, so this routine is a no-op.
//!
//! Call this after constructing Bface and before centering Bface into the cell-centered
//! conserved magnetic field.

template <typename BfaceHost>
void ReconcileSelfPeriodicFaces(MeshBlock *pmb, BfaceHost &Bface) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int ndim = pmb->pmy_mesh->ndim;

  auto bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto by_face = Bface.Get(IBF2, 0, 0, 0);
  auto bz_face = Bface.Get(IBF3, 0, 0, 0);

  const bool self_periodic_x1 =
      pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] ==
          parthenon::BoundaryFlag::periodic &&
      pmb->boundary_flag[parthenon::BoundaryFace::outer_x1] ==
          parthenon::BoundaryFlag::periodic;
  if (self_periodic_x1) {
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        bx_face(k, j, ib.e + 1) = bx_face(k, j, ib.s);
      }
    }
  }

  const bool self_periodic_x2 =
      ndim > 1 &&
      pmb->boundary_flag[parthenon::BoundaryFace::inner_x2] ==
          parthenon::BoundaryFlag::periodic &&
      pmb->boundary_flag[parthenon::BoundaryFace::outer_x2] ==
          parthenon::BoundaryFlag::periodic;
  if (self_periodic_x2) {
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int i = ib.s; i <= ib.e; ++i) {
        by_face(k, jb.e + 1, i) = by_face(k, jb.s, i);
      }
    }
  }

  const bool self_periodic_x3 =
      ndim > 2 &&
      pmb->boundary_flag[parthenon::BoundaryFace::inner_x3] ==
          parthenon::BoundaryFlag::periodic &&
      pmb->boundary_flag[parthenon::BoundaryFace::outer_x3] ==
          parthenon::BoundaryFlag::periodic;
  if (self_periodic_x3) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        bz_face(kb.e + 1, j, i) = bz_face(kb.s, j, i);
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Evaluate an analytic point-conserved state and construct cell averages.
//!
//! The evaluator must return a state indexable from IDN through IB3 and must construct
//! total energy from the complete pointwise state. Only the finite-volume hydro
//! components IDN through IEN are written here; magnetic components are supplied by the
//! vector-potential initialization below.

template <typename ConsHost, typename PointConsEvaluator>
void PointConservedToCellAverage(MeshBlock *pmb, ConsHost &u,
                                 PointConsEvaluator &&evaluate_point_cons) {
  const int ndim = pmb->pmy_mesh->ndim;
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Kokkos::View<Real ****, parthenon::LayoutWrapper, parthenon::HostMemSpace>
      u_point("smooth MHD point conserved", IB3 + 1,
              pmb->cellbounds.ncellsk(IndexDomain::entire),
              pmb->cellbounds.ncellsj(IndexDomain::entire),
              pmb->cellbounds.ncellsi(IndexDomain::entire));

  const int jl = ndim > 1 ? jb.s - 1 : jb.s;
  const int ju = ndim > 1 ? jb.e + 1 : jb.e;
  const int kl = ndim > 2 ? kb.s - 1 : kb.s;
  const int ku = ndim > 2 ? kb.e + 1 : kb.e;
  auto &coords = pmb->coords;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = ib.s - 1; i <= ib.e + 1; ++i) {
        const auto point_cons = evaluate_point_cons(
            coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k));
        for (int n = IDN; n <= IB3; ++n) {
          u_point(n, k, j, i) = point_cons[n];
        }
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        for (int n = IDN; n <= IEN; ++n) {
          const Real dx_q = u_point(n, k, j, i - 1) -
                            2.0 * u_point(n, k, j, i) +
                            u_point(n, k, j, i + 1);
          const Real dy_q =
              ndim > 1 ? u_point(n, k, j - 1, i) -
                             2.0 * u_point(n, k, j, i) +
                             u_point(n, k, j + 1, i)
                       : 0.0;
          const Real dz_q =
              ndim > 2 ? u_point(n, k - 1, j, i) -
                             2.0 * u_point(n, k, j, i) +
                             u_point(n, k + 1, j, i)
                       : 0.0;

          u(n, k, j, i) =
              u_point(n, k, j, i) + (dx_q + dy_q + dz_q) / 24.0;
        }
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Construct line-averaged edge A, face-averaged B, and centered cons B.
//!
//! The evaluator must return {A1, A2, A3} at an arbitrary physical position. Applying
//! the point-to-line-average formula along each edge before the discrete curl produces
//! fourth-order face-area averages while preserving the discrete divergence constraint.

template <typename ConsHost, typename BfaceHost, typename VectorPotentialEvaluator>
void VectorPotentialToFaceAverage(
    MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
    VectorPotentialEvaluator &&evaluate_vector_potential,
    const bool reconcile_periodic_faces = false) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  const int ndim = pmb->pmy_mesh->ndim;
  const bool reduced_dimensional = ndim < 3;

  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ax(
      "smooth MHD line-averaged A1",
      pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> ay(
      "smooth MHD line-averaged A2",
      pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));
  Kokkos::View<Real ***, parthenon::LayoutWrapper, parthenon::HostMemSpace> az(
      "smooth MHD line-averaged A3",
      pmb->cellbounds.ncellsk(IndexDomain::entire),
      pmb->cellbounds.ncellsj(IndexDomain::entire),
      pmb->cellbounds.ncellsi(IndexDomain::entire));

  auto &coords = pmb->coords;
  const int kl = reduced_dimensional ? kb.s : kb.s - 1;
  const int ku = reduced_dimensional ? kb.e : kb.e + 1;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
      for (int i = ib.s - 1; i <= ib.e + 1; ++i) {
        const auto a1_0 = evaluate_vector_potential(
            coords.X<1, TE::E1>(k, j, i), coords.X<2, TE::E1>(k, j, i),
            coords.X<3, TE::E1>(k, j, i));
        const auto a1_m = evaluate_vector_potential(
            coords.X<1, TE::E1>(k, j, i - 1),
            coords.X<2, TE::E1>(k, j, i - 1),
            coords.X<3, TE::E1>(k, j, i - 1));
        const auto a1_p = evaluate_vector_potential(
            coords.X<1, TE::E1>(k, j, i + 1),
            coords.X<2, TE::E1>(k, j, i + 1),
            coords.X<3, TE::E1>(k, j, i + 1));
        ax(k, j, i) = a1_0[0] + (a1_m[0] - 2.0 * a1_0[0] + a1_p[0]) / 24.0;

        const auto a2_0 = evaluate_vector_potential(
            coords.X<1, TE::E2>(k, j, i), coords.X<2, TE::E2>(k, j, i),
            coords.X<3, TE::E2>(k, j, i));
        const auto a2_m = evaluate_vector_potential(
            coords.X<1, TE::E2>(k, j - 1, i),
            coords.X<2, TE::E2>(k, j - 1, i),
            coords.X<3, TE::E2>(k, j - 1, i));
        const auto a2_p = evaluate_vector_potential(
            coords.X<1, TE::E2>(k, j + 1, i),
            coords.X<2, TE::E2>(k, j + 1, i),
            coords.X<3, TE::E2>(k, j + 1, i));
        ay(k, j, i) = a2_0[1] + (a2_m[1] - 2.0 * a2_0[1] + a2_p[1]) / 24.0;

        const auto a3_0 = evaluate_vector_potential(
            coords.X<1, TE::E3>(k, j, i), coords.X<2, TE::E3>(k, j, i),
            coords.X<3, TE::E3>(k, j, i));
        if (ndim > 2) {
          const auto a3_m = evaluate_vector_potential(
              coords.X<1, TE::E3>(k - 1, j, i),
              coords.X<2, TE::E3>(k - 1, j, i),
              coords.X<3, TE::E3>(k - 1, j, i));
          const auto a3_p = evaluate_vector_potential(
              coords.X<1, TE::E3>(k + 1, j, i),
              coords.X<2, TE::E3>(k + 1, j, i),
              coords.X<3, TE::E3>(k + 1, j, i));
          az(k, j, i) =
              a3_0[2] + (a3_m[2] - 2.0 * a3_0[2] + a3_p[2]) / 24.0;
        } else {
          // The E3 edge direction is inactive in 2D, so this line average is its
          // point value.
          az(k, j, i) = a3_0[2];
        }
      }
    }
  }

  auto bx_face = Bface.Get(IBF1, 0, 0, 0);
  auto by_face = Bface.Get(IBF2, 0, 0, 0);
  auto bz_face = Bface.Get(IBF3, 0, 0, 0);

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e + 1; ++i) {
        const Real da2_dx3 =
            reduced_dimensional
                ? 0.0
                : (ay(k + 1, j, i) - ay(k, j, i)) / coords.Dxc<3>(k);
        bx_face(k, j, i) =
            (az(k, j + 1, i) - az(k, j, i)) / coords.Dxc<2>(j) - da2_dx3;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e + 1; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const Real da1_dx3 =
            reduced_dimensional
                ? 0.0
                : (ax(k + 1, j, i) - ax(k, j, i)) / coords.Dxc<3>(k);
        by_face(k, j, i) =
            da1_dx3 - (az(k, j, i + 1) - az(k, j, i)) / coords.Dxc<1>(i);
      }
    }
  }

  for (int k = kb.s; k <= ku; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        bz_face(k, j, i) =
            reduced_dimensional
                ? 0.0
                : (ay(k, j, i + 1) - ay(k, j, i)) / coords.Dxc<1>(i) -
                      (ax(k, j + 1, i) - ax(k, j, i)) / coords.Dxc<2>(j);
      }
    }
  }

  if (reconcile_periodic_faces) {
    ReconcileSelfPeriodicFaces(pmb, Bface);
  }

  // Match the cell-centered magnetic representation reconstructed by AthenaPK's
  // runtime CT FillDerived path. In 2D, B3 is a cell-area average obtained directly
  // from the line-averaged in-plane vector potential.
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        u(IB1, k, j, i) =
            0.5 * (bx_face(k, j, i) + bx_face(k, j, i + 1));
        u(IB2, k, j, i) =
            0.5 * (by_face(k, j, i) + by_face(k, j + 1, i));
        u(IB3, k, j, i) =
            reduced_dimensional
                ? (ay(k, j, i + 1) - ay(k, j, i)) / coords.Dxc<1>(i) -
                      (ax(k, j + 1, i) - ax(k, j, i)) / coords.Dxc<2>(j)
                : 0.5 * (bz_face(k, j, i) + bz_face(k + 1, j, i));
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Initialize all fourth-order smooth-MHD finite-volume representations.

template <typename ConsHost, typename BfaceHost, typename PointConsEvaluator,
          typename VectorPotentialEvaluator>
void InitializeFourthOrderSmoothMHD(
    MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
    PointConsEvaluator &&evaluate_point_cons,
    VectorPotentialEvaluator &&evaluate_vector_potential,
    const bool reconcile_periodic_faces = false) {
  PointConservedToCellAverage(
      pmb, u, std::forward<PointConsEvaluator>(evaluate_point_cons));
  VectorPotentialToFaceAverage(
      pmb, u, Bface,
      std::forward<VectorPotentialEvaluator>(evaluate_vector_potential),
      reconcile_periodic_faces);
}

} // namespace mhd_pgen_utils

#endif // PGEN_CT_GLM_COMPATIBLE_MHD_PGEN_UTILS_HPP_
