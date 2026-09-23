//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file mhd_pgen_utils.hpp
//! \brief Shared host-side initialization utilities for high-order FV MHD pgens.

#ifndef PGEN_CT_GLM_COMPATIBLE_MHD_PGEN_UTILS_HPP_
#define PGEN_CT_GLM_COMPATIBLE_MHD_PGEN_UTILS_HPP_

#include <array>
#include <utility>

#include "mesh/mesh.hpp"

#include "../../main.hpp"

namespace mhd_pgen_utils {

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

// These transformations construct the volume-, area-, and line-averaged initial data
// required by Cartesian high-order finite-volume schemes. They intentionally do not
// handle refinement interfaces. Problem-specific nonsmooth initialization may require
// quadrature rather than the smooth point-to-average transformations below.

inline bool UseFourthOrderFVInitialization(MeshBlock *pmb) {
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  return hydro_pkg->Param<Fluid>("fluid") == Fluid::ucthlldmhd &&
         hydro_pkg->Param<int>("convergence_order") == 4;
}

inline bool UseHighOrderFVInitialization(MeshBlock *pmb) {
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  const int order = hydro_pkg->Param<int>("convergence_order");
  return hydro_pkg->Param<Fluid>("fluid") == Fluid::ucthlldmhd &&
         (order == 4 || order == 6 || order == 8);
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
//! total energy from the complete pointwise state. All conserved components are
//! converted to cell averages here. The vector-potential initialization below
//! independently constructs the divergence-free face-averaged magnetic field.

template <int ORDER, typename ConsHost, typename PointConsEvaluator>
void PointConservedToCellAverageImpl(
    MeshBlock *pmb, ConsHost &u,
    PointConsEvaluator &&evaluate_point_cons) {
  static_assert(ORDER == 4 || ORDER == 6 || ORDER == 8,
                "Smooth MHD initialization supports orders four, six, and eight");
  const int ndim = pmb->pmy_mesh->ndim;
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  Kokkos::View<Real ****, parthenon::LayoutWrapper, parthenon::HostMemSpace>
      u_point("smooth MHD point conserved", IB3 + 1,
              pmb->cellbounds.ncellsk(IndexDomain::entire),
              pmb->cellbounds.ncellsj(IndexDomain::entire),
              pmb->cellbounds.ncellsi(IndexDomain::entire));

  constexpr int point_halo = ORDER / 2 - 1;
  const int jl = ndim > 1 ? jb.s - point_halo : jb.s;
  const int ju = ndim > 1 ? jb.e + point_halo : jb.e;
  const int kl = ndim > 2 ? kb.s - point_halo : kb.s;
  const int ku = ndim > 2 ? kb.e + point_halo : kb.e;
  auto &coords = pmb->coords;

  for (int k = kl; k <= ku; ++k) {
    for (int j = jl; j <= ju; ++j) {
      for (int i = ib.s - point_halo; i <= ib.e + point_halo; ++i) {
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
        for (int n = IDN; n <= IB3; ++n) {
          const Real center = u_point(n, k, j, i);
          if constexpr (ORDER == 4) {
            const Real dx_q = u_point(n, k, j, i - 1) - 2.0 * center +
                              u_point(n, k, j, i + 1);
            const Real dy_q =
                ndim > 1 ? u_point(n, k, j - 1, i) - 2.0 * center +
                               u_point(n, k, j + 1, i)
                         : 0.0;
            const Real dz_q =
                ndim > 2 ? u_point(n, k - 1, j, i) - 2.0 * center +
                               u_point(n, k + 1, j, i)
                         : 0.0;
            u(n, k, j, i) = center + (dx_q + dy_q + dz_q) / 24.0;
          } else if constexpr (ORDER == 6) {
            Real axial_1 = (u_point(n, k, j, i - 1) - center) +
                           (u_point(n, k, j, i + 1) - center);
            Real axial_2 = (u_point(n, k, j, i - 2) - center) +
                           (u_point(n, k, j, i + 2) - center);

            if (ndim > 1) {
              axial_1 += (u_point(n, k, j - 1, i) - center) +
                           (u_point(n, k, j + 1, i) - center);
              axial_2 += (u_point(n, k, j - 2, i) - center) +
                           (u_point(n, k, j + 2, i) - center);
            }
            if (ndim > 2) {
              axial_1 += (u_point(n, k - 1, j, i) - center) +
                           (u_point(n, k + 1, j, i) - center);
              axial_2 += (u_point(n, k - 2, j, i) - center) +
                           (u_point(n, k + 2, j, i) - center);
            }

            if (ndim == 1) {
              u(n, k, j, i) = center + (77.0 / 1440.0) * axial_1 -
                              (17.0 / 5760.0) * axial_2;
            } else {
              Real diagonal_1 =
                  (u_point(n, k, j - 1, i - 1) - center) +
                  (u_point(n, k, j - 1, i + 1) - center) +
                  (u_point(n, k, j + 1, i - 1) - center) +
                  (u_point(n, k, j + 1, i + 1) - center);
              if (ndim > 2) {
                diagonal_1 +=
                    (u_point(n, k - 1, j, i - 1) - center) +
                    (u_point(n, k - 1, j, i + 1) - center) +
                    (u_point(n, k + 1, j, i - 1) - center) +
                    (u_point(n, k + 1, j, i + 1) - center) +
                    (u_point(n, k - 1, j - 1, i) - center) +
                    (u_point(n, k - 1, j + 1, i) - center) +
                    (u_point(n, k + 1, j - 1, i) - center) +
                    (u_point(n, k + 1, j + 1, i) - center);
              }

              const Real axial_1_coeff =
                  ndim == 2 ? 1.0 / 20.0 : 67.0 / 1440.0;
              u(n, k, j, i) =
                  center + axial_1_coeff * axial_1 -
                  (17.0 / 5760.0) * axial_2 +
                  (1.0 / 576.0) * diagonal_1;
            }
          } else {
            const auto dq = [&](const int dk, const int dj, const int di) {
              return u_point(n, k + dk, j + dj, i + di) - center;
            };

            Real axial_1 = dq(0, 0, -1) + dq(0, 0, 1);
            Real axial_2 = dq(0, 0, -2) + dq(0, 0, 2);
            Real axial_3 = dq(0, 0, -3) + dq(0, 0, 3);
            if (ndim > 1) {
              axial_1 += dq(0, -1, 0) + dq(0, 1, 0);
              axial_2 += dq(0, -2, 0) + dq(0, 2, 0);
              axial_3 += dq(0, -3, 0) + dq(0, 3, 0);
            }
            if (ndim > 2) {
              axial_1 += dq(-1, 0, 0) + dq(1, 0, 0);
              axial_2 += dq(-2, 0, 0) + dq(2, 0, 0);
              axial_3 += dq(-3, 0, 0) + dq(3, 0, 0);
            }

            if (ndim == 1) {
              u(n, k, j, i) =
                  center + (6361.0 / 107520.0) * axial_1 -
                  (281.0 / 53760.0) * axial_2 +
                  (367.0 / 967680.0) * axial_3;
            } else {
              Real diagonal_11 =
                  dq(0, -1, -1) + dq(0, -1, 1) +
                  dq(0, 1, -1) + dq(0, 1, 1);
              Real diagonal_21 =
                  dq(0, -2, -1) + dq(0, -2, 1) +
                  dq(0, 2, -1) + dq(0, 2, 1) +
                  dq(0, -1, -2) + dq(0, -1, 2) +
                  dq(0, 1, -2) + dq(0, 1, 2);

              if (ndim > 2) {
                diagonal_11 +=
                    dq(-1, 0, -1) + dq(-1, 0, 1) +
                    dq(1, 0, -1) + dq(1, 0, 1) +
                    dq(-1, -1, 0) + dq(-1, 1, 0) +
                    dq(1, -1, 0) + dq(1, 1, 0);
                diagonal_21 +=
                    dq(-2, 0, -1) + dq(-2, 0, 1) +
                    dq(2, 0, -1) + dq(2, 0, 1) +
                    dq(-1, 0, -2) + dq(-1, 0, 2) +
                    dq(1, 0, -2) + dq(1, 0, 2) +
                    dq(-2, -1, 0) + dq(-2, 1, 0) +
                    dq(2, -1, 0) + dq(2, 1, 0) +
                    dq(-1, -2, 0) + dq(-1, 2, 0) +
                    dq(1, -2, 0) + dq(1, 2, 0);
              }

              if (ndim == 2) {
                u(n, k, j, i) =
                    center + (52223.0 / 967680.0) * axial_1 -
                    (241.0 / 48384.0) * axial_2 +
                    (367.0 / 967680.0) * axial_3 +
                    (47.0 / 17280.0) * diagonal_11 -
                    (17.0 / 138240.0) * diagonal_21;
              } else {
                const Real corner_111 =
                    dq(-1, -1, -1) + dq(-1, -1, 1) +
                    dq(-1, 1, -1) + dq(-1, 1, 1) +
                    dq(1, -1, -1) + dq(1, -1, 1) +
                    dq(1, 1, -1) + dq(1, 1, 1);
                u(n, k, j, i) =
                    center + (47477.0 / 967680.0) * axial_1 -
                    (2291.0 / 483840.0) * axial_2 +
                    (367.0 / 967680.0) * axial_3 +
                    (89.0 / 34560.0) * diagonal_11 -
                    (17.0 / 138240.0) * diagonal_21 +
                    (1.0 / 13824.0) * corner_111;
              }
            }
          }
        }
      }
    }
  }
}

template <typename ConsHost, typename PointConsEvaluator>
void PointConservedToCellAverage(
    MeshBlock *pmb, ConsHost &u,
    PointConsEvaluator &&evaluate_point_cons) {
  PointConservedToCellAverageImpl<4>(
      pmb, u, std::forward<PointConsEvaluator>(evaluate_point_cons));
}

//----------------------------------------------------------------------------------------
//! \brief Construct line-averaged edge A and divergence-free face-averaged B.
//!
//! The evaluator must return {A1, A2, A3} at an arbitrary physical position. Applying
//! the point-to-line-average formula along each edge before the discrete curl produces
//! high-order face-area averages while preserving the discrete divergence constraint.

template <int ORDER, typename BfaceHost, typename VectorPotentialEvaluator>
void VectorPotentialToFaceAverageImpl(
    MeshBlock *pmb, BfaceHost &Bface,
    VectorPotentialEvaluator &&evaluate_vector_potential,
    const bool reconcile_periodic_faces = false) {
  static_assert(ORDER == 4 || ORDER == 6 || ORDER == 8,
                "Smooth MHD initialization supports orders four, six, and eight");
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
        if constexpr (ORDER == 4) {
          ax(k, j, i) =
              a1_0[0] + (a1_m[0] - 2.0 * a1_0[0] + a1_p[0]) / 24.0;
        } else if constexpr (ORDER == 6) {
          const auto a1_mm = evaluate_vector_potential(
              coords.X<1, TE::E1>(k, j, i - 2),
              coords.X<2, TE::E1>(k, j, i - 2),
              coords.X<3, TE::E1>(k, j, i - 2));
          const auto a1_pp = evaluate_vector_potential(
              coords.X<1, TE::E1>(k, j, i + 2),
              coords.X<2, TE::E1>(k, j, i + 2),
              coords.X<3, TE::E1>(k, j, i + 2));
          ax(k, j, i) =
              a1_0[0] +
              (77.0 / 1440.0) *
                  ((a1_m[0] - a1_0[0]) + (a1_p[0] - a1_0[0])) -
              (17.0 / 5760.0) *
                  ((a1_mm[0] - a1_0[0]) + (a1_pp[0] - a1_0[0]));
        } else {
          const auto a1_mm = evaluate_vector_potential(
              coords.X<1, TE::E1>(k, j, i - 2),
              coords.X<2, TE::E1>(k, j, i - 2),
              coords.X<3, TE::E1>(k, j, i - 2));
          const auto a1_pp = evaluate_vector_potential(
              coords.X<1, TE::E1>(k, j, i + 2),
              coords.X<2, TE::E1>(k, j, i + 2),
              coords.X<3, TE::E1>(k, j, i + 2));
          const auto a1_mmm = evaluate_vector_potential(
              coords.X<1, TE::E1>(k, j, i - 3),
              coords.X<2, TE::E1>(k, j, i - 3),
              coords.X<3, TE::E1>(k, j, i - 3));
          const auto a1_ppp = evaluate_vector_potential(
              coords.X<1, TE::E1>(k, j, i + 3),
              coords.X<2, TE::E1>(k, j, i + 3),
              coords.X<3, TE::E1>(k, j, i + 3));
          ax(k, j, i) =
              a1_0[0] +
              (6361.0 / 107520.0) *
                  ((a1_m[0] - a1_0[0]) + (a1_p[0] - a1_0[0])) -
              (281.0 / 53760.0) *
                  ((a1_mm[0] - a1_0[0]) + (a1_pp[0] - a1_0[0])) +
              (367.0 / 967680.0) *
                  ((a1_mmm[0] - a1_0[0]) + (a1_ppp[0] - a1_0[0]));
        }

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
        if constexpr (ORDER == 4) {
          ay(k, j, i) =
              a2_0[1] + (a2_m[1] - 2.0 * a2_0[1] + a2_p[1]) / 24.0;
        } else if constexpr (ORDER == 6) {
          const auto a2_mm = evaluate_vector_potential(
              coords.X<1, TE::E2>(k, j - 2, i),
              coords.X<2, TE::E2>(k, j - 2, i),
              coords.X<3, TE::E2>(k, j - 2, i));
          const auto a2_pp = evaluate_vector_potential(
              coords.X<1, TE::E2>(k, j + 2, i),
              coords.X<2, TE::E2>(k, j + 2, i),
              coords.X<3, TE::E2>(k, j + 2, i));
          ay(k, j, i) =
              a2_0[1] +
              (77.0 / 1440.0) *
                  ((a2_m[1] - a2_0[1]) + (a2_p[1] - a2_0[1])) -
              (17.0 / 5760.0) *
                  ((a2_mm[1] - a2_0[1]) + (a2_pp[1] - a2_0[1]));
        } else {
          const auto a2_mm = evaluate_vector_potential(
              coords.X<1, TE::E2>(k, j - 2, i),
              coords.X<2, TE::E2>(k, j - 2, i),
              coords.X<3, TE::E2>(k, j - 2, i));
          const auto a2_pp = evaluate_vector_potential(
              coords.X<1, TE::E2>(k, j + 2, i),
              coords.X<2, TE::E2>(k, j + 2, i),
              coords.X<3, TE::E2>(k, j + 2, i));
          const auto a2_mmm = evaluate_vector_potential(
              coords.X<1, TE::E2>(k, j - 3, i),
              coords.X<2, TE::E2>(k, j - 3, i),
              coords.X<3, TE::E2>(k, j - 3, i));
          const auto a2_ppp = evaluate_vector_potential(
              coords.X<1, TE::E2>(k, j + 3, i),
              coords.X<2, TE::E2>(k, j + 3, i),
              coords.X<3, TE::E2>(k, j + 3, i));
          ay(k, j, i) =
              a2_0[1] +
              (6361.0 / 107520.0) *
                  ((a2_m[1] - a2_0[1]) + (a2_p[1] - a2_0[1])) -
              (281.0 / 53760.0) *
                  ((a2_mm[1] - a2_0[1]) + (a2_pp[1] - a2_0[1])) +
              (367.0 / 967680.0) *
                  ((a2_mmm[1] - a2_0[1]) + (a2_ppp[1] - a2_0[1]));
        }

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
          if constexpr (ORDER == 4) {
            az(k, j, i) =
                a3_0[2] + (a3_m[2] - 2.0 * a3_0[2] + a3_p[2]) / 24.0;
          } else if constexpr (ORDER == 6) {
            const auto a3_mm = evaluate_vector_potential(
                coords.X<1, TE::E3>(k - 2, j, i),
                coords.X<2, TE::E3>(k - 2, j, i),
                coords.X<3, TE::E3>(k - 2, j, i));
            const auto a3_pp = evaluate_vector_potential(
                coords.X<1, TE::E3>(k + 2, j, i),
                coords.X<2, TE::E3>(k + 2, j, i),
                coords.X<3, TE::E3>(k + 2, j, i));
            az(k, j, i) =
                a3_0[2] +
                (77.0 / 1440.0) *
                    ((a3_m[2] - a3_0[2]) + (a3_p[2] - a3_0[2])) -
                (17.0 / 5760.0) *
                    ((a3_mm[2] - a3_0[2]) + (a3_pp[2] - a3_0[2]));
          } else {
            const auto a3_mm = evaluate_vector_potential(
                coords.X<1, TE::E3>(k - 2, j, i),
                coords.X<2, TE::E3>(k - 2, j, i),
                coords.X<3, TE::E3>(k - 2, j, i));
            const auto a3_pp = evaluate_vector_potential(
                coords.X<1, TE::E3>(k + 2, j, i),
                coords.X<2, TE::E3>(k + 2, j, i),
                coords.X<3, TE::E3>(k + 2, j, i));
            const auto a3_mmm = evaluate_vector_potential(
                coords.X<1, TE::E3>(k - 3, j, i),
                coords.X<2, TE::E3>(k - 3, j, i),
                coords.X<3, TE::E3>(k - 3, j, i));
            const auto a3_ppp = evaluate_vector_potential(
                coords.X<1, TE::E3>(k + 3, j, i),
                coords.X<2, TE::E3>(k + 3, j, i),
                coords.X<3, TE::E3>(k + 3, j, i));
            az(k, j, i) =
                a3_0[2] +
                (6361.0 / 107520.0) *
                    ((a3_m[2] - a3_0[2]) + (a3_p[2] - a3_0[2])) -
                (281.0 / 53760.0) *
                    ((a3_mm[2] - a3_0[2]) + (a3_pp[2] - a3_0[2])) +
                (367.0 / 967680.0) *
                    ((a3_mmm[2] - a3_0[2]) + (a3_ppp[2] - a3_0[2]));
          }
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

  // Do not derive cell-centered magnetic fields here. The caller has already
  // initialized their true volume averages with PointConservedToCellAverage().
  // During evolution, Teissier's CT FillDerived path reconstructs the corresponding
  // high-order cell averages from these divergence-free face averages.
}

template <typename BfaceHost, typename VectorPotentialEvaluator>
void VectorPotentialToFaceAverage(
    MeshBlock *pmb, BfaceHost &Bface,
    VectorPotentialEvaluator &&evaluate_vector_potential,
    const bool reconcile_periodic_faces = false) {
  VectorPotentialToFaceAverageImpl<4>(
      pmb, Bface,
      std::forward<VectorPotentialEvaluator>(evaluate_vector_potential),
      reconcile_periodic_faces);
}

//----------------------------------------------------------------------------------------
//! \brief Initialize all smooth-MHD finite-volume representations at ORDER.

template <int ORDER, typename ConsHost, typename BfaceHost,
          typename PointConsEvaluator, typename VectorPotentialEvaluator>
void InitializeSmoothMHDImpl(
    MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
    PointConsEvaluator &&evaluate_point_cons,
    VectorPotentialEvaluator &&evaluate_vector_potential,
    const bool reconcile_periodic_faces = false) {
  PointConservedToCellAverageImpl<ORDER>(
      pmb, u, std::forward<PointConsEvaluator>(evaluate_point_cons));
  VectorPotentialToFaceAverageImpl<ORDER>(
      pmb, Bface,
      std::forward<VectorPotentialEvaluator>(evaluate_vector_potential),
      reconcile_periodic_faces);
}

template <typename ConsHost, typename BfaceHost, typename PointConsEvaluator,
          typename VectorPotentialEvaluator>
void InitializeFourthOrderSmoothMHD(
    MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
    PointConsEvaluator &&evaluate_point_cons,
    VectorPotentialEvaluator &&evaluate_vector_potential,
    const bool reconcile_periodic_faces = false) {
  InitializeSmoothMHDImpl<4>(
      pmb, u, Bface,
      std::forward<PointConsEvaluator>(evaluate_point_cons),
      std::forward<VectorPotentialEvaluator>(evaluate_vector_potential),
      reconcile_periodic_faces);
}

template <typename ConsHost, typename BfaceHost, typename PointConsEvaluator,
          typename VectorPotentialEvaluator>
void InitializeHighOrderSmoothMHD(
    MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
    PointConsEvaluator &&evaluate_point_cons,
    VectorPotentialEvaluator &&evaluate_vector_potential,
    const bool reconcile_periodic_faces = false) {
  const int order =
      pmb->packages.Get("Hydro")->Param<int>("convergence_order");
  PARTHENON_REQUIRE(
      order == 4 || order == 6 || order == 8,
      "High-order smooth MHD initialization requires order four, six, or eight");
  if (order == 4) {
    InitializeSmoothMHDImpl<4>(
        pmb, u, Bface,
        std::forward<PointConsEvaluator>(evaluate_point_cons),
        std::forward<VectorPotentialEvaluator>(evaluate_vector_potential),
        reconcile_periodic_faces);
  } else if (order == 6) {
    InitializeSmoothMHDImpl<6>(
        pmb, u, Bface,
        std::forward<PointConsEvaluator>(evaluate_point_cons),
        std::forward<VectorPotentialEvaluator>(evaluate_vector_potential),
        reconcile_periodic_faces);
  } else {
    InitializeSmoothMHDImpl<8>(
        pmb, u, Bface,
        std::forward<PointConsEvaluator>(evaluate_point_cons),
        std::forward<VectorPotentialEvaluator>(evaluate_vector_potential),
        reconcile_periodic_faces);
  }
}

} // namespace mhd_pgen_utils

#endif // PGEN_CT_GLM_COMPATIBLE_MHD_PGEN_UTILS_HPP_
