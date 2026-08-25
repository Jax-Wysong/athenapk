//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../main.hpp"
#include "conversions.hpp"
#include "../../recon/plm_simple.hpp"
#include "../../recon/weno3_simple.hpp"
#include "../../recon/weno5_classic.hpp"



using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;


namespace Hydro::HIGHORDER {


// Convert the point-valued numerical flux at each face center into the
// corresponding fourth-order area average (Teissier et al. 2024, eq. 46).
// The input and output are deliberately separate because the transverse
// stencil cannot be evaluated safely in place.
TaskStatus PointToAreaAverage(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const auto pkg = pmb->packages.Get("Hydro");
  const int nvars = pkg->Param<int>("nhydro") + pkg->Param<int>("nscalars");

  auto cons_pack =
      md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  auto consflux_point_pack =
      md->PackVariables(std::vector<std::string>{"consflux_point"});

  const IndexRange ib =
      md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb =
      md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb =
      md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Teissier X1 point flux to area average",
      parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e + 1, 0, nvars - 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    const int n) {
        auto &cons = cons_pack(b);
        const auto &point_flux = consflux_point_pack(b);
        const Real center = point_flux(TE::F1, n, k, j, i);
        const Real d2y =
            (ndim > 1)
                ? point_flux(TE::F1, n, k, j - 1, i) - 2.0 * center +
                      point_flux(TE::F1, n, k, j + 1, i)
                : 0.0;
        const Real d2z =
            (ndim > 2)
                ? point_flux(TE::F1, n, k - 1, j, i) - 2.0 * center +
                      point_flux(TE::F1, n, k + 1, j, i)
                : 0.0;
        cons.flux(X1DIR, n, k, j, i) = center + (d2y + d2z) / 24.0;
      });

  if (ndim > 1) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier X2 point flux to area average",
        parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s, kb.e,
        jb.s, jb.e + 1, ib.s, ib.e, 0, nvars - 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      const int n) {
          auto &cons = cons_pack(b);
          const auto &point_flux = consflux_point_pack(b);
          const Real center = point_flux(TE::F2, n, k, j, i);
          const Real d2x = point_flux(TE::F2, n, k, j, i - 1) -
                           2.0 * center +
                           point_flux(TE::F2, n, k, j, i + 1);
          const Real d2z =
              (ndim > 2)
                  ? point_flux(TE::F2, n, k - 1, j, i) - 2.0 * center +
                        point_flux(TE::F2, n, k + 1, j, i)
                  : 0.0;
          cons.flux(X2DIR, n, k, j, i) = center + (d2x + d2z) / 24.0;
        });
  }

  if (ndim > 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier X3 point flux to area average",
        parthenon::DevExecSpace(), 0, cons_pack.GetDim(5) - 1, kb.s,
        kb.e + 1, jb.s, jb.e, ib.s, ib.e, 0, nvars - 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      const int n) {
          auto &cons = cons_pack(b);
          const auto &point_flux = consflux_point_pack(b);
          const Real center = point_flux(TE::F3, n, k, j, i);
          const Real d2x = point_flux(TE::F3, n, k, j, i - 1) -
                           2.0 * center +
                           point_flux(TE::F3, n, k, j, i + 1);
          const Real d2y = point_flux(TE::F3, n, k, j - 1, i) -
                           2.0 * center +
                           point_flux(TE::F3, n, k, j + 1, i);
          cons.flux(X3DIR, n, k, j, i) = center + (d2x + d2y) / 24.0;
        });
  }

  return TaskStatus::complete;
}

TaskStatus PointElectricFieldsToAreaAverage(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const int nvars = 4;

  auto teissier_eface_avg_pack =
      md->PackVariables(std::vector<std::string>{"teissier_eface_avg"});
  auto teissier_eface_point_pack =
      md->PackVariables(std::vector<std::string>{"teissier_eface_point"});

  const IndexRange ib =
      md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb =
      md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb =
      md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Teissier X1 point flux to area average",
      parthenon::DevExecSpace(), 0,
      teissier_eface_avg_pack.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e, ib.s, ib.e + 1, 0, nvars - 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                    const int n) {
        auto &eface_avg = teissier_eface_avg_pack(b);
        const auto &eface_point = teissier_eface_point_pack(b);

        const Real center = eface_point(TE::F1, n, k, j, i);
        const Real d2y =
            (ndim > 1)
                ? eface_point(TE::F1, n, k, j - 1, i) - 2.0 * center +
                      eface_point(TE::F1, n, k, j + 1, i)
                : 0.0;
        const Real d2z =
            (ndim > 2)
                ? eface_point(TE::F1, n, k - 1, j, i) - 2.0 * center +
                      eface_point(TE::F1, n, k + 1, j, i)
                : 0.0;
        eface_avg(TE::F1, n, k, j, i) = center + (d2y + d2z) / 24.0;
      });

  if (ndim > 1) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier X2 point flux to area average",
        parthenon::DevExecSpace(), 0,
        teissier_eface_avg_pack.GetDim(5) - 1, kb.s, kb.e,
        jb.s, jb.e + 1, ib.s, ib.e, 0, nvars - 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      const int n) {
          auto &eface_avg = teissier_eface_avg_pack(b);
          const auto &eface_point = teissier_eface_point_pack(b);

          const Real center = eface_point(TE::F2, n, k, j, i);
          const Real d2x = eface_point(TE::F2, n, k, j, i - 1) -
                           2.0 * center +
                           eface_point(TE::F2, n, k, j, i + 1);
          const Real d2z =
              (ndim > 2)
                  ? eface_point(TE::F2, n, k - 1, j, i) - 2.0 * center +
                        eface_point(TE::F2, n, k + 1, j, i)
                  : 0.0;
          eface_avg(TE::F2, n, k, j, i) = center + (d2x + d2z) / 24.0;
        });
  }

  if (ndim > 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier X3 point flux to area average",
        parthenon::DevExecSpace(), 0,
        teissier_eface_avg_pack.GetDim(5) - 1, kb.s,
        kb.e + 1, jb.s, jb.e, ib.s, ib.e, 0, nvars - 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      const int n) {
          auto &eface_avg = teissier_eface_avg_pack(b);
          const auto &eface_point = teissier_eface_point_pack(b);
          const Real center = eface_point(TE::F3, n, k, j, i);
          const Real d2x = eface_point(TE::F3, n, k, j, i - 1) -
                           2.0 * center +
                           eface_point(TE::F3, n, k, j, i + 1);
          const Real d2y = eface_point(TE::F3, n, k, j - 1, i) -
                           2.0 * center +
                           eface_point(TE::F3, n, k, j + 1, i);
          eface_avg(TE::F3, n, k, j, i) = center + (d2x + d2y) / 24.0;
        });
  }

  return TaskStatus::complete;
}

// Convert point-valued edge EMFs into the line averages evolved by CT. In
// two dimensions Ez has no active variation along its edge and is copied
// directly. In three dimensions this is Teissier et al. (2024), eq. (46),
// restricted to the one direction tangent to each edge.
TaskStatus PointEmfToLineAverage(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto Bface_pack =
      md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  auto Bface_point_pack =
      md->PackVariablesAndFluxes(std::vector<std::string>{"Bface_point"});

  const IndexRange ib =
      md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb =
      md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb =
      md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Teissier Ez point EMF to line average",
      parthenon::DevExecSpace(), 0, Bface_pack.GetDim(5) - 1, kb.s, kb.e,
      jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &Bface = Bface_pack(b);
        const auto &Bface_point = Bface_point_pack(b);
        const Real center =
            Bface_point.template flux<parthenon::TopologicalType::Edge>(
                X3DIR, 0, k, j, i);
        Real line_average = center;
        if (ndim > 2) {
          const Real d2z =
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X3DIR, 0, k - 1, j, i) -
              2.0 * center +
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X3DIR, 0, k + 1, j, i);
          line_average += d2z / 24.0;
        }
        Bface.template flux<parthenon::TopologicalType::Edge>(
            X3DIR, 0, k, j, i) = line_average;
      });

  if (ndim > 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier Ey point EMF to line average",
        parthenon::DevExecSpace(), 0, Bface_pack.GetDim(5) - 1, kb.s,
        kb.e + 1, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &Bface = Bface_pack(b);
          const auto &Bface_point = Bface_point_pack(b);
          const Real center =
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X2DIR, 0, k, j, i);
          const Real d2y =
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X2DIR, 0, k, j - 1, i) -
              2.0 * center +
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X2DIR, 0, k, j + 1, i);
          Bface.template flux<parthenon::TopologicalType::Edge>(
              X2DIR, 0, k, j, i) = center + d2y / 24.0;
        });

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier Ex point EMF to line average",
        parthenon::DevExecSpace(), 0, Bface_pack.GetDim(5) - 1, kb.s,
        kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &Bface = Bface_pack(b);
          const auto &Bface_point = Bface_point_pack(b);
          const Real center =
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X1DIR, 0, k, j, i);
          const Real d2x =
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X1DIR, 0, k, j, i - 1) -
              2.0 * center +
              Bface_point.template flux<parthenon::TopologicalType::Edge>(
                  X1DIR, 0, k, j, i + 1);
          Bface.template flux<parthenon::TopologicalType::Edge>(
              X1DIR, 0, k, j, i) = center + d2x / 24.0;
        });
  }

  return TaskStatus::complete;
}

// A block spanning an entire periodic direction owns two representations of the
// same endpoint face.  The same is true for edges transverse to that direction.
// Make the final area-averaged FV fluxes and line-averaged CT EMFs bitwise
// identical before either update consumes them.  Blocks with a distinct periodic
// neighbor have at most one periodic flag in a direction, so this is a no-op for
// ordinary multiblock boundaries.
TaskStatus ReconcileSelfPeriodicFluxesAndEmfs(MeshData<Real> *md) {
  auto cons_pack =
      md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  auto Bface_pack =
      md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});

  const auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const int nvars = pkg->Param<int>("nhydro") + pkg->Param<int>("nscalars");

  for (int b = 0; b < md->NumBlocks(); ++b) {
    auto *pmb = md->GetBlockData(b)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
    const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

    const bool self_periodic_x1 =
        pmb->boundary_flag[parthenon::BoundaryFace::inner_x1] ==
            parthenon::BoundaryFlag::periodic &&
        pmb->boundary_flag[parthenon::BoundaryFace::outer_x1] ==
            parthenon::BoundaryFlag::periodic;
    const bool self_periodic_x2 =
        ndim > 1 &&
        pmb->boundary_flag[parthenon::BoundaryFace::inner_x2] ==
            parthenon::BoundaryFlag::periodic &&
        pmb->boundary_flag[parthenon::BoundaryFace::outer_x2] ==
            parthenon::BoundaryFlag::periodic;
    const bool self_periodic_x3 =
        ndim > 2 &&
        pmb->boundary_flag[parthenon::BoundaryFace::inner_x3] ==
            parthenon::BoundaryFlag::periodic &&
        pmb->boundary_flag[parthenon::BoundaryFace::outer_x3] ==
            parthenon::BoundaryFlag::periodic;

    if (!(self_periodic_x1 || self_periodic_x2 || self_periodic_x3)) continue;

    if (self_periodic_x1) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "Reconcile self-periodic X1 FV flux",
          parthenon::DevExecSpace(), kb.s, kb.e, jb.s, jb.e, 0, nvars - 1,
          KOKKOS_LAMBDA(const int k, const int j, const int n) {
            auto &cons = cons_pack(b);
            cons.flux(X1DIR, n, k, j, ib.e + 1) =
                cons.flux(X1DIR, n, k, j, ib.s);
          });
    }
    if (self_periodic_x2) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "Reconcile self-periodic X2 FV flux",
          parthenon::DevExecSpace(), kb.s, kb.e, ib.s, ib.e, 0, nvars - 1,
          KOKKOS_LAMBDA(const int k, const int i, const int n) {
            auto &cons = cons_pack(b);
            cons.flux(X2DIR, n, k, jb.e + 1, i) =
                cons.flux(X2DIR, n, k, jb.s, i);
          });
    }
    if (self_periodic_x3) {
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "Reconcile self-periodic X3 FV flux",
          parthenon::DevExecSpace(), jb.s, jb.e, ib.s, ib.e, 0, nvars - 1,
          KOKKOS_LAMBDA(const int j, const int i, const int n) {
            auto &cons = cons_pack(b);
            cons.flux(X3DIR, n, kb.e + 1, j, i) =
                cons.flux(X3DIR, n, kb.s, j, i);
          });
    }

    if (ndim > 1) {
      // Each destination edge is written by exactly one iteration.  If an edge
      // lies on two self-periodic upper boundaries, both source indices are
      // replaced at once, so the corner is copied directly from lower/lower.
      parthenon::par_for(
          DEFAULT_LOOP_PATTERN, "Reconcile self-periodic CT EMFs",
          parthenon::DevExecSpace(), kb.s, kb.e + (ndim > 2), jb.s, jb.e + 1,
          ib.s, ib.e + 1,
          KOKKOS_LAMBDA(const int k, const int j, const int i) {
            auto &Bface = Bface_pack(b);
            const int i_lower =
                (self_periodic_x1 && i == ib.e + 1) ? ib.s : i;
            const int j_lower =
                (self_periodic_x2 && j == jb.e + 1) ? jb.s : j;
            const int k_lower =
                (self_periodic_x3 && k == kb.e + 1) ? kb.s : k;

            // Ez is transverse to X1 and X2.
            if (k <= kb.e && (i_lower != i || j_lower != j)) {
              Bface.template flux<parthenon::TopologicalType::Edge>(
                  X3DIR, 0, k, j, i) =
                  Bface.template flux<parthenon::TopologicalType::Edge>(
                      X3DIR, 0, k, j_lower, i_lower);
            }

            if (ndim > 2) {
              // Ey is transverse to X1 and X3.
              if (j <= jb.e && (i_lower != i || k_lower != k)) {
                Bface.template flux<parthenon::TopologicalType::Edge>(
                    X2DIR, 0, k, j, i) =
                    Bface.template flux<parthenon::TopologicalType::Edge>(
                        X2DIR, 0, k_lower, j, i_lower);
              }
              // Ex is transverse to X2 and X3.
              if (i <= ib.e && (j_lower != j || k_lower != k)) {
                Bface.template flux<parthenon::TopologicalType::Edge>(
                    X1DIR, 0, k, j, i) =
                    Bface.template flux<parthenon::TopologicalType::Edge>(
                        X1DIR, 0, k_lower, j_lower, i);
              }
            }
          });
    }
  }

  return TaskStatus::complete;
}


} // namespace Hydro::HIGHORDER
