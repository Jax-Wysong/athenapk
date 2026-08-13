//========================================================================================
// AthenaPK - a performance portable block structured AMR MHD code
// Copyright (c) 2021-2023, Athena Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================
//! \file mhd_blast.cpp
//! \brief Problem generator for Berta et al.'s three-dimensional C1 MHD blast.
//!
//! REFERENCE: Berta et al. (2024), Section 5.7 and Table 3.

// C++ headers
#include <array>
#include <cmath>

// Parthenon headers
#include "mesh/mesh.hpp"
#include <parthenon/driver.hpp>
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

namespace mhd_blast {
using namespace parthenon::driver::prelude;

struct BlastParameters {
  Real radius;
  Real x1_center;
  Real x2_center;
  Real x3_center;
  Real density_inner;
  Real density_ambient;
  Real pressure_inner;
  Real pressure_ambient;
  Real b1;
  Real b2;
  Real b3;
  Real gm1;
};

//----------------------------------------------------------------------------------------
//! \brief Return the piecewise-constant conserved state assigned to a cell.

std::array<Real, IB3 + 1> EvaluateCellConserved(const Real x1, const Real x2,
                                                const Real x3,
                                                const BlastParameters &params) {
  const Real radius_sq = SQR(x1 - params.x1_center) +
                         SQR(x2 - params.x2_center) +
                         SQR(x3 - params.x3_center);
  const bool inside = radius_sq < SQR(params.radius);
  const Real density = inside ? params.density_inner : params.density_ambient;
  const Real pressure = inside ? params.pressure_inner : params.pressure_ambient;

  std::array<Real, IB3 + 1> u_cell{};
  u_cell[IDN] = density;
  u_cell[IM1] = 0.0;
  u_cell[IM2] = 0.0;
  u_cell[IM3] = 0.0;
  u_cell[IB1] = params.b1;
  u_cell[IB2] = params.b2;
  u_cell[IB3] = params.b3;
  u_cell[IEN] =
      pressure / params.gm1 +
      0.5 * (SQR(params.b1) + SQR(params.b2) + SQR(params.b3));
  return u_cell;
}

//----------------------------------------------------------------------------------------
//! \brief Fill the constant staggered CT field and its cell-centered representation.

template <typename ConsHost, typename BfaceHost>
void InitializeConstantCTField(MeshBlock *pmb, ConsHost &u, BfaceHost &Bface,
                               const BlastParameters &params) {
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto b1_face = Bface.Get(IBF1, 0, 0, 0);
  auto b2_face = Bface.Get(IBF2, 0, 0, 0);
  auto b3_face = Bface.Get(IBF3, 0, 0, 0);

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e + 1; ++i) {
        b1_face(k, j, i) = params.b1;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e + 1; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        b2_face(k, j, i) = params.b2;
      }
    }
  }

  for (int k = kb.s; k <= kb.e + 1; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        b3_face(k, j, i) = params.b3;
      }
    }
  }

  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        u(IB1, k, j, i) =
            0.5 * (b1_face(k, j, i) + b1_face(k, j, i + 1));
        u(IB2, k, j, i) =
            0.5 * (b2_face(k, j, i) + b2_face(k, j + 1, i));
        u(IB3, k, j, i) =
            0.5 * (b3_face(k, j, i) + b3_face(k + 1, j, i));
      }
    }
  }
}

//----------------------------------------------------------------------------------------
//! \brief Problem generator for Berta et al.'s three-dimensional C1 MHD blast.

void ProblemGenerator(MeshBlock *pmb, ParameterInput *pin) {
  const auto hydro_pkg = pmb->packages.Get("Hydro");
  const auto fluid = hydro_pkg->Param<Fluid>("fluid");

  PARTHENON_REQUIRE_THROWS(
      pmb->pmy_mesh->ndim == 3,
      "The mhd_blast problem is the three-dimensional Cartesian Berta24 C1 test");
  PARTHENON_REQUIRE_THROWS(
      fluid == Fluid::glmmhd || fluid == Fluid::ctmhd ||
          fluid == Fluid::ucthlldmhd,
      "The mhd_blast problem requires an MHD fluid");

  const Real pi = std::acos(-1.0);
  const Real b0 = pin->GetOrAddReal("problem/mhd_blast", "b0", 3.0);
  const Real theta =
      pin->GetOrAddReal("problem/mhd_blast", "theta", 0.5 * pi);
  const Real phi = pin->GetOrAddReal("problem/mhd_blast", "phi", 0.25 * pi);
  const Real gamma = pin->GetReal("hydro", "gamma");
  BlastParameters params{
      pin->GetOrAddReal("problem/mhd_blast", "radius", 0.1),
      pin->GetOrAddReal("problem/mhd_blast", "x1_0", 0.0),
      pin->GetOrAddReal("problem/mhd_blast", "x2_0", 0.0),
      pin->GetOrAddReal("problem/mhd_blast", "x3_0", 0.0),
      pin->GetOrAddReal("problem/mhd_blast", "density_inner", 1.0),
      pin->GetOrAddReal("problem/mhd_blast", "density_ambient", 1.0),
      pin->GetOrAddReal("problem/mhd_blast", "pressure_inner", 10.0),
      pin->GetOrAddReal("problem/mhd_blast", "pressure_ambient", 0.1),
      b0 * std::sin(theta) * std::cos(phi),
      b0 * std::sin(theta) * std::sin(phi),
      b0 * std::cos(theta),
      gamma - 1.0};

  PARTHENON_REQUIRE_THROWS(params.radius > 0.0,
                           "problem/mhd_blast/radius must be positive");
  PARTHENON_REQUIRE_THROWS(
      params.density_inner > 0.0 && params.density_ambient > 0.0,
      "mhd_blast densities must be positive");
  PARTHENON_REQUIRE_THROWS(
      params.pressure_inner > 0.0 && params.pressure_ambient > 0.0,
      "mhd_blast pressures must be positive");
  PARTHENON_REQUIRE_THROWS(gamma > 1.0,
                           "mhd_blast requires hydro/gamma > 1");
  PARTHENON_REQUIRE_THROWS(b0 >= 0.0,
                           "problem/mhd_blast/b0 must be nonnegative");

  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto &coords = pmb->coords;

  auto &mbd = pmb->meshblock_data.Get();
  auto &u_dev = mbd->Get("cons").data;
  auto u = u_dev.GetHostMirrorAndCopy();

  // The initial profile is discontinuous. Assign the piecewise-constant state
  // selected at the cell center as the cell's finite-volume average.
  for (int k = kb.s; k <= kb.e; ++k) {
    for (int j = jb.s; j <= jb.e; ++j) {
      for (int i = ib.s; i <= ib.e; ++i) {
        const auto u_cell = EvaluateCellConserved(
            coords.Xc<1>(i), coords.Xc<2>(j), coords.Xc<3>(k), params);
        for (int n = IDN; n <= IEN; ++n) {
          u(n, k, j, i) = u_cell[n];
        }
      }
    }
  }

  if (fluid == Fluid::ctmhd || fluid == Fluid::ucthlldmhd) {
    auto &bface_dev = mbd->Get("Bface").data;
    auto Bface = bface_dev.GetHostMirrorAndCopy();
    InitializeConstantCTField(pmb, u, Bface, params);
    bface_dev.DeepCopy(Bface);
  } else {
    for (int k = kb.s; k <= kb.e; ++k) {
      for (int j = jb.s; j <= jb.e; ++j) {
        for (int i = ib.s; i <= ib.e; ++i) {
          u(IB1, k, j, i) = params.b1;
          u(IB2, k, j, i) = params.b2;
          u(IB3, k, j, i) = params.b3;
          u(IPS, k, j, i) = 0.0;
        }
      }
    }
  }

  u_dev.DeepCopy(u);
}

} // namespace mhd_blast
