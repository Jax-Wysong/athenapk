// This file was made in part with generative AI
#ifndef HYDRO_CTMHD_CT_FOFC_HELPER_HPP_
#define HYDRO_CTMHD_CT_FOFC_HELPER_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "ct_update_helpers.hpp"

// Kokkos headers
#include <Kokkos_MathematicalFunctions.hpp>

using namespace parthenon::package::prelude;

namespace Hydro::CTMHD {
using TE = parthenon::TopologicalElement;


// NOTE:
// before this function is called,
// new_cons must already contain the
// FV stage prediction
template <typename FacePack, typename Coordinates>
KOKKOS_INLINE_FUNCTION
void PredictCTStageConserved(
    const FacePack &B0, const FacePack &B1,
    const Coordinates &coords, const int ndim,
    const int k, const int j, const int i,
    const parthenon::Real gam0, const parthenon::Real gam1,
    const parthenon::Real beta_dt,
    const bool correct_ct_energy,
    parthenon::Real *new_cons) {

          const Real b1_god = new_cons[IB1];
          const Real b2_god = new_cons[IB2];
          const Real b3_god = new_cons[IB3];

          // ------------------------------------------------------------------
          // compute the updated magnetic field and interpolate to cell-center
          // ------------------------------------------------------------------
          // 1D //
          const Real bx_lower = StageB1FaceValue(
              B0, B1, coords, ndim, k, j, i, gam0, gam1, beta_dt);

          const Real bx_upper = StageB1FaceValue(
              B0, B1, coords, ndim, k, j, i + 1, gam0, gam1, beta_dt);

          const Real b1_ct = 0.5 * (bx_lower + bx_upper);

          Real b2_ct = b2_god;
          Real b3_ct = b3_god;

          // 2D //
          if (ndim >= 2) {
          const Real by_lower = StageB2FaceValue(
              B0, B1, coords, ndim, k, j, i, gam0, gam1, beta_dt);

          const Real by_upper = StageB2FaceValue(
              B0, B1, coords, ndim, k, j + 1, i, gam0, gam1, beta_dt);

          b2_ct = 0.5 * (by_lower + by_upper);
          }

          // 3D //
          if (ndim > 2) {
          const Real bz_lower = StageB3FaceValue(
              B0, B1, coords, ndim, k, j, i, gam0, gam1, beta_dt);

          const Real bz_upper = StageB3FaceValue(
              B0, B1, coords, ndim, k + 1, j, i, gam0, gam1, beta_dt);

          b3_ct = 0.5 * (bz_lower + bz_upper);
        }
          // ------------------------------------------------------------------
          // ------------------------------------------------------------------

          // ------------------------------------------------------------------
          // Apply ct energy correction if enabled
          // ------------------------------------------------------------------
          if (correct_ct_energy) {
            const Real emag_god = 0.5 * (SQR(b1_god) + SQR(b2_god) + SQR(b3_god));
            const Real emag_ct = 0.5 * (SQR(b1_ct) + SQR(b2_ct) + SQR(b3_ct));
            new_cons[IEN] += emag_ct - emag_god;
          }
          // ------------------------------------------------------------------
          // ------------------------------------------------------------------

          // ------------------------------------------------------------------
          // Replace the B values of cons with the cell-centered values given by the CT update
          // ------------------------------------------------------------------
          new_cons[IB1] = b1_ct;
          new_cons[IB2] = b2_ct;
          new_cons[IB3] = b3_ct;
          // ------------------------------------------------------------------
          // ------------------------------------------------------------------

}

KOKKOS_INLINE_FUNCTION
bool IsCTStateAdmissible(const parthenon::Real *state) {
    for (int v = 0; v < IB3+1; ++v) {
        if (!Kokkos::isfinite(state[v])) {
            return false;
        }
    }

    // Check density before dividing by it. Count each bad cell once.
    if (state[IDN] <= 0.0) {
        return false;
    }

    // Thermal energy density has the same sign as pressure for gamma > 1.
    const Real thermal_energy =
        state[IEN] -
        0.5 * (SQR(state[IM1]) + SQR(state[IM2]) + SQR(state[IM3])) /
            state[IDN] -
        0.5 * (SQR(state[IB1]) + SQR(state[IB2]) + SQR(state[IB3]));
    if (!Kokkos::isfinite(thermal_energy) || thermal_energy <= 0.0) {
        return false;
    }

    return true;
}

} // namespace Hydro::CTMHD

#endif // HYDRO_CTMHD_CT_FOFC_HELPER_HPP_