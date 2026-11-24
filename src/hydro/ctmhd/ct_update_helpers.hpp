// This file was made in part with generative AI
#ifndef HYDRO_CTMHD_CT_UPDATE_HELPERS_HPP_
#define HYDRO_CTMHD_CT_UPDATE_HELPERS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>


using namespace parthenon::package::prelude;

namespace Hydro::CTMHD {
using TE = parthenon::TopologicalElement;


template <typename FacePack, typename Coordinates>
KOKKOS_INLINE_FUNCTION
parthenon::Real StageB1FaceValue(
    const FacePack &B0, const FacePack &B1,
    const Coordinates &coords, const int ndim,
    const int k, const int j, const int i,
    const parthenon::Real gam0, const parthenon::Real gam1,
    const parthenon::Real beta_dt) {

        const Real b0n = B0(TE::F1, 0, k, j, i);
        const Real b1n = B1(TE::F1, 0, k, j, i);

        Real result = gam0 * b0n + gam1 * b1n;

        // Only Bx is face-evolved in 1D. No edge fluxes exist in 1D;
        // By/Bz remain in the cell-centered FV update.
        if (ndim == 1) return result;
        const Real dy = coords.template Dxc<2>(k, j, i);

        const Real emfz_corner_j_i =
            B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
        const Real emfz_corner_jp1_i =
            B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j + 1, i);
        result += beta_dt * (emfz_corner_j_i - emfz_corner_jp1_i) / dy;

        if (ndim >= 3) {
          const Real dz = coords.template Dxc<3>(k, j, i);
          const Real emfy_corner_k_i =
              B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);
          const Real emfy_corner_kp1_i =
              B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k + 1, j, i);
          result += -beta_dt * (emfy_corner_k_i - emfy_corner_kp1_i) / dz;
        }

        return result;

}

template <typename FacePack, typename Coordinates>
KOKKOS_INLINE_FUNCTION
parthenon::Real StageB2FaceValue(
    const FacePack &B0, const FacePack &B1,
    const Coordinates &coords, const int ndim,
    const int k, const int j, const int i,
    const parthenon::Real gam0, const parthenon::Real gam1,
    const parthenon::Real beta_dt) {

        const Real b0n = B0(TE::F2, 0, k, j, i);
        const Real b1n = B1(TE::F2, 0, k, j, i);

        Real result = gam0 * b0n + gam1 * b1n;

        const Real dx = coords.template Dxc<1>(k, j, i); // cell-spacing 'width' in x

        const Real emfz_corner_j_i =
            B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
        const Real emfz_corner_j_ip1 =
            B0.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i + 1);

        result += -beta_dt * (emfz_corner_j_i - emfz_corner_j_ip1) / dx;

          if (ndim >= 3) {
            const Real dz = coords.template Dxc<3>(k, j, i);
            const Real emfx_corner_j_k =
                B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);
            const Real emfx_corner_j_kp1 =
                B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k + 1, j, i);

            result += beta_dt * (emfx_corner_j_k - emfx_corner_j_kp1) / dz;
          }

        return result;

}

template <typename FacePack, typename Coordinates>
KOKKOS_INLINE_FUNCTION
parthenon::Real StageB3FaceValue(
    const FacePack &B0, const FacePack &B1,
    const Coordinates &coords, const int ndim,
    const int k, const int j, const int i,
    const parthenon::Real gam0, const parthenon::Real gam1,
    const parthenon::Real beta_dt) {

        const Real b0n = B0(TE::F3, 0, k, j, i);
        const Real b1n = B1(TE::F3, 0, k, j, i);

        Real result = gam0 * b0n + gam1 * b1n;

        const Real dx = coords.template Dxc<1>(k, j, i); // cell-spacing 'width' in x
        const Real dy = coords.template Dxc<2>(k, j, i);

        const Real emfy_corner_k_i =
            B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);
        const Real emfy_corner_k_ip1 =
            B0.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i + 1);

        const Real emfx_corner_j_k =
            B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);
        const Real emfx_corner_jp1_k =
            B0.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j + 1, i);

        result += beta_dt * (emfy_corner_k_i - emfy_corner_k_ip1) / dx;
        result += -beta_dt * (emfx_corner_j_k - emfx_corner_jp1_k) / dy;

        return result;

}

} // namespace Hydro::CTMHD

#endif // HYDRO_CTMHD_CT_UPDATE_HELPERS_HPP_
