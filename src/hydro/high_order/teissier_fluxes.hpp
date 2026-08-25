#ifndef HYDRO_HIGH_ORDER_TEISSIER_FLUXES_HPP_
#define HYDRO_HIGH_ORDER_TEISSIER_FLUXES_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::HIGHORDER {

TaskStatus ReconstructFaceAveragesTeissier(MeshData<Real> *md);
TaskStatus CalculatePointFluxesTeissier(MeshData<Real> *md);


} // namespace Hydro::HIGHORDER

#endif // HYDRO_HIGH_ORDER_TEISSIER_FLUXES_HPP_
