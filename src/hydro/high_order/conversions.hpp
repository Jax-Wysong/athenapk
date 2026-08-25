#ifndef HYDRO_CONVERSIONS_HPP_
#define HYDRO_CONVERSIONS_HPP_
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

TaskStatus PointToAreaAverage(MeshData<Real> *md);
TaskStatus PointEmfToLineAverage(MeshData<Real> *md);
TaskStatus ReconcileSelfPeriodicFluxesAndEmfs(MeshData<Real> *md);
TaskStatus PointElectricFieldsToAreaAverage(MeshData<Real> *md);

} // namespace Hydro::HIGHORDER

#endif // HYDRO_CONVERSIONS_HPP_
