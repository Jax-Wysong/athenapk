#ifndef HYDRO_CTMHD_UCTLLFMHD_HPP_
#define HYDRO_CTMHD_UCTLLFMHD_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <parthenon/package.hpp>

#include "../../main.hpp"

using namespace parthenon::package::prelude;

namespace Hydro::UCTLLFMHD {

TaskStatus Assemble_LLF_Line_Edge_EMF(MeshData<Real> *md);

} // namespace Hydro::UCTLLFMHD

#endif // HYDRO_CTMHD_UCTLLFMHD_HPP_
