#ifndef HYDRO_CTMHD_UCT_EDGE_RECONSTRUCTION_HPP_
#define HYDRO_CTMHD_UCT_EDGE_RECONSTRUCTION_HPP_

#include <parthenon/package.hpp>

#include "../../recon/weno5_classic.hpp"
#include "../../recon/weno7_classic.hpp"
#include "../../recon/weno9_classic.hpp"


namespace Hydro::UCTEdgeReconstruction {

using parthenon::Real;
using TE = parthenon::TopologicalElement;

// The interface lies between samples at offsets -1 and 0. Each side uses
// its own WENO stencil, with the unused output discarded as in the LLF route.
KOKKOS_INLINE_FUNCTION void ReconstructWENO5Interface(
    const Real q_m3, const Real q_m2, const Real q_m1, const Real q_0,
    const Real q_p1, const Real q_p2, Real &from_lower, Real &from_upper) {
  Real unused;
  WENO5JS(q_m3, q_m2, q_m1, q_0, q_p1, from_lower, unused);
  WENO5JS(q_m2, q_m1, q_0, q_p1, q_p2, unused, from_upper);
}

KOKKOS_INLINE_FUNCTION void ReconstructWENO7Interface(
    const Real q_m4, const Real q_m3, const Real q_m2, const Real q_m1,
    const Real q_0, const Real q_p1, const Real q_p2, const Real q_p3,
    Real &from_lower, Real &from_upper) {
  Real unused;
  WENO7JS(q_m4, q_m3, q_m2, q_m1, q_0, q_p1, q_p2, from_lower, unused);
  WENO7JS(q_m3, q_m2, q_m1, q_0, q_p1, q_p2, q_p3, unused, from_upper);
}

KOKKOS_INLINE_FUNCTION void ReconstructWENO9Interface(
    const Real q_m5, const Real q_m4, const Real q_m3, const Real q_m2, const Real q_m1,
    const Real q_0, const Real q_p1, const Real q_p2, const Real q_p3, const Real q_p4,
    Real &from_lower, Real &from_upper) {
  Real unused;
  WENO9JS(q_m5, q_m4, q_m3, q_m2, q_m1, q_0, q_p1, q_p2, q_p3,
          from_lower, unused);
  WENO9JS(q_m4, q_m3, q_m2, q_m1, q_0, q_p1, q_p2, q_p3, q_p4,
          unused, from_upper);
}

template <int DIR, typename FacePack>
KOKKOS_INLINE_FUNCTION Real ReadFaceOffset(const FacePack &face_values,
                                           const TE face, const int n,
                                           const int k, const int j, const int i,
                                           const int offset) {
  static_assert(DIR == parthenon::X1DIR || DIR == parthenon::X2DIR ||
                DIR == parthenon::X3DIR);
  if constexpr (DIR == parthenon::X1DIR) {
    return face_values(face, n, k, j, i + offset);
  } else if constexpr (DIR == parthenon::X2DIR) {
    return face_values(face, n, k, j + offset, i);
  } else {
    return face_values(face, n, k + offset, j, i);
  }
}

template <int ORDER, int DIR, typename FacePack>
KOKKOS_INLINE_FUNCTION void ReconstructFaceToEdge(
    const FacePack &face_values, const TE face, const int n, const int k,
    const int j, const int i, Real &from_lower, Real &from_upper) {
  static_assert(ORDER == 4 || ORDER == 6 || ORDER == 8);
  if constexpr (ORDER == 4) {
    ReconstructWENO5Interface(
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -3),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -2),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -1),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 0),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 1),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 2),
        from_lower, from_upper);
  } else if constexpr (ORDER == 6) {
    ReconstructWENO7Interface(
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -4),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -3),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -2),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -1),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 0),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 1),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 2),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 3),
        from_lower, from_upper);
  } else if constexpr (ORDER == 8) {
    ReconstructWENO9Interface(
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -5),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -4),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -3),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -2),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, -1),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 0),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 1),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 2),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 3),
        ReadFaceOffset<DIR>(face_values, face, n, k, j, i, 4),
        from_lower, from_upper);
  } 
}

} // namespace Hydro::UCTEdgeReconstruction

#endif // HYDRO_CTMHD_UCT_EDGE_RECONSTRUCTION_HPP_
