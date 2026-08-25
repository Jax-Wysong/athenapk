//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <cmath>

#include <parthenon/package.hpp>

#include "../../main.hpp"

using parthenon::Real;

#include "../../recon/weno5_classic.hpp"
#include "uctllfmhd.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

namespace Hydro::UCTLLFMHD {
namespace {

// Reconstruct the two states at the interface between the lower-index and
// upper-index cells. Each WENO5-JS state has its own five-point stencil.
KOKKOS_INLINE_FUNCTION
void ReconstructWENO5Interface(const Real q_m3, const Real q_m2,
                               const Real q_m1, const Real q_0,
                               const Real q_p1, const Real q_p2,
                               Real &q_from_lower, Real &q_from_upper) {
  Real unused;
  WENO5JS(q_m3, q_m2, q_m1, q_0, q_p1, q_from_lower, unused);
  WENO5JS(q_m2, q_m1, q_0, q_p1, q_p2, unused, q_from_upper);
}

} // namespace

TaskStatus Assemble_LLF_Line_Edge_EMF(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto Bface_pack =
      md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  auto eface_avg_pack =
      md->PackVariables(std::vector<std::string>{"teissier_eface_avg"});
  auto llf_speed_pack =
      md->PackVariables(std::vector<std::string>{"teissier_llf_speed"});

  const IndexRange ib =
      md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  const IndexRange jb =
      md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  const IndexRange kb =
      md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // E_z at the intersection of x1- and x2-normal faces.
  if (ndim > 1) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Teissier UCT-LLF Ez edges",
        parthenon::DevExecSpace(), 0, Bface_pack.GetDim(5) - 1, kb.s,
        kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &Bface = Bface_pack(b);
          const auto &eface = eface_avg_pack(b);
          const auto &speed = llf_speed_pack(b);

          // F1 is reconstructed along x2. Its LEFT/RIGHT states belong to
          // the west/east cells adjacent to the x1 face.
          Real f1_left_s, f1_left_n, f1_right_s, f1_right_n;
          ReconstructWENO5Interface(
              eface(TE::F1, ELEFT_T2, k, j - 3, i),
              eface(TE::F1, ELEFT_T2, k, j - 2, i),
              eface(TE::F1, ELEFT_T2, k, j - 1, i),
              eface(TE::F1, ELEFT_T2, k, j, i),
              eface(TE::F1, ELEFT_T2, k, j + 1, i),
              eface(TE::F1, ELEFT_T2, k, j + 2, i), f1_left_s,
              f1_left_n);
          ReconstructWENO5Interface(
              eface(TE::F1, ERIGHT_T2, k, j - 3, i),
              eface(TE::F1, ERIGHT_T2, k, j - 2, i),
              eface(TE::F1, ERIGHT_T2, k, j - 1, i),
              eface(TE::F1, ERIGHT_T2, k, j, i),
              eface(TE::F1, ERIGHT_T2, k, j + 1, i),
              eface(TE::F1, ERIGHT_T2, k, j + 2, i), f1_right_s,
              f1_right_n);

          // F2 is reconstructed along x1. Its LEFT/RIGHT states belong to
          // the south/north cells adjacent to the x2 face.
          Real f2_left_w, f2_left_e, f2_right_w, f2_right_e;
          ReconstructWENO5Interface(
              eface(TE::F2, ELEFT_T1, k, j, i - 3),
              eface(TE::F2, ELEFT_T1, k, j, i - 2),
              eface(TE::F2, ELEFT_T1, k, j, i - 1),
              eface(TE::F2, ELEFT_T1, k, j, i),
              eface(TE::F2, ELEFT_T1, k, j, i + 1),
              eface(TE::F2, ELEFT_T1, k, j, i + 2), f2_left_w,
              f2_left_e);
          ReconstructWENO5Interface(
              eface(TE::F2, ERIGHT_T1, k, j, i - 3),
              eface(TE::F2, ERIGHT_T1, k, j, i - 2),
              eface(TE::F2, ERIGHT_T1, k, j, i - 1),
              eface(TE::F2, ERIGHT_T1, k, j, i),
              eface(TE::F2, ERIGHT_T1, k, j, i + 1),
              eface(TE::F2, ERIGHT_T1, k, j, i + 2), f2_right_w,
              f2_right_e);

          const Real EzSW = 0.5 * (f1_left_s + f2_left_w);
          const Real EzSE = 0.5 * (f1_right_s + f2_left_e);
          const Real EzNW = 0.5 * (f1_left_n + f2_right_w);
          const Real EzNE = 0.5 * (f1_right_n + f2_right_e);

          Real ByW, ByE;
          ReconstructWENO5Interface(
              Bface(TE::F2, 0, k, j, i - 3),
              Bface(TE::F2, 0, k, j, i - 2),
              Bface(TE::F2, 0, k, j, i - 1),
              Bface(TE::F2, 0, k, j, i),
              Bface(TE::F2, 0, k, j, i + 1),
              Bface(TE::F2, 0, k, j, i + 2), ByW, ByE);

          Real BxS, BxN;
          ReconstructWENO5Interface(
              Bface(TE::F1, 0, k, j - 3, i),
              Bface(TE::F1, 0, k, j - 2, i),
              Bface(TE::F1, 0, k, j - 1, i),
              Bface(TE::F1, 0, k, j, i),
              Bface(TE::F1, 0, k, j + 1, i),
              Bface(TE::F1, 0, k, j + 2, i), BxS, BxN);

          const Real axS = speed(TE::F1, 0, k, j - 1, i);
          const Real axN = speed(TE::F1, 0, k, j, i);
          const Real ayW = speed(TE::F2, 0, k, j, i - 1);
          const Real ayE = speed(TE::F2, 0, k, j, i);
          const Real S = fmax(fmax(axS, axN), fmax(ayW, ayE));

          const Real Ez = 0.25 * (EzSW + EzSE + EzNW + EzNE) +
                          0.5 * S * (ByE - ByW) -
                          0.5 * S * (BxN - BxS);
          Bface.template flux<parthenon::TopologicalType::Edge>(
              X3DIR, 0, k, j, i) = Ez;
        });
  }

  if (ndim > 2) {
    // E_y at the intersection of x3- and x1-normal faces. Positional
    // directions are z (lower/upper) and x (lower/upper).
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Teissier UCT-LLF Ey edges",
        parthenon::DevExecSpace(), 0, Bface_pack.GetDim(5) - 1, kb.s,
        kb.e + 1, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &Bface = Bface_pack(b);
          const auto &eface = eface_avg_pack(b);
          const auto &speed = llf_speed_pack(b);

          // F3 is reconstructed along x1; F1 is reconstructed along x3.
          Real f3_left_xlow, f3_left_xhigh, f3_right_xlow,
              f3_right_xhigh;
          ReconstructWENO5Interface(
              eface(TE::F3, ELEFT_T2, k, j, i - 3),
              eface(TE::F3, ELEFT_T2, k, j, i - 2),
              eface(TE::F3, ELEFT_T2, k, j, i - 1),
              eface(TE::F3, ELEFT_T2, k, j, i),
              eface(TE::F3, ELEFT_T2, k, j, i + 1),
              eface(TE::F3, ELEFT_T2, k, j, i + 2), f3_left_xlow,
              f3_left_xhigh);
          ReconstructWENO5Interface(
              eface(TE::F3, ERIGHT_T2, k, j, i - 3),
              eface(TE::F3, ERIGHT_T2, k, j, i - 2),
              eface(TE::F3, ERIGHT_T2, k, j, i - 1),
              eface(TE::F3, ERIGHT_T2, k, j, i),
              eface(TE::F3, ERIGHT_T2, k, j, i + 1),
              eface(TE::F3, ERIGHT_T2, k, j, i + 2), f3_right_xlow,
              f3_right_xhigh);

          Real f1_left_zlow, f1_left_zhigh, f1_right_zlow,
              f1_right_zhigh;
          ReconstructWENO5Interface(
              eface(TE::F1, ELEFT_T1, k - 3, j, i),
              eface(TE::F1, ELEFT_T1, k - 2, j, i),
              eface(TE::F1, ELEFT_T1, k - 1, j, i),
              eface(TE::F1, ELEFT_T1, k, j, i),
              eface(TE::F1, ELEFT_T1, k + 1, j, i),
              eface(TE::F1, ELEFT_T1, k + 2, j, i), f1_left_zlow,
              f1_left_zhigh);
          ReconstructWENO5Interface(
              eface(TE::F1, ERIGHT_T1, k - 3, j, i),
              eface(TE::F1, ERIGHT_T1, k - 2, j, i),
              eface(TE::F1, ERIGHT_T1, k - 1, j, i),
              eface(TE::F1, ERIGHT_T1, k, j, i),
              eface(TE::F1, ERIGHT_T1, k + 1, j, i),
              eface(TE::F1, ERIGHT_T1, k + 2, j, i), f1_right_zlow,
              f1_right_zhigh);

          const Real EyZlowXlow =
              0.5 * (f3_left_xlow + f1_left_zlow);
          const Real EyZlowXhigh =
              0.5 * (f3_left_xhigh + f1_right_zlow);
          const Real EyZhighXlow =
              0.5 * (f3_right_xlow + f1_left_zhigh);
          const Real EyZhighXhigh =
              0.5 * (f3_right_xhigh + f1_right_zhigh);

          Real BxZlow, BxZhigh;
          ReconstructWENO5Interface(
              Bface(TE::F1, 0, k - 3, j, i),
              Bface(TE::F1, 0, k - 2, j, i),
              Bface(TE::F1, 0, k - 1, j, i),
              Bface(TE::F1, 0, k, j, i),
              Bface(TE::F1, 0, k + 1, j, i),
              Bface(TE::F1, 0, k + 2, j, i), BxZlow, BxZhigh);

          Real BzXlow, BzXhigh;
          ReconstructWENO5Interface(
              Bface(TE::F3, 0, k, j, i - 3),
              Bface(TE::F3, 0, k, j, i - 2),
              Bface(TE::F3, 0, k, j, i - 1),
              Bface(TE::F3, 0, k, j, i),
              Bface(TE::F3, 0, k, j, i + 1),
              Bface(TE::F3, 0, k, j, i + 2), BzXlow, BzXhigh);

          const Real azXlow = speed(TE::F3, 0, k, j, i - 1);
          const Real azXhigh = speed(TE::F3, 0, k, j, i);
          const Real axZlow = speed(TE::F1, 0, k - 1, j, i);
          const Real axZhigh = speed(TE::F1, 0, k, j, i);
          const Real S =
              fmax(fmax(azXlow, azXhigh), fmax(axZlow, axZhigh));

          const Real Ey =
              0.25 * (EyZlowXlow + EyZlowXhigh + EyZhighXlow +
                      EyZhighXhigh) +
              0.5 * S * (BxZhigh - BxZlow) -
              0.5 * S * (BzXhigh - BzXlow);
          Bface.template flux<parthenon::TopologicalType::Edge>(
              X2DIR, 0, k, j, i) = Ey;
        });

    // E_x at the intersection of x2- and x3-normal faces. Positional
    // directions are y (lower/upper) and z (lower/upper).
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Teissier UCT-LLF Ex edges",
        parthenon::DevExecSpace(), 0, Bface_pack.GetDim(5) - 1, kb.s,
        kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &Bface = Bface_pack(b);
          const auto &eface = eface_avg_pack(b);
          const auto &speed = llf_speed_pack(b);

          // F2 is reconstructed along x3; F3 is reconstructed along x2.
          Real f2_left_zlow, f2_left_zhigh, f2_right_zlow,
              f2_right_zhigh;
          ReconstructWENO5Interface(
              eface(TE::F2, ELEFT_T2, k - 3, j, i),
              eface(TE::F2, ELEFT_T2, k - 2, j, i),
              eface(TE::F2, ELEFT_T2, k - 1, j, i),
              eface(TE::F2, ELEFT_T2, k, j, i),
              eface(TE::F2, ELEFT_T2, k + 1, j, i),
              eface(TE::F2, ELEFT_T2, k + 2, j, i), f2_left_zlow,
              f2_left_zhigh);
          ReconstructWENO5Interface(
              eface(TE::F2, ERIGHT_T2, k - 3, j, i),
              eface(TE::F2, ERIGHT_T2, k - 2, j, i),
              eface(TE::F2, ERIGHT_T2, k - 1, j, i),
              eface(TE::F2, ERIGHT_T2, k, j, i),
              eface(TE::F2, ERIGHT_T2, k + 1, j, i),
              eface(TE::F2, ERIGHT_T2, k + 2, j, i), f2_right_zlow,
              f2_right_zhigh);

          Real f3_left_ylow, f3_left_yhigh, f3_right_ylow,
              f3_right_yhigh;
          ReconstructWENO5Interface(
              eface(TE::F3, ELEFT_T1, k, j - 3, i),
              eface(TE::F3, ELEFT_T1, k, j - 2, i),
              eface(TE::F3, ELEFT_T1, k, j - 1, i),
              eface(TE::F3, ELEFT_T1, k, j, i),
              eface(TE::F3, ELEFT_T1, k, j + 1, i),
              eface(TE::F3, ELEFT_T1, k, j + 2, i), f3_left_ylow,
              f3_left_yhigh);
          ReconstructWENO5Interface(
              eface(TE::F3, ERIGHT_T1, k, j - 3, i),
              eface(TE::F3, ERIGHT_T1, k, j - 2, i),
              eface(TE::F3, ERIGHT_T1, k, j - 1, i),
              eface(TE::F3, ERIGHT_T1, k, j, i),
              eface(TE::F3, ERIGHT_T1, k, j + 1, i),
              eface(TE::F3, ERIGHT_T1, k, j + 2, i), f3_right_ylow,
              f3_right_yhigh);

          const Real ExYlowZlow =
              0.5 * (f2_left_zlow + f3_left_ylow);
          const Real ExYlowZhigh =
              0.5 * (f2_left_zhigh + f3_right_ylow);
          const Real ExYhighZlow =
              0.5 * (f2_right_zlow + f3_left_yhigh);
          const Real ExYhighZhigh =
              0.5 * (f2_right_zhigh + f3_right_yhigh);

          Real BzYlow, BzYhigh;
          ReconstructWENO5Interface(
              Bface(TE::F3, 0, k, j - 3, i),
              Bface(TE::F3, 0, k, j - 2, i),
              Bface(TE::F3, 0, k, j - 1, i),
              Bface(TE::F3, 0, k, j, i),
              Bface(TE::F3, 0, k, j + 1, i),
              Bface(TE::F3, 0, k, j + 2, i), BzYlow, BzYhigh);

          Real ByZlow, ByZhigh;
          ReconstructWENO5Interface(
              Bface(TE::F2, 0, k - 3, j, i),
              Bface(TE::F2, 0, k - 2, j, i),
              Bface(TE::F2, 0, k - 1, j, i),
              Bface(TE::F2, 0, k, j, i),
              Bface(TE::F2, 0, k + 1, j, i),
              Bface(TE::F2, 0, k + 2, j, i), ByZlow, ByZhigh);

          const Real ayZlow = speed(TE::F2, 0, k - 1, j, i);
          const Real ayZhigh = speed(TE::F2, 0, k, j, i);
          const Real azYlow = speed(TE::F3, 0, k, j - 1, i);
          const Real azYhigh = speed(TE::F3, 0, k, j, i);
          const Real S =
              fmax(fmax(ayZlow, ayZhigh), fmax(azYlow, azYhigh));

          const Real Ex =
              0.25 * (ExYlowZlow + ExYlowZhigh + ExYhighZlow +
                      ExYhighZhigh) +
              0.5 * S * (BzYhigh - BzYlow) -
              0.5 * S * (ByZhigh - ByZlow);
          Bface.template flux<parthenon::TopologicalType::Edge>(
              X1DIR, 0, k, j, i) = Ex;
        });
  }

  return TaskStatus::complete;
}

} // namespace Hydro::UCTLLFMHD
