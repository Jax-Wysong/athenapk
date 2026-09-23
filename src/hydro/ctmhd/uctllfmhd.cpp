//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <cmath>

#include <parthenon/package.hpp>

#include "../../main.hpp"

using parthenon::Real;

#include "uct_edge_reconstruction.hpp"
#include "uctllfmhd.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

namespace Hydro::UCTLLFMHD {
using UCTEdgeReconstruction::ReconstructFaceToEdge;

template <int ORDER>
TaskStatus AssembleLLFLineEdgeEMFImpl(MeshData<Real> *md) {
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
          ReconstructFaceToEdge<ORDER, X2DIR>(
              eface, TE::F1, ELEFT_T2, k, j, i, f1_left_s, f1_left_n);
          ReconstructFaceToEdge<ORDER, X2DIR>(
              eface, TE::F1, ERIGHT_T2, k, j, i, f1_right_s, f1_right_n);

          // F2 is reconstructed along x1. Its LEFT/RIGHT states belong to
          // the south/north cells adjacent to the x2 face.
          Real f2_left_w, f2_left_e, f2_right_w, f2_right_e;
          ReconstructFaceToEdge<ORDER, X1DIR>(
              eface, TE::F2, ELEFT_T1, k, j, i, f2_left_w, f2_left_e);
          ReconstructFaceToEdge<ORDER, X1DIR>(
              eface, TE::F2, ERIGHT_T1, k, j, i, f2_right_w, f2_right_e);

          const Real EzSW = 0.5 * (f1_left_s + f2_left_w);
          const Real EzSE = 0.5 * (f1_right_s + f2_left_e);
          const Real EzNW = 0.5 * (f1_left_n + f2_right_w);
          const Real EzNE = 0.5 * (f1_right_n + f2_right_e);

          Real ByW, ByE;
          ReconstructFaceToEdge<ORDER, X1DIR>(
              Bface, TE::F2, 0, k, j, i, ByW, ByE);

          Real BxS, BxN;
          ReconstructFaceToEdge<ORDER, X2DIR>(
              Bface, TE::F1, 0, k, j, i, BxS, BxN);

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
          ReconstructFaceToEdge<ORDER, X1DIR>(
              eface, TE::F3, ELEFT_T2, k, j, i, f3_left_xlow, f3_left_xhigh);
          ReconstructFaceToEdge<ORDER, X1DIR>(
              eface, TE::F3, ERIGHT_T2, k, j, i, f3_right_xlow,
              f3_right_xhigh);

          Real f1_left_zlow, f1_left_zhigh, f1_right_zlow,
              f1_right_zhigh;
          ReconstructFaceToEdge<ORDER, X3DIR>(
              eface, TE::F1, ELEFT_T1, k, j, i, f1_left_zlow, f1_left_zhigh);
          ReconstructFaceToEdge<ORDER, X3DIR>(
              eface, TE::F1, ERIGHT_T1, k, j, i, f1_right_zlow,
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
          ReconstructFaceToEdge<ORDER, X3DIR>(
              Bface, TE::F1, 0, k, j, i, BxZlow, BxZhigh);

          Real BzXlow, BzXhigh;
          ReconstructFaceToEdge<ORDER, X1DIR>(
              Bface, TE::F3, 0, k, j, i, BzXlow, BzXhigh);

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
          ReconstructFaceToEdge<ORDER, X3DIR>(
              eface, TE::F2, ELEFT_T2, k, j, i, f2_left_zlow, f2_left_zhigh);
          ReconstructFaceToEdge<ORDER, X3DIR>(
              eface, TE::F2, ERIGHT_T2, k, j, i, f2_right_zlow,
              f2_right_zhigh);

          Real f3_left_ylow, f3_left_yhigh, f3_right_ylow,
              f3_right_yhigh;
          ReconstructFaceToEdge<ORDER, X2DIR>(
              eface, TE::F3, ELEFT_T1, k, j, i, f3_left_ylow, f3_left_yhigh);
          ReconstructFaceToEdge<ORDER, X2DIR>(
              eface, TE::F3, ERIGHT_T1, k, j, i, f3_right_ylow,
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
          ReconstructFaceToEdge<ORDER, X2DIR>(
              Bface, TE::F3, 0, k, j, i, BzYlow, BzYhigh);

          Real ByZlow, ByZhigh;
          ReconstructFaceToEdge<ORDER, X3DIR>(
              Bface, TE::F2, 0, k, j, i, ByZlow, ByZhigh);

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

TaskStatus Assemble_LLF_Line_Edge_EMF(MeshData<Real> *md) {
  const auto pkg = md->GetBlockData(0)->GetBlockPointer()->packages.Get("Hydro");
  const int order = pkg->Param<int>("convergence_order");
  if (order == 4) return AssembleLLFLineEdgeEMFImpl<4>(md);
  if (order == 6) return AssembleLLFLineEdgeEMFImpl<6>(md);
  if (order == 8) return AssembleLLFLineEdgeEMFImpl<8>(md);
  PARTHENON_FAIL("Unsupported Teissier LLF-line edge-EMF order");
}

} // namespace Hydro::UCTLLFMHD
