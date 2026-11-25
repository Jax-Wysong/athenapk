// This file was made in part with generative AI
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>

// Parthenon headers
#include <parthenon/package.hpp>
#include "interface/update.hpp"


// AthenaPK headers
#include "../../main.hpp"
#include "ctmhd.hpp"
#include "ct_update_helpers.hpp"
#include "ct_fofc_helper.hpp"

// Kokkos headers
#include <Kokkos_MathematicalFunctions.hpp>


using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

namespace Hydro::CTMHD {

TaskStatus Assemble_Corner_EMF(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  auto cons_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  // the edge EMFs will be stored in the fluxes of Bface
  auto Bface_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});

  const auto &prim_pack = md->PackVariables(std::vector<std::string>{"prim"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  // all steps and naming are exactly valid when z is the normal direction
  // which by itself is valid for 2D CT.
  // When doing 3D CT, follow the same rules but permute cyclically x->y->z->x
  //  ------------------- Step 1 ---------------------
  // |    Extract face EMFs from fluxes               |
  // |    EMFx_face = -Fx(By) = -Fx(6)                |
  // |    EMFy_face =  Fy(Bx) =  Fy(5)                |
  //  ------------------------------------------------

  //  ------------------- Step 2 ---------------------
  // | Create cell-centered EMF needed for updating   |
  // | (need prim for this)                           |
  // | ux = w(1), uy = w(2),                          |
  // | Bx = w(5), By = w(6)                           |
  // | EMFcc = uy*Bx - ux*By (E = -v x B)             |
  //  ------------------------------------------------

  //  ------------------- Step 3 ---------------------
  // | get rho_ux at faces from the flux              |
  // | rho_ux_face = Fx(0)                            |
  // | rho_uy_face = Fy(0)                            |
  //  ------------------------------------------------

  //  ------------------ Step 4 ----------------------
  // |    interpolate EMF_face to cell corners via    |
  // |    Gardiner and Stone (2005) upwind scheme     |
  // |    need the following:                         |
  // |        EMFx_face, EMFy_face                    |
  // |        EMF_cc                                  |
  // |        rho_ux = Fx(0)                          |
  // |        rho_uy = Fy(0)                          |
  //  ------------------------------------------------

  // loops need to run over the interior domain + 1 so
  // they reach the end of the mesh
  // for z-normal direction (Ez_edges)
  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Assemble Ez_edges", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        auto &Bface = Bface_pack(b);
        const auto &prim = prim_pack(b);
        const auto &coords = prim_pack.GetCoords(b);
        const Real dx = coords.Dxc<1>(k, j, i); // cell-spacing 'width' in x
        const Real dy = coords.Dxc<2>(k, j, i);

        // |----------- Step 1 -----------|
        // the edge EMFs will be stored in the flux of Bface, Artemis-style (mhd.cpp)
        //                                     x-flux, By-var
        const Real emfx_face_j_i = -cons.flux(X1DIR, IB2, k, j, i);
        const Real emfx_face_jm1_i = -cons.flux(X1DIR, IB2, k, j - 1, i);
        //                                     y-flux, Bx-var
        const Real emfy_face_j_i = cons.flux(X2DIR, IB1, k, j, i);
        const Real emfy_face_j_im1 = cons.flux(X2DIR, IB1, k, j, i - 1);
        //  |------------------------------|

        // |----------- Step 2 -----------|
        // const Real &ux = prim(IV1, k, j, i);
        // const Real &uy = prim(IV2, k, j, i);
        // const Real &bx = prim(IB1, k, j, i);
        // const Real &by = prim(IB2, k, j, i);

        // emfz_cell_center = uy*bx - ux*by
        const Real emfz_cell_j_i = prim(IV2, k, j, i) * prim(IB1, k, j, i) -
                                   prim(IV1, k, j, i) * prim(IB2, k, j, i);
        const Real emfz_cell_jm1_i = prim(IV2, k, j - 1, i) * prim(IB1, k, j - 1, i) -
                                     prim(IV1, k, j - 1, i) * prim(IB2, k, j - 1, i);
        const Real emfz_cell_jm1_im1 =
            prim(IV2, k, j - 1, i - 1) * prim(IB1, k, j - 1, i - 1) -
            prim(IV1, k, j - 1, i - 1) * prim(IB2, k, j - 1, i - 1);
        const Real emfz_cell_j_im1 = prim(IV2, k, j, i - 1) * prim(IB1, k, j, i - 1) -
                                     prim(IV1, k, j, i - 1) * prim(IB2, k, j, i - 1);
        // |------------------------------|

        // |----------- Step 3 -----------|
        const Real rho_ux_face_j_i = cons.flux(X1DIR, IDN, k, j, i);
        const Real rho_ux_face_jm1_i = cons.flux(X1DIR, IDN, k, j - 1, i);

        const Real rho_uy_face_j_i = cons.flux(X2DIR, IDN, k, j, i);
        const Real rho_uy_face_j_im1 = cons.flux(X2DIR, IDN, k, j, i - 1);
        // NOTE! what the upwinded scheme really calls for is the sign of ux and uy
        // at the face, but since rho is (should be) positive, using
        // rho_ux and rho_uy should be fine
        // |------------------------------|

        // |----------- Step 4 -----------|
        // Store the edge electric field in the face variable's flux array.
        Real &emfz_corner =
            Bface.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
        Real dEz_dy_xp_quarter = 0;
        Real dEz_dy_xp_three_quarter = 0;
        Real dEz_dx_yp_quarter = 0;
        Real dEz_dx_yp_three_quarter = 0;

        // upwinded corner emfs, following GS05
        if (rho_ux_face_jm1_i > 0) {
          dEz_dy_xp_quarter = 2.0 * (emfy_face_j_im1 - emfz_cell_jm1_im1) / dy;
        } else if (rho_ux_face_jm1_i < 0) {
          dEz_dy_xp_quarter = 2.0 * (emfy_face_j_i - emfz_cell_jm1_i) / dy;
        } else {
          dEz_dy_xp_quarter = 0.5 * (2.0 * (emfy_face_j_im1 - emfz_cell_jm1_im1) / dy +
                                     2.0 * (emfy_face_j_i - emfz_cell_jm1_i) / dy);
        }

        if (rho_ux_face_j_i > 0) {
          dEz_dy_xp_three_quarter = 2.0 * (emfz_cell_j_im1 - emfy_face_j_im1) / dy;
        } else if (rho_ux_face_j_i < 0) {
          dEz_dy_xp_three_quarter = 2.0 * (emfz_cell_j_i - emfy_face_j_i) / dy;
        } else {
          dEz_dy_xp_three_quarter =
              0.5 * (2.0 * (emfz_cell_j_im1 - emfy_face_j_im1) / dy +
                     2.0 * (emfz_cell_j_i - emfy_face_j_i) / dy);
        }

        if (rho_uy_face_j_im1 > 0) {
          dEz_dx_yp_quarter = 2.0 * (emfx_face_jm1_i - emfz_cell_jm1_im1) / dx;
        } else if (rho_uy_face_j_im1 < 0) {
          dEz_dx_yp_quarter = 2.0 * (emfx_face_j_i - emfz_cell_j_im1) / dx;
        } else {
          dEz_dx_yp_quarter = 0.5 * (2.0 * (emfx_face_jm1_i - emfz_cell_jm1_im1) / dx +
                                     2.0 * (emfx_face_j_i - emfz_cell_j_im1) / dx);
        }

        if (rho_uy_face_j_i > 0) {
          dEz_dx_yp_three_quarter = 2.0 * (emfz_cell_jm1_i - emfx_face_jm1_i) / dx;
        } else if (rho_uy_face_j_i < 0) {
          dEz_dx_yp_three_quarter = 2.0 * (emfz_cell_j_i - emfx_face_j_i) / dx;
        } else {
          dEz_dx_yp_three_quarter =
              0.5 * (2.0 * (emfz_cell_jm1_i - emfx_face_jm1_i) / dx +
                     2.0 * (emfz_cell_j_i - emfx_face_j_i) / dx);
        }

        // finally
        emfz_corner =
            (0.25 * (emfx_face_jm1_i + emfx_face_j_i + emfy_face_j_im1 + emfy_face_j_i) +
             dy / 8.0 * (dEz_dy_xp_quarter - dEz_dy_xp_three_quarter) +
             dx / 8.0 * (dEz_dx_yp_quarter - dEz_dx_yp_three_quarter));
      });

  // for y-normal direction (Ey_edges)
  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ey_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &cons = cons_pack(b);
          auto &Bface = Bface_pack(b);
          const auto &prim = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real dx = coords.Dxc<1>(k, j, i); // cell-spacing 'width' in x
          const Real dz = coords.Dxc<3>(k, j, i);

          // |----------- Step 1 -----------|
          // the edge EMFs will be stored in the flux of Bface, Artemis-style (mhd.cpp)
          //                                     x-flux, Bz-var
          const Real emfx_face_k_i = cons.flux(X1DIR, IB3, k, j, i);
          const Real emfx_face_km1_i = cons.flux(X1DIR, IB3, k - 1, j, i);
          //                                     z-flux, Bx-var
          const Real emfz_face_k_i = -cons.flux(X3DIR, IB1, k, j, i);
          const Real emfz_face_k_im1 = -cons.flux(X3DIR, IB1, k, j, i - 1);
          //  |------------------------------|

          // |----------- Step 2 -----------|

          // emfy_cell_center = uxBz - vzBx
          const Real emfy_cell_k_i = -prim(IV3, k, j, i) * prim(IB1, k, j, i) +
                                     prim(IV1, k, j, i) * prim(IB3, k, j, i);
          const Real emfy_cell_km1_i = -prim(IV3, k - 1, j, i) * prim(IB1, k - 1, j, i) +
                                       prim(IV1, k - 1, j, i) * prim(IB3, k - 1, j, i);
          const Real emfy_cell_km1_im1 =
              -prim(IV3, k - 1, j, i - 1) * prim(IB1, k - 1, j, i - 1) +
              prim(IV1, k - 1, j, i - 1) * prim(IB3, k - 1, j, i - 1);
          const Real emfy_cell_k_im1 = -prim(IV3, k, j, i - 1) * prim(IB1, k, j, i - 1) +
                                       prim(IV1, k, j, i - 1) * prim(IB3, k, j, i - 1);
          // |------------------------------|

          // |----------- Step 3 -----------|
          const Real rho_ux_face_k_i = cons.flux(X1DIR, IDN, k, j, i);
          const Real rho_ux_face_km1_i = cons.flux(X1DIR, IDN, k - 1, j, i);

          const Real rho_uz_face_k_i = cons.flux(X3DIR, IDN, k, j, i);
          const Real rho_uz_face_k_im1 = cons.flux(X3DIR, IDN, k, j, i - 1);
          // NOTE! what the upwinded scheme really calls for is the sign of ux and uy
          // at the face, but since rho is (should be) positive, using
          // rho_ux and rho_uy should be fine
          // |------------------------------|

          // |----------- Step 4 -----------|
          // Store the edge electric field in the face variable's flux array.
          Real &emfy_corner =
              Bface.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);
          Real dEy_dz_xp_quarter = 0;
          Real dEy_dz_xp_three_quarter = 0;
          Real dEy_dx_zp_quarter = 0;
          Real dEy_dx_zp_three_quarter = 0;

          // upwinded corner emfs, following GS05
          if (rho_ux_face_km1_i > 0) {
            dEy_dz_xp_quarter = 2.0 * (emfz_face_k_im1 - emfy_cell_km1_im1) / dz;
          } else if (rho_ux_face_km1_i < 0) {
            dEy_dz_xp_quarter = 2.0 * (emfz_face_k_i - emfy_cell_km1_i) / dz;
          } else {
            dEy_dz_xp_quarter = 0.5 * (2.0 * (emfz_face_k_im1 - emfy_cell_km1_im1) / dz +
                                       2.0 * (emfz_face_k_i - emfy_cell_km1_i) / dz);
          }

          if (rho_ux_face_k_i > 0) {
            dEy_dz_xp_three_quarter = 2.0 * (emfy_cell_k_im1 - emfz_face_k_im1) / dz;
          } else if (rho_ux_face_k_i < 0) {
            dEy_dz_xp_three_quarter = 2.0 * (emfy_cell_k_i - emfz_face_k_i) / dz;
          } else {
            dEy_dz_xp_three_quarter =
                0.5 * (2.0 * (emfy_cell_k_im1 - emfz_face_k_im1) / dz +
                       2.0 * (emfy_cell_k_i - emfz_face_k_i) / dz);
          }

          if (rho_uz_face_k_im1 > 0) {
            dEy_dx_zp_quarter = 2.0 * (emfx_face_km1_i - emfy_cell_km1_im1) / dx;
          } else if (rho_uz_face_k_im1 < 0) {
            dEy_dx_zp_quarter = 2.0 * (emfx_face_k_i - emfy_cell_k_im1) / dx;
          } else {
            dEy_dx_zp_quarter = 0.5 * (2.0 * (emfx_face_km1_i - emfy_cell_km1_im1) / dx +
                                       2.0 * (emfx_face_k_i - emfy_cell_k_im1) / dx);
          }

          if (rho_uz_face_k_i > 0) {
            dEy_dx_zp_three_quarter = 2.0 * (emfy_cell_km1_i - emfx_face_km1_i) / dx;
          } else if (rho_uz_face_k_i < 0) {
            dEy_dx_zp_three_quarter = 2.0 * (emfy_cell_k_i - emfx_face_k_i) / dx;
          } else {
            dEy_dx_zp_three_quarter =
                0.5 * (2.0 * (emfy_cell_km1_i - emfx_face_km1_i) / dx +
                       2.0 * (emfy_cell_k_i - emfx_face_k_i) / dx);
          }

          // finally
          emfy_corner = (0.25 * (emfx_face_km1_i + emfx_face_k_i + emfz_face_k_im1 +
                                 emfz_face_k_i) +
                         dz / 8.0 * (dEy_dz_xp_quarter - dEy_dz_xp_three_quarter) +
                         dx / 8.0 * (dEy_dx_zp_quarter - dEy_dx_zp_three_quarter));
        });

    // for x- normal direction (Ex_edges)
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ex_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          auto &cons = cons_pack(b);
          auto &Bface = Bface_pack(b);
          const auto &prim = prim_pack(b);
          const auto &coords = prim_pack.GetCoords(b);
          const Real dy = coords.Dxc<2>(k, j, i);
          const Real dz = coords.Dxc<3>(k, j, i);

          // |----------- Step 1 -----------|
          // the edge EMFs will be stored in the flux of Bface, Artemis-style (mhd.cpp)
          //                                     y-flux, Bz-var
          const Real emfy_face_k_j = -cons.flux(X2DIR, IB3, k, j, i);
          const Real emfy_face_km1_j = -cons.flux(X2DIR, IB3, k - 1, j, i);
          //                                     z-flux, By-var
          const Real emfz_face_k_j = cons.flux(X3DIR, IB2, k, j, i);
          const Real emfz_face_k_jm1 = cons.flux(X3DIR, IB2, k, j - 1, i);
          //  |------------------------------|

          // |----------- Step 2 -----------|

          // emfy_cell_center = uzBy - vyBz
          const Real emfx_cell_k_j = prim(IV3, k, j, i) * prim(IB2, k, j, i) -
                                     prim(IV2, k, j, i) * prim(IB3, k, j, i);
          const Real emfx_cell_km1_j = prim(IV3, k - 1, j, i) * prim(IB2, k - 1, j, i) -
                                       prim(IV2, k - 1, j, i) * prim(IB3, k - 1, j, i);
          const Real emfx_cell_km1_jm1 =
              prim(IV3, k - 1, j - 1, i) * prim(IB2, k - 1, j - 1, i) -
              prim(IV2, k - 1, j - 1, i) * prim(IB3, k - 1, j - 1, i);
          const Real emfx_cell_k_jm1 = prim(IV3, k, j - 1, i) * prim(IB2, k, j - 1, i) -
                                       prim(IV2, k, j - 1, i) * prim(IB3, k, j - 1, i);
          // |------------------------------|

          // |----------- Step 3 -----------|
          const Real rho_uy_face_k_j = cons.flux(X2DIR, IDN, k, j, i);
          const Real rho_uy_face_km1_j = cons.flux(X2DIR, IDN, k - 1, j, i);

          const Real rho_uz_face_k_j = cons.flux(X3DIR, IDN, k, j, i);
          const Real rho_uz_face_k_jm1 = cons.flux(X3DIR, IDN, k, j - 1, i);
          // NOTE! what the upwinded scheme really calls for is the sign of ux and uy
          // at the face, but since rho is (should be) positive, using
          // rho_ux and rho_uy should be fine
          // |------------------------------|

          // |----------- Step 4 -----------|
          // Store the edge electric field in the face variable's flux array.
          Real &emfx_corner =
              Bface.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);
          Real dEx_dz_yp_quarter = 0;
          Real dEx_dz_yp_three_quarter = 0;
          Real dEx_dy_zp_quarter = 0;
          Real dEx_dy_zp_three_quarter = 0;

          // upwinded corner emfs, following GS05
          if (rho_uy_face_km1_j > 0) {
            dEx_dz_yp_quarter = 2.0 * (emfz_face_k_jm1 - emfx_cell_km1_jm1) / dz;
          } else if (rho_uy_face_km1_j < 0) {
            dEx_dz_yp_quarter = 2.0 * (emfz_face_k_j - emfx_cell_km1_j) / dz;
          } else {
            dEx_dz_yp_quarter = 0.5 * (2.0 * (emfz_face_k_jm1 - emfx_cell_km1_jm1) / dz +
                                       2.0 * (emfz_face_k_j - emfx_cell_km1_j) / dz);
          }

          if (rho_uy_face_k_j > 0) {
            dEx_dz_yp_three_quarter = 2.0 * (emfx_cell_k_jm1 - emfz_face_k_jm1) / dz;
          } else if (rho_uy_face_k_j < 0) {
            dEx_dz_yp_three_quarter = 2.0 * (emfx_cell_k_j - emfz_face_k_j) / dz;
          } else {
            dEx_dz_yp_three_quarter =
                0.5 * (2.0 * (emfx_cell_k_jm1 - emfz_face_k_jm1) / dz +
                       2.0 * (emfx_cell_k_j - emfz_face_k_j) / dz);
          }

          if (rho_uz_face_k_jm1 > 0) {
            dEx_dy_zp_quarter = 2.0 * (emfy_face_km1_j - emfx_cell_km1_jm1) / dy;
          } else if (rho_uz_face_k_jm1 < 0) {
            dEx_dy_zp_quarter = 2.0 * (emfy_face_k_j - emfx_cell_k_jm1) / dy;
          } else {
            dEx_dy_zp_quarter = 0.5 * (2.0 * (emfy_face_km1_j - emfx_cell_km1_jm1) / dy +
                                       2.0 * (emfy_face_k_j - emfx_cell_k_jm1) / dy);
          }

          if (rho_uz_face_k_j > 0) {
            dEx_dy_zp_three_quarter = 2.0 * (emfx_cell_km1_j - emfy_face_km1_j) / dy;
          } else if (rho_uz_face_k_j < 0) {
            dEx_dy_zp_three_quarter = 2.0 * (emfx_cell_k_j - emfy_face_k_j) / dy;
          } else {
            dEx_dy_zp_three_quarter =
                0.5 * (2.0 * (emfx_cell_km1_j - emfy_face_km1_j) / dy +
                       2.0 * (emfx_cell_k_j - emfy_face_k_j) / dy);
          }

          // finally
          emfx_corner = (0.25 * (emfy_face_km1_j + emfy_face_k_j + emfz_face_k_jm1 +
                                 emfz_face_k_j) +
                         dz / 8.0 * (dEx_dz_yp_quarter - dEx_dz_yp_three_quarter) +
                         dy / 8.0 * (dEx_dy_zp_quarter - dEx_dy_zp_three_quarter));
        });
  }

  return TaskStatus::complete;
}

TaskStatus UpdateWithFluxEmfCurl(MeshData<Real> *mu0, MeshData<Real> *mu1,
                                 const Real gam0, const Real gam1, const Real beta_dt) {
  auto pmb = mu0->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;

  // the edge EMFs will be stored in the fluxes of Bface
  auto B0_pack = mu0->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  auto B1_pack = mu1->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});

  IndexRange ib = mu0->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
  IndexRange jb = mu0->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
  IndexRange kb = mu0->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X1", parthenon::DevExecSpace(), 0,
      B0_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &coords = B0_pack.GetCoords(b);

        auto &B0 = B0_pack(b);
        auto &B1 = B1_pack(b);

        B0(TE::F1, 0, k, j, i) = 
          Hydro::CTMHD::StageB1FaceValue(
            B0, B1, coords, ndim, k, j, i,
            gam0, gam1, beta_dt);

      });
  if (ndim >= 2) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X2", parthenon::DevExecSpace(), 0,
        B0_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e + 1, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = B0_pack.GetCoords(b);

          auto &B0 = B0_pack(b);
          auto &B1 = B1_pack(b);

          B0(TE::F2, 0, k, j, i) = 
            Hydro::CTMHD::StageB2FaceValue(
              B0, B1, coords, ndim, k, j, i,
              gam0, gam1, beta_dt);

        });
  }
  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "ApplyFaceUpdate::X3", parthenon::DevExecSpace(), 0,
        B0_pack.GetDim(5) - 1, kb.s, kb.e + 1, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &coords = B0_pack.GetCoords(b);

          auto &B0 = B0_pack(b);
          auto &B1 = B1_pack(b);

          B0(TE::F3, 0, k, j, i) = 
            Hydro::CTMHD::StageB3FaceValue(
              B0, B1, coords, ndim, k, j, i,
              gam0, gam1, beta_dt);

        });
  }
  return TaskStatus::complete;
}

void center_Mag_Field(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  auto hydro_pkg = pmb->packages.Get("Hydro");
  const bool correct_ct_energy = hydro_pkg->Param<bool>("ct_energy_correction");
  const int ndim = pmb->pmy_mesh->ndim;

  auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
  // the edge EMFs will be stored in the fluxes of Bface
  auto Bface_pack = md->PackVariables(std::vector<std::string>{"Bface"});

  IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::entire);
  IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::entire);
  IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::entire);

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "center_Mag_field", parthenon::DevExecSpace(), 0,
      cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        auto &cons = cons_pack(b);
        const auto &Bface = Bface_pack(b);

        const Real b1_god = cons(IB1, k, j, i);
        const Real b2_god = cons(IB2, k, j, i);
        const Real b3_god = cons(IB3, k, j, i);

        const Real b1_ct =
            0.5 * (Bface(TE::F1, 0, k, j, i + 1) + Bface(TE::F1, 0, k, j, i));
        const Real b2_ct =
            (ndim >= 2)
                ? 0.5 * (Bface(TE::F2, 0, k, j + 1, i) + Bface(TE::F2, 0, k, j, i))
                : b2_god;
        const Real b3_ct =
            (ndim >= 3)
                ? 0.5 * (Bface(TE::F3, 0, k + 1, j, i) + Bface(TE::F3, 0, k, j, i))
                : b3_god;

        // Mignone/Del Zanna (2021) "Systematic Construction of Upwind CT schemes for MHD"
        // mention that this may be needed to avoid negative pressures (paragraph under
        // eq. 61), but I believe that it removes energy conservation, so it's implemented
        // as a user option for now...
        if (correct_ct_energy) {
          const Real emag_god = 0.5 * (SQR(b1_god) + SQR(b2_god) + SQR(b3_god));
          const Real emag_ct = 0.5 * (SQR(b1_ct) + SQR(b2_ct) + SQR(b3_ct));
          cons(IEN, k, j, i) += emag_ct - emag_god;
        }

        cons(IB1, k, j, i) = b1_ct;
        cons(IB2, k, j, i) = b2_ct;
        cons(IB3, k, j, i) = b3_ct;
      });
}

TaskStatus FirstOrderFluxCorrectCT(
    MeshData<Real> *mu0, MeshData<Real> *mu1,
    const Real gam0, const Real gam1, const Real beta_dt)
{
  // Diagnostic only: predict the source-free FV/CT stage without modifying
  // conserved variables, staggered fields, face fluxes or edge EMFs.
  auto pmb = mu0->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &mu0_cons_pack =
      mu0->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  const auto &mu1_cons_pack = mu1->PackVariables(std::vector<std::string>{"cons"});

  const auto &mu0_Bface_pack =
      mu0->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  const auto &mu1_Bface_pack =
      mu1->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});
  auto pkg = pmb->packages.Get("Hydro");

  const auto &trial_pack =
      mu0->PackVariables(std::vector<std::string>{"ct_fofc_trial"});

  const bool correct_ct_energy = pkg->Param<bool>("ct_energy_correction");

  const int ndim = pmb->pmy_mesh->ndim;

  constexpr int NVAR = IB3 + 1; // Eight physical CT-MHD components, no GLM psi.
  std::int64_t num_bad = 0;

    Kokkos::parallel_reduce(
        "DiagnoseCTTrialState",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {mu0_cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      std::int64_t &lnum_bad) {
          const auto &coords = mu0_cons_pack.GetCoords(b);
          const auto &mu0_cons = mu0_cons_pack(b);
          Real new_cons[NVAR];
          for (auto v = 0; v < NVAR; v++) {
            new_cons[v] =
                gam0 * mu0_cons(v, k, j, i) + gam1 * mu1_cons_pack(b, v, k, j, i) +
                beta_dt *
                    parthenon::Update::FluxDivHelper(v, k, j, i, ndim, coords, mu0_cons);
          }

          // Update magnetic field and put it in new_cons[IB1:IB3]
          Hydro::CTMHD::PredictCTStageConserved(
            mu0_Bface_pack(b), mu1_Bface_pack(b),
            coords, ndim,
            k, j, i,
            gam0, gam1,
            beta_dt,
            correct_ct_energy,
            new_cons);

          for (int v = 0; v < NVAR; ++v){
            trial_pack(b, v, k, j, i) = new_cons[v];
          }

          // helper to check dens or thermal energy
          // is negative or nan
          if (!IsCTStateAdmissible(new_cons)) {
            ++lnum_bad;
          }
          

        },
        Kokkos::Sum<std::int64_t>(num_bad));

  const std::int64_t num_cells =
      static_cast<std::int64_t>(mu0_cons_pack.GetDim(5)) * (kb.e - kb.s + 1) *
      (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  std::cout << "[CT trial diagnostic] rank=" << parthenon::Globals::my_rank
            << " local_pack_cells=" << num_cells << " invalid_trial_cells=" << num_bad
            << std::endl;

  return TaskStatus::complete;

}

// diagnostic only -- checking if predicted stage update
// is actually equal to the real stage update value
TaskStatus CompareCTTrialState(MeshData<Real> *mu0, const int stage)
{
  auto pmb = mu0->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  const auto &mu0_cons_pack = mu0->PackVariables(std::vector<std::string>{"cons"});
  const auto &mu0_trial_cons_pack = mu0->PackVariables(std::vector<std::string>{"ct_fofc_trial"});


  constexpr int NVAR = IB3 + 1; // Eight physical CT-MHD components, no GLM psi.
  Real max_abs_error;
  Real max_scaled_error;
  std::int64_t num_nonfinite = 0;

    Kokkos::parallel_reduce(
        "CompareCTTrialState",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {mu0_cons_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      Real &lmax_abs_error, Real &lmax_scaled_error, std::int64_t &lnum_nonfinite) {
          const auto &cons = mu0_cons_pack(b);
          const auto &trial_cons = mu0_trial_cons_pack(b);
          for (int v = 0; v < NVAR; ++v){
            const Real predicted = trial_cons(v, k, j, i);
            const Real actual = cons(v, k, j, i);

            if (!Kokkos::isfinite(predicted) || !Kokkos::isfinite(actual)) {
              ++lnum_nonfinite;
              continue;
            }

            const Real d = Kokkos::fabs(predicted - actual);

            if (!Kokkos::isfinite(d)) {
              ++lnum_nonfinite;
              continue;
            }

            const Real s = Kokkos::fmax(
                1.0, Kokkos::fmax(Kokkos::fabs(predicted), Kokkos::fabs(actual)));
            const Real r = d / s;

            lmax_abs_error = Kokkos::fmax(lmax_abs_error, d);
            lmax_scaled_error = Kokkos::fmax(lmax_scaled_error, r);
          }
        },
        Kokkos::Max<Real>(max_abs_error),
        Kokkos::Max<Real>(max_scaled_error),
        Kokkos::Sum<std::int64_t>(num_nonfinite));

  const std::int64_t num_cells =
      static_cast<std::int64_t>(mu0_cons_pack.GetDim(5)) * (kb.e - kb.s + 1) *
      (jb.e - jb.s + 1) * (ib.e - ib.s + 1);
  const bool agrees =
      num_nonfinite == 0 && max_scaled_error <= 1.0e-12;

  std::ostringstream message;
  message << std::scientific << std::setprecision(16)
          << "[CT trial comparison] rank=" << parthenon::Globals::my_rank
          << " stage=" << stage
          << " local_pack_cells=" << num_cells
          << " max_abs_error=" << max_abs_error
          << " max_scaled_error=" << max_scaled_error
          << " nonfinite_comparisons=" << num_nonfinite
          << " status=" << (agrees ? "PASS" : "FAIL");

  std::cout << message.str() << std::endl;

  return TaskStatus::complete;

}

TaskStatus TestCTStateAdmissibility() {
  // Synthetic local states only: no mesh data or simulation fields are touched.
  constexpr int NVAR = IB3 + 1;
  constexpr int NUM_CASES = 28;
  const Real nan_value = std::numeric_limits<Real>::quiet_NaN();
  const Real inf_value = std::numeric_limits<Real>::infinity();
  const Real max_value = std::numeric_limits<Real>::max();
  std::int64_t num_mismatches = 0;

  Kokkos::parallel_reduce(
      "TestCTStateAdmissibility",
      Kokkos::RangePolicy<parthenon::DevExecSpace>(0, NUM_CASES),
      KOKKOS_LAMBDA(const int case_id, std::int64_t &local_mismatches) {
        // Valid reference: rho=1, momentum=0, E=3, B=(1,0,0), e_th=2.5.
        Real state[NVAR] = {};
        state[IDN] = 1.0;
        state[IEN] = 3.0;
        state[IB1] = 1.0;

        if (case_id == 1) {
          state[IDN] = 0.0;
        } else if (case_id == 2) {
          state[IDN] = -1.0;
        } else if (case_id == 3) {
          state[IEN] = 0.5; // Exactly zero thermal energy.
        } else if (case_id == 4) {
          state[IEN] = 0.25; // Negative thermal energy despite positive total E.
        } else if (case_id >= 5 && case_id <= 7) {
          state[IM1 + case_id - 5] = 3.0; // Excess kinetic energy, each component.
        } else if (case_id >= 8 && case_id <= 10) {
          state[IB1 + case_id - 8] = 3.0; // Excess magnetic energy, each component.
        } else if (case_id >= 11 && case_id <= 18) {
          state[case_id - 11] = nan_value; // NaN in each conserved component.
        } else if (case_id >= 19 && case_id <= 26) {
          state[case_id - 19] = inf_value; // Infinity in each component.
        } else if (case_id == 27) {
          state[IM1] = max_value; // Finite input whose kinetic energy overflows.
        }

        const bool expected = (case_id == 0);
        const bool actual = IsCTStateAdmissible(state);
        if (actual != expected) {
          ++local_mismatches;
        }
      },
      Kokkos::Sum<std::int64_t>(num_mismatches));

  std::cout << "[CT admissibility self-test] rank=" << parthenon::Globals::my_rank
            << " cases=" << NUM_CASES << " mismatches=" << num_mismatches
            << " status=" << (num_mismatches == 0 ? "PASS" : "FAIL") << std::endl;
  PARTHENON_REQUIRE(num_mismatches == 0,
                    "CT admissibility self-test classified a synthetic state incorrectly.");

  return TaskStatus::complete;
}

} // namespace Hydro::CTMHD
