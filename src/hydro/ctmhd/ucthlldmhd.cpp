//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "ucthlldmhd.hpp"
#include "../../recon/plm_simple.hpp"
#include "../../recon/weno3_simple.hpp"
#include "../../recon/wenoz_point.hpp"


using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;


namespace Hydro::UCTHLLDMHD {
template <Reconstruction recon>
KOKKOS_INLINE_FUNCTION void ReconstructAverageToEdge(
    const Real &q_m1, const Real &q_0, const Real &q_p1, Real &ql, Real &qr,
    const Real dx2) {
  static_assert(recon == Reconstruction::plm || recon == Reconstruction::weno3,
                "Only PLM and WENO3 reconstruct face averages to UCT edges.");
  if constexpr (recon == Reconstruction::weno3) {
    WENO3(q_m1, q_0, q_p1, ql, qr, dx2);
  } else {
    PLM(q_m1, q_0, q_p1, ql, qr);
  }
}

template <bool pointwise, Reconstruction edge_recon = Reconstruction::plm>
TaskStatus Assemble_HLLD_Edge_EMF_Impl(MeshData<Real> *md) {
    static_assert(pointwise || edge_recon == Reconstruction::plm ||
                      edge_recon == Reconstruction::weno3,
                  "Unsupported reconstruction for averaged UCT edge states.");
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;

    auto cons_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
    // the edge EMFs will be stored in the fluxes of Bface
    auto B_pack = md->PackVariablesAndFluxes(
        std::vector<std::string>{pointwise ? "Bface_point" : "Bface"});


    const auto &uct_hlld_pack = md->PackVariables(std::vector<std::string>{"uct_hlld"});
    

    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);


    // Pointwise edge EMFs need one extra point along the edge direction for
    // the subsequent point-to-line conversion. The transverse ranges remain
    // the ordinary edge ranges.
    const int ez_kl = (pointwise && ndim > 2) ? kb.s - 1 : kb.s;
    const int ez_ku = (pointwise && ndim > 2) ? kb.e + 1 : kb.e;
    const int ey_jl = pointwise ? jb.s - 1 : jb.s;
    const int ey_ju = pointwise ? jb.e + 1 : jb.e;
    const int ex_il = pointwise ? ib.s - 1 : ib.s;
    const int ex_iu = pointwise ? ib.e + 1 : ib.e;
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ez_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, ez_kl, ez_ku, jb.s, jb.e+1, ib.s, ib.e+1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            auto &Bface = B_pack(b);
            const auto &uct_hlld = uct_hlld_pack(b);


            // |----------- Step 1 -----------|
            // compute the N-S-E-W flux and diffusion coefficients
            // eqs. (34) and (35) in [MDZ21]
            const Real axW = 0.5 * (uct_hlld(TE::F1, AL, k, j-1, i) + uct_hlld(TE::F1, AL, k, j, i));
            const Real axE = 0.5 * (uct_hlld(TE::F1, AR, k, j-1, i) + uct_hlld(TE::F1, AR, k, j, i));
            const Real ayS = 0.5 * (uct_hlld(TE::F2, AL, k, j, i-1) + uct_hlld(TE::F2, AL, k, j, i));
            const Real ayN = 0.5 * (uct_hlld(TE::F2, AR, k, j, i-1) + uct_hlld(TE::F2, AR, k, j, i));

            const Real dxW = 0.5 * (uct_hlld(TE::F1, DL, k, j-1, i) + uct_hlld(TE::F1, DL, k, j, i));
            const Real dxE = 0.5 * (uct_hlld(TE::F1, DR, k, j-1, i) + uct_hlld(TE::F1, DR, k, j, i));
            const Real dyS = 0.5 * (uct_hlld(TE::F2, DL, k, j, i-1) + uct_hlld(TE::F2, DL, k, j, i));
            const Real dyN = 0.5 * (uct_hlld(TE::F2, DR, k, j, i-1) + uct_hlld(TE::F2, DR, k, j, i));

            // |----------- Step 2 -----------|
            // reconstruct transverse velocity and 
            // magnetic fields to the edge
            // ByW = reconstructed to corner in +x dir
            // ByE = reconstructed to corner in -x dir
            // BxS = reconstructed to corner in +y dir
            // BxN = reconstructed to corner in -y dir

            Real ByW, ByE, unused;
            Real BxS, BxN;
            Real vxW, vxE;
            Real vyS, vyN;
            if constexpr (pointwise) {
                WENOZ_POINT(
                    Bface(TE::F2, 0, k, j, i-3),
                    Bface(TE::F2, 0, k, j, i-2),
                    Bface(TE::F2, 0, k, j, i-1),
                    Bface(TE::F2, 0, k, j, i  ),
                    Bface(TE::F2, 0, k, j, i+1),
                    ByW, unused);

                WENOZ_POINT(
                    Bface(TE::F2, 0, k, j, i-2),
                    Bface(TE::F2, 0, k, j, i-1),
                    Bface(TE::F2, 0, k, j, i),
                    Bface(TE::F2, 0, k, j, i+1),
                    Bface(TE::F2, 0, k, j, i+2),
                    unused, ByE);

                WENOZ_POINT(
                    Bface(TE::F1, 0, k, j-3, i),
                    Bface(TE::F1, 0, k, j-2, i),
                    Bface(TE::F1, 0, k, j-1, i),
                    Bface(TE::F1, 0, k, j, i  ),
                    Bface(TE::F1, 0, k, j+1, i),
                    BxS, unused);

                WENOZ_POINT(
                    Bface(TE::F1, 0, k, j-2, i),
                    Bface(TE::F1, 0, k, j-1, i),
                    Bface(TE::F1, 0, k, j, i),
                    Bface(TE::F1, 0, k, j+1, i),
                    Bface(TE::F1, 0, k, j+2, i),
                    unused, BxN);

                // reconstruct velocities
                WENOZ_POINT(
                    uct_hlld(TE::F2, VBART2, k, j, i-3),
                    uct_hlld(TE::F2, VBART2, k, j, i-2),
                    uct_hlld(TE::F2, VBART2, k, j, i-1),
                    uct_hlld(TE::F2, VBART2, k, j, i  ),
                    uct_hlld(TE::F2, VBART2, k, j, i+1),
                    vxW, unused);

                WENOZ_POINT(
                    uct_hlld(TE::F2, VBART2, k, j, i-2),
                    uct_hlld(TE::F2, VBART2, k, j, i-1),
                    uct_hlld(TE::F2, VBART2, k, j, i),
                    uct_hlld(TE::F2, VBART2, k, j, i+1),
                    uct_hlld(TE::F2, VBART2, k, j, i+2),
                    unused, vxE);

                WENOZ_POINT(
                    uct_hlld(TE::F1, VBART1, k, j-3, i),
                    uct_hlld(TE::F1, VBART1, k, j-2, i),
                    uct_hlld(TE::F1, VBART1, k, j-1, i),
                    uct_hlld(TE::F1, VBART1, k, j, i  ),
                    uct_hlld(TE::F1, VBART1, k, j+1, i),
                    vyS, unused);

                WENOZ_POINT(
                    uct_hlld(TE::F1, VBART1, k, j-2, i),
                    uct_hlld(TE::F1, VBART1, k, j-1, i),
                    uct_hlld(TE::F1, VBART1, k, j, i),
                    uct_hlld(TE::F1, VBART1, k, j+1, i),
                    uct_hlld(TE::F1, VBART1, k, j+2, i),
                    unused, vyN);
            } else {
                const auto &coords = B_pack.GetCoords(b);
                const Real dx1 = coords.Dxc<X1DIR>(k, j, i);
                const Real dx2 = coords.Dxc<X2DIR>(k, j, i);

                ReconstructAverageToEdge<edge_recon>(
                    Bface(TE::F2, 0, k, j, i-2),
                    Bface(TE::F2, 0, k, j, i-1),
                    Bface(TE::F2, 0, k, j, i  ),
                    ByW, unused, dx1 * dx1);

                ReconstructAverageToEdge<edge_recon>(
                    Bface(TE::F2, 0, k, j, i-1),
                    Bface(TE::F2, 0, k, j, i),
                    Bface(TE::F2, 0, k, j, i+1),
                    unused, ByE, dx1 * dx1);

                ReconstructAverageToEdge<edge_recon>(
                    Bface(TE::F1, 0, k, j-2, i),
                    Bface(TE::F1, 0, k, j-1, i),
                    Bface(TE::F1, 0, k, j, i  ),
                    BxS, unused, dx2 * dx2);

                ReconstructAverageToEdge<edge_recon>(
                    Bface(TE::F1, 0, k, j-1, i),
                    Bface(TE::F1, 0, k, j, i),
                    Bface(TE::F1, 0, k, j+1, i),
                    unused, BxN, dx2 * dx2);

                // reconstruct velocities
                ReconstructAverageToEdge<edge_recon>(
                    uct_hlld(TE::F2, VBART2, k, j, i-2),
                    uct_hlld(TE::F2, VBART2, k, j, i-1),
                    uct_hlld(TE::F2, VBART2, k, j, i  ),
                    vxW, unused, dx1 * dx1);

                ReconstructAverageToEdge<edge_recon>(
                    uct_hlld(TE::F2, VBART2, k, j, i-1),
                    uct_hlld(TE::F2, VBART2, k, j, i),
                    uct_hlld(TE::F2, VBART2, k, j, i+1),
                    unused, vxE, dx1 * dx1);

                ReconstructAverageToEdge<edge_recon>(
                    uct_hlld(TE::F1, VBART1, k, j-2, i),
                    uct_hlld(TE::F1, VBART1, k, j-1, i),
                    uct_hlld(TE::F1, VBART1, k, j, i  ),
                    vyS, unused, dx2 * dx2);

                ReconstructAverageToEdge<edge_recon>(
                    uct_hlld(TE::F1, VBART1, k, j-1, i),
                    uct_hlld(TE::F1, VBART1, k, j, i),
                    uct_hlld(TE::F1, VBART1, k, j+1, i),
                    unused, vyN, dx2 * dx2);
            }


            // |----------- Step 3 -----------|
            // build corner EMF
            Real &emfz_corner =
                Bface.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);    
            
            // finally
            emfz_corner = (
                -((axW * vxW * ByW) + (axE * vxE * ByE)) +
                 ((ayN * vyN * BxN) + (ayS * vyS * BxS)) +
                 ((dxE * ByE) - (dxW * ByW)) -
                 ((dyN * BxN) - (dyS * BxS))
            );
        });

    if (ndim > 2) {
        // for y-directed edges (Ey_edges)
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "Assemble Ey_edges", parthenon::DevExecSpace(), 0,
            cons_pack.GetDim(5) - 1, kb.s, kb.e+1, ey_jl, ey_ju, ib.s, ib.e+1,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                auto &Bface = B_pack(b);
                const auto &uct_hlld = uct_hlld_pack(b);


                // |----------- Step 1 -----------|
                // compute the N-S-E-W flux and diffusion coefficients
                // eqs. (34) and (35) in [MDZ21]
                const Real azW = 0.5 * (uct_hlld(TE::F3, AL, k, j, i-1) + uct_hlld(TE::F3, AL, k, j, i));
                const Real azE = 0.5 * (uct_hlld(TE::F3, AR, k, j, i-1) + uct_hlld(TE::F3, AR, k, j, i));
                const Real axS = 0.5 * (uct_hlld(TE::F1, AL, k-1, j, i) + uct_hlld(TE::F1, AL, k, j, i));
                const Real axN = 0.5 * (uct_hlld(TE::F1, AR, k-1, j, i) + uct_hlld(TE::F1, AR, k, j, i));

                const Real dzW = 0.5 * (uct_hlld(TE::F3, DL, k, j, i-1) + uct_hlld(TE::F3, DL, k, j, i));
                const Real dzE = 0.5 * (uct_hlld(TE::F3, DR, k, j, i-1) + uct_hlld(TE::F3, DR, k, j, i));
                const Real dxS = 0.5 * (uct_hlld(TE::F1, DL, k-1, j, i) + uct_hlld(TE::F1, DL, k, j, i));
                const Real dxN = 0.5 * (uct_hlld(TE::F1, DR, k-1, j, i) + uct_hlld(TE::F1, DR, k, j, i));

                // |----------- Step 2 -----------|
                // reconstruct transverse velocity and 
                // magnetic fields to the edge

                Real BxW, BxE, unused;
                Real BzS, BzN;
                Real vzW, vzE;
                Real vxS, vxN;
                if constexpr (pointwise) {
                    WENOZ_POINT(
                        Bface(TE::F1, 0, k-3, j, i),
                        Bface(TE::F1, 0, k-2, j, i),
                        Bface(TE::F1, 0, k-1, j, i),
                        Bface(TE::F1, 0, k, j, i),
                        Bface(TE::F1, 0, k+1, j, i),
                        BxW, unused);

                    WENOZ_POINT(
                        Bface(TE::F1, 0, k-2, j, i),
                        Bface(TE::F1, 0, k-1, j, i),
                        Bface(TE::F1, 0, k, j, i),
                        Bface(TE::F1, 0, k+1, j, i),
                        Bface(TE::F1, 0, k+2, j, i),
                        unused, BxE);

                    WENOZ_POINT(
                        Bface(TE::F3, 0, k, j, i-3),
                        Bface(TE::F3, 0, k, j, i-2),
                        Bface(TE::F3, 0, k, j, i-1),
                        Bface(TE::F3, 0, k, j, i),
                        Bface(TE::F3, 0, k, j, i+1),
                        BzS, unused);

                    WENOZ_POINT(
                        Bface(TE::F3, 0, k, j, i-2),
                        Bface(TE::F3, 0, k, j, i-1),
                        Bface(TE::F3, 0, k, j, i),
                        Bface(TE::F3, 0, k, j, i+1),
                        Bface(TE::F3, 0, k, j, i+2),
                        unused, BzN);

                    // reconstruct velocities
                    WENOZ_POINT(
                        uct_hlld(TE::F1, VBART2, k-3, j, i),
                        uct_hlld(TE::F1, VBART2, k-2, j, i),
                        uct_hlld(TE::F1, VBART2, k-1, j, i),
                        uct_hlld(TE::F1, VBART2, k, j, i),
                        uct_hlld(TE::F1, VBART2, k+1, j, i),
                        vzW, unused);

                    WENOZ_POINT(
                        uct_hlld(TE::F1, VBART2, k-2, j, i),
                        uct_hlld(TE::F1, VBART2, k-1, j, i),
                        uct_hlld(TE::F1, VBART2, k, j, i),
                        uct_hlld(TE::F1, VBART2, k+1, j, i),
                        uct_hlld(TE::F1, VBART2, k+2, j, i),
                        unused, vzE);

                    WENOZ_POINT(
                        uct_hlld(TE::F3, VBART1, k, j, i-3),
                        uct_hlld(TE::F3, VBART1, k, j, i-2),
                        uct_hlld(TE::F3, VBART1, k, j, i-1),
                        uct_hlld(TE::F3, VBART1, k, j, i),
                        uct_hlld(TE::F3, VBART1, k, j, i+1),
                        vxS, unused);

                    WENOZ_POINT(
                        uct_hlld(TE::F3, VBART1, k, j, i-2),
                        uct_hlld(TE::F3, VBART1, k, j, i-1),
                        uct_hlld(TE::F3, VBART1, k, j, i),
                        uct_hlld(TE::F3, VBART1, k, j, i+1),
                        uct_hlld(TE::F3, VBART1, k, j, i+2),
                        unused, vxN);
                } else {
                    const auto &coords = B_pack.GetCoords(b);
                    const Real dx1 = coords.Dxc<X1DIR>(k, j, i);
                    const Real dx3 = coords.Dxc<X3DIR>(k, j, i);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F1, 0, k-2, j, i),
                        Bface(TE::F1, 0, k-1, j, i),
                        Bface(TE::F1, 0, k, j, i),
                        BxW, unused, dx3 * dx3);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F1, 0, k-1, j, i),
                        Bface(TE::F1, 0, k, j, i),
                        Bface(TE::F1, 0, k+1, j, i),
                        unused, BxE, dx3 * dx3);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F3, 0, k, j, i-2),
                        Bface(TE::F3, 0, k, j, i-1),
                        Bface(TE::F3, 0, k, j, i),
                        BzS, unused, dx1 * dx1);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F3, 0, k, j, i-1),
                        Bface(TE::F3, 0, k, j, i),
                        Bface(TE::F3, 0, k, j, i+1),
                        unused, BzN, dx1 * dx1);

                    // reconstruct velocities
                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F1, VBART2, k-2, j, i),
                        uct_hlld(TE::F1, VBART2, k-1, j, i),
                        uct_hlld(TE::F1, VBART2, k, j, i),
                        vzW, unused, dx3 * dx3);

                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F1, VBART2, k-1, j, i),
                        uct_hlld(TE::F1, VBART2, k, j, i),
                        uct_hlld(TE::F1, VBART2, k+1, j, i),
                        unused, vzE, dx3 * dx3);

                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F3, VBART1, k, j, i-2),
                        uct_hlld(TE::F3, VBART1, k, j, i-1),
                        uct_hlld(TE::F3, VBART1, k, j, i),
                        vxS, unused, dx1 * dx1);

                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F3, VBART1, k, j, i-1),
                        uct_hlld(TE::F3, VBART1, k, j, i),
                        uct_hlld(TE::F3, VBART1, k, j, i+1),
                        unused, vxN, dx1 * dx1);
                }


                // |----------- Step 3 -----------|
                // build corner EMF
                Real &emfy_corner =
                    Bface.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);    
                
                // finally
                emfy_corner = (
                    -((azW * vzW * BxW) + (azE * vzE * BxE)) +
                    ((axN * vxN * BzN) + (axS * vxS * BzS)) +
                    ((dzE * BxE) - (dzW * BxW)) -
                    ((dxN * BzN) - (dxS * BzS))
                );
            });
    
        // for x- directed edges (Ex_edges)
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "Assemble Ex_edges", parthenon::DevExecSpace(), 0,
            cons_pack.GetDim(5) - 1, kb.s, kb.e+1, jb.s, jb.e+1, ex_il, ex_iu,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                auto &Bface = B_pack(b);
                const auto &uct_hlld = uct_hlld_pack(b);


                // |----------- Step 1 -----------|
                // compute the N-S-E-W flux and diffusion coefficients
                // eqs. (34) and (35) in [MDZ21]
                const Real ayW = 0.5 * (uct_hlld(TE::F2, AL, k-1, j, i) + uct_hlld(TE::F2, AL, k, j, i));
                const Real ayE = 0.5 * (uct_hlld(TE::F2, AR, k-1, j, i) + uct_hlld(TE::F2, AR, k, j, i));
                const Real azS = 0.5 * (uct_hlld(TE::F3, AL, k, j-1, i) + uct_hlld(TE::F3, AL, k, j, i));
                const Real azN = 0.5 * (uct_hlld(TE::F3, AR, k, j-1, i) + uct_hlld(TE::F3, AR, k, j, i));

                const Real dyW = 0.5 * (uct_hlld(TE::F2, DL, k-1, j, i) + uct_hlld(TE::F2, DL, k, j, i));
                const Real dyE = 0.5 * (uct_hlld(TE::F2, DR, k-1, j, i) + uct_hlld(TE::F2, DR, k, j, i));
                const Real dzS = 0.5 * (uct_hlld(TE::F3, DL, k, j-1, i) + uct_hlld(TE::F3, DL, k, j, i));
                const Real dzN = 0.5 * (uct_hlld(TE::F3, DR, k, j-1, i) + uct_hlld(TE::F3, DR, k, j, i));

                // |----------- Step 2 -----------|
                // reconstruct transverse velocity and 
                // magnetic fields to the edge

                Real BzW, BzE, unused;
                Real ByS, ByN;
                Real vyW, vyE;
                Real vzS, vzN;
                if constexpr (pointwise) {
                    WENOZ_POINT(
                        Bface(TE::F3, 0, k, j-3, i),
                        Bface(TE::F3, 0, k, j-2, i),
                        Bface(TE::F3, 0, k, j-1, i),
                        Bface(TE::F3, 0, k, j, i),
                        Bface(TE::F3, 0, k, j+1, i),
                        BzW, unused);

                    WENOZ_POINT(
                        Bface(TE::F3, 0, k, j-2, i),
                        Bface(TE::F3, 0, k, j-1, i),
                        Bface(TE::F3, 0, k, j, i),
                        Bface(TE::F3, 0, k, j+1, i),
                        Bface(TE::F3, 0, k, j+2, i),
                        unused, BzE);

                    WENOZ_POINT(
                        Bface(TE::F2, 0, k-3, j, i),
                        Bface(TE::F2, 0, k-2, j, i),
                        Bface(TE::F2, 0, k-1, j, i),
                        Bface(TE::F2, 0, k, j, i),
                        Bface(TE::F2, 0, k+1, j, i),
                        ByS, unused);

                    WENOZ_POINT(
                        Bface(TE::F2, 0, k-2, j, i),
                        Bface(TE::F2, 0, k-1, j, i),
                        Bface(TE::F2, 0, k, j, i),
                        Bface(TE::F2, 0, k+1, j, i),
                        Bface(TE::F2, 0, k+2, j, i),
                        unused, ByN);

                    // reconstruct velocities
                    WENOZ_POINT(
                        uct_hlld(TE::F3, VBART2, k, j-3, i),
                        uct_hlld(TE::F3, VBART2, k, j-2, i),
                        uct_hlld(TE::F3, VBART2, k, j-1, i),
                        uct_hlld(TE::F3, VBART2, k, j, i),
                        uct_hlld(TE::F3, VBART2, k, j+1, i),
                        vyW, unused);

                    WENOZ_POINT(
                        uct_hlld(TE::F3, VBART2, k, j-2, i),
                        uct_hlld(TE::F3, VBART2, k, j-1, i),
                        uct_hlld(TE::F3, VBART2, k, j, i),
                        uct_hlld(TE::F3, VBART2, k, j+1, i),
                        uct_hlld(TE::F3, VBART2, k, j+2, i),
                        unused, vyE);

                    WENOZ_POINT(
                        uct_hlld(TE::F2, VBART1, k-3, j, i),
                        uct_hlld(TE::F2, VBART1, k-2, j, i),
                        uct_hlld(TE::F2, VBART1, k-1, j, i),
                        uct_hlld(TE::F2, VBART1, k, j, i),
                        uct_hlld(TE::F2, VBART1, k+1, j, i),
                        vzS, unused);

                    WENOZ_POINT(
                        uct_hlld(TE::F2, VBART1, k-2, j, i),
                        uct_hlld(TE::F2, VBART1, k-1, j, i),
                        uct_hlld(TE::F2, VBART1, k, j, i),
                        uct_hlld(TE::F2, VBART1, k+1, j, i),
                        uct_hlld(TE::F2, VBART1, k+2, j, i),
                        unused, vzN);
                } else {
                    const auto &coords = B_pack.GetCoords(b);
                    const Real dx2 = coords.Dxc<X2DIR>(k, j, i);
                    const Real dx3 = coords.Dxc<X3DIR>(k, j, i);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F3, 0, k, j-2, i),
                        Bface(TE::F3, 0, k, j-1, i),
                        Bface(TE::F3, 0, k, j, i),
                        BzW, unused, dx2 * dx2);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F3, 0, k, j-1, i),
                        Bface(TE::F3, 0, k, j, i),
                        Bface(TE::F3, 0, k, j+1, i),
                        unused, BzE, dx2 * dx2);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F2, 0, k-2, j, i),
                        Bface(TE::F2, 0, k-1, j, i),
                        Bface(TE::F2, 0, k, j, i),
                        ByS, unused, dx3 * dx3);

                    ReconstructAverageToEdge<edge_recon>(
                        Bface(TE::F2, 0, k-1, j, i),
                        Bface(TE::F2, 0, k, j, i),
                        Bface(TE::F2, 0, k+1, j, i),
                        unused, ByN, dx3 * dx3);

                    // reconstruct velocities
                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F3, VBART2, k, j-2, i),
                        uct_hlld(TE::F3, VBART2, k, j-1, i),
                        uct_hlld(TE::F3, VBART2, k, j, i),
                        vyW, unused, dx2 * dx2);

                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F3, VBART2, k, j-1, i),
                        uct_hlld(TE::F3, VBART2, k, j, i),
                        uct_hlld(TE::F3, VBART2, k, j+1, i),
                        unused, vyE, dx2 * dx2);

                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F2, VBART1, k-2, j, i),
                        uct_hlld(TE::F2, VBART1, k-1, j, i),
                        uct_hlld(TE::F2, VBART1, k, j, i),
                        vzS, unused, dx3 * dx3);

                    ReconstructAverageToEdge<edge_recon>(
                        uct_hlld(TE::F2, VBART1, k-1, j, i),
                        uct_hlld(TE::F2, VBART1, k, j, i),
                        uct_hlld(TE::F2, VBART1, k+1, j, i),
                        unused, vzN, dx3 * dx3);
                }


                // |----------- Step 3 -----------|
                // build corner EMF
                Real &emfx_corner =
                    Bface.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);    
                
                // finally
                emfx_corner = (
                    -((ayW * vyW * BzW) + (ayE * vyE * BzE)) +
                    ((azN * vzN * ByN) + (azS * vzS * ByS)) +
                    ((dyE * BzE) - (dyW * BzW)) -
                    ((dzN * ByN) - (dzS * ByS))
                );
            });
    }


   
    return TaskStatus::complete;
}

TaskStatus Assemble_HLLD_Edge_EMF(MeshData<Real> *md) {
  return Assemble_HLLD_Edge_EMF_Impl<false, Reconstruction::plm>(md);
}

TaskStatus Assemble_HLLD_WENO3_Edge_EMF(MeshData<Real> *md) {
  return Assemble_HLLD_Edge_EMF_Impl<false, Reconstruction::weno3>(md);
}

TaskStatus Assemble_HLLD_Point_Edge_EMF(MeshData<Real> *md) {
  return Assemble_HLLD_Edge_EMF_Impl<true, Reconstruction::plm>(md);
}


} // namespace Hydro::UCTHLLDMHD
