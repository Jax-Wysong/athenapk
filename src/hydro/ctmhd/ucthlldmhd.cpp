//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../eos/adiabatic_ctmhd.hpp"
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

Real MaxShockIndicatorHst(MeshData<Real> *md) {
    const auto indicator_pack =
        md->PackVariables(std::vector<std::string>{"berta24_shock_indicator"});

    const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    Real max_indicator = 0.0;
    Kokkos::parallel_reduce(
        "MaxShockIndicatorHst",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            parthenon::DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {indicator_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      Real &local_max) {
            const auto &indicator = indicator_pack(b);
            local_max = Kokkos::fmax(local_max, indicator(0, k, j, i));
        },
        Kokkos::Max<Real>(max_indicator));

    return max_indicator;
}

Real CountTroubledHst(MeshData<Real> *md) {
    const auto troubled_pack =
        md->PackVariables(std::vector<std::string>{"berta24_troubled"});

    const IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    const IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    const IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    Real num_troubled = 0.0;
    Kokkos::parallel_reduce(
        "CountTroubledHst",
        Kokkos::MDRangePolicy<Kokkos::Rank<4>>(
            parthenon::DevExecSpace(), {0, kb.s, jb.s, ib.s},
            {troubled_pack.GetDim(5), kb.e + 1, jb.e + 1, ib.e + 1},
            {1, 1, 1, ib.e + 1 - ib.s}),
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i,
                      Real &local_sum) {
            const auto &troubled = troubled_pack(b);
            local_sum += troubled(0, k, j, i) > 0.5 ? 1.0 : 0.0;
        },
        num_troubled);

    return num_troubled;
}

TaskStatus CalculateJamesonShockDetector(MeshData<Real> *md){
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;

    auto pkg = pmb->packages.Get("Hydro");

    const auto &dd_eps = pkg->Param<Real>("hydro/discontinuity_detector_epsilon");
    const auto &dd_threshold = pkg->Param<Real>("hydro/discontinuity_detector_threshold");

    // prim should hold the the second-order conversion of the
    // averaged conserved state at this time, which means
    // it already has cell-centered B available from the
    // face-averaging procedure
    const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
    auto indicator_pack = md->PackVariables(std::vector<std::string>{"berta24_shock_indicator"});
    auto troubled_pack  = md->PackVariables(std::vector<std::string>{"berta24_troubled"});
    
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    int kl, ku, jl, ju, il, iu;
    if (ndim > 2){
        kl = kb.s - 1;
        ku = kb.e + 1;
        jl = jb.s - 1;
        ju = jb.e + 1;
        il = ib.s - 1;
        iu = ib.e + 1;
    } else if (ndim > 1){
        kl = kb.s;
        ku = kb.e;
        jl = jb.s-1;
        ju = jb.e+1;
        il = ib.s-1;
        iu = ib.e+1;
    } else {
        kl = kb.s;
        ku = kb.e;
        jl = jb.s;
        ju = jb.e;
        il = ib.s-1;
        iu = ib.e+1;
    }
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Discontinuity detector calculation", parthenon::DevExecSpace(), 0,
        prim_pack.GetDim(5) - 1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            auto &indicator = indicator_pack(b);
            const auto &prim = prim_pack(b);

            Real rho_xp1 = prim(IDN, k, j, i+1);
            Real rho_x = prim(IDN, k, j, i);
            Real rho_xm1 = prim(IDN, k, j, i-1);

            Real rho_yp1 = prim(IDN, k, j+1, i);
            Real rho_y   = prim(IDN, k, j, i);
            Real rho_ym1 = prim(IDN, k, j-1, i);

            Real rho_zp1 = ndim > 2 ? prim(IDN, k+1, j, i) : 0.0;
            Real rho_z   = ndim > 2 ? prim(IDN, k, j, i) : 0.0;
            Real rho_zm1 = ndim > 2 ? prim(IDN, k-1, j, i) : 0.0;

            Real p_xp1 = prim(IPR, k, j, i+1);
            Real p_x = prim(IPR, k, j, i);
            Real p_xm1 = prim(IPR, k, j, i-1);

            Real p_yp1 = prim(IPR, k, j+1, i);
            Real p_y = prim(IPR, k, j, i);
            Real p_ym1 = prim(IPR, k, j-1, i);

            Real p_zp1 = ndim > 2 ? prim(IPR, k+1, j, i) : 0.0;
            Real p_z   = ndim > 2 ? prim(IPR, k, j, i) : 0.0;
            Real p_zm1 = ndim > 2 ? prim(IPR, k-1, j, i) : 0.0;

            Real pmag_xp1 = 0.5 * (SQR(prim(IB1, k, j, i+1)) + SQR(prim(IB2, k, j, i+1)) + SQR(prim(IB3, k, j, i+1)));
            Real pmag_x =   0.5 * (SQR(prim(IB1, k, j, i)) + SQR(prim(IB2, k, j, i)) + SQR(prim(IB3, k, j, i)));
            Real pmag_xm1 = 0.5 * (SQR(prim(IB1, k, j, i-1)) + SQR(prim(IB2, k, j, i-1)) + SQR(prim(IB3, k, j, i-1)));

            Real pmag_yp1 = 0.5 * (SQR(prim(IB1, k, j+1, i)) + SQR(prim(IB2, k, j+1, i)) + SQR(prim(IB3, k, j+1, i)));
            Real pmag_y =   0.5 * (SQR(prim(IB1, k, j, i)) + SQR(prim(IB2, k, j, i)) + SQR(prim(IB3, k, j, i)));
            Real pmag_ym1 = 0.5 * (SQR(prim(IB1, k, j-1, i)) + SQR(prim(IB2, k, j-1, i)) + SQR(prim(IB3, k, j-1, i)));

            Real pmag_zp1 = ndim > 2 ? 0.5 * (SQR(prim(IB1, k+1, j, i)) + SQR(prim(IB2, k+1, j, i)) + SQR(prim(IB3, k+1, j, i))) : 0.0;
            Real pmag_z   = ndim > 2 ? 0.5 * (SQR(prim(IB1, k, j, i)) + SQR(prim(IB2, k, j, i)) + SQR(prim(IB3, k, j, i))) : 0.0;
            Real pmag_zm1 = ndim > 2 ? 0.5 * (SQR(prim(IB1, k-1, j, i)) + SQR(prim(IB2, k-1, j, i)) + SQR(prim(IB3, k-1, j, i))) : 0.0;

            Real Sx_rho = fabs(rho_xp1 - 2.0*rho_x + rho_xm1) / 
                    (fabs(rho_xp1) + 2.0*fabs(rho_x) + 
                    fabs(rho_xm1) + dd_eps);

            Real Sx_p   = fabs(p_xp1 - 2.0*p_x + p_xm1) / 
                    (fabs(p_xp1) + 2.0*fabs(p_x) + 
                    fabs(p_xm1) + dd_eps);

            Real Sx_pmag = fabs(pmag_xp1 - 2.0*pmag_x + pmag_xm1) / 
                    (fabs(pmag_xp1) + 2.0*fabs(pmag_x) + 
                    fabs(pmag_xm1) + dd_eps);

            Real eta_x = fmax(fmax(Sx_rho, Sx_p), Sx_pmag);

            Real Sy_rho = fabs(rho_yp1 - 2.0*rho_y + rho_ym1) / 
                    (fabs(rho_yp1) + 2.0*fabs(rho_y) + 
                    fabs(rho_ym1) + dd_eps);

            Real Sy_p = fabs(p_yp1 - 2.0*p_y + p_ym1) / 
                    (fabs(p_yp1) + 2.0*fabs(p_y) + 
                    fabs(p_ym1) + dd_eps);

            Real Sy_pmag = fabs(pmag_yp1 - 2.0*pmag_y + pmag_ym1) / 
                    (fabs(pmag_yp1) + 2.0*fabs(pmag_y) + 
                    fabs(pmag_ym1) + dd_eps);

            Real eta_y = fmax(fmax(Sy_rho, Sy_p), Sy_pmag);

            Real Sz_rho = ndim > 2 ? 
                    fabs(rho_zp1 - 2.0*rho_z + rho_zm1) / 
                    (fabs(rho_zp1) + 2.0*fabs(rho_z) + 
                    fabs(rho_zm1) + dd_eps) : 0.0;

            Real Sz_p = ndim > 2 ? 
                    fabs(p_zp1 - 2.0*p_z + p_zm1) / 
                    (fabs(p_zp1) + 2.0*fabs(p_z) + 
                    fabs(p_zm1) + dd_eps) : 0.0;

            Real Sz_pmag = ndim > 2 ? 
                    fabs(pmag_zp1 - 2.0*pmag_z + pmag_zm1) / 
                    (fabs(pmag_zp1) + 2.0*fabs(pmag_z) + 
                    fabs(pmag_zm1) + dd_eps) : 0.0;

            Real eta_z = fmax(fmax(Sz_rho, Sz_p), Sz_pmag);

            Real eta_c = sqrt(SQR(eta_x) + SQR(eta_y) + SQR(eta_z));

            indicator(0, k, j, i) = eta_c;



        });

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Troubled array filling", parthenon::DevExecSpace(), 0,
        prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto &indicator = indicator_pack(b);
            auto &troubled  = troubled_pack(b);

            // troubled means that Berta's theta = 0.0 in that cell !
            if (ndim > 2){
                if (indicator(0,k,j,i) >= dd_threshold || indicator(0,k,j,i+1) >= dd_threshold || indicator(0,k,j+1,i) >= dd_threshold || indicator(0,k+1,j,i) >= dd_threshold || indicator(0,k,j,i-1) >= dd_threshold || indicator(0,k,j-1,i) >= dd_threshold || indicator(0,k-1,j,i) >= dd_threshold){
                    troubled(0,k,j,i) = 1.0;
                } else {
                    troubled(0,k,j,i) = 0.0;
                }
            } else if (ndim > 1) {
                if (indicator(0,k,j,i) >= dd_threshold || indicator(0,k,j,i+1) >= dd_threshold || indicator(0,k,j+1,i) >= dd_threshold || indicator(0,k,j,i-1) >= dd_threshold || indicator(0,k,j-1,i) >= dd_threshold){
                    troubled(0,k,j,i) = 1.0;
                } else {
                    troubled(0,k,j,i) = 0.0;
                }
            } else {
                if (indicator(0,k,j,i) >= dd_threshold || indicator(0,k,j,i+1) >= dd_threshold || indicator(0,k,j,i-1) >= dd_threshold){
                    troubled(0,k,j,i) = 1.0;
                } else {
                    troubled(0,k,j,i) = 0.0;
                }
            }

        });


    return TaskStatus::complete;
}

KOKKOS_INLINE_FUNCTION
Real CalculateHODRatio(const Real qm2, const Real qm1, const Real q0,
                       const Real qp1, const Real qp2, const Real qref,
                       const Real eps) {
    const Real d1 = 0.5 * (qp1 - qm1);
    const Real d2 = qp1 - 2.0 * q0 + qm1;
    const Real d3 = 0.5 * (qp2 - 2.0 * qp1 + 2.0 * qm1 - qm2);
    const Real d4 = qp2 - 4.0 * qp1 + 6.0 * q0 - 4.0 * qm1 + qm2;

    const Real eta_odd =
        fabs(d3) / (fabs(qref) + fabs(d1) + fabs(d3) + eps);
    const Real eta_even =
        fabs(d4) / (fabs(qref) + fabs(d2) + fabs(d4) + eps);

    return fmax(eta_odd, eta_even);
}

TaskStatus CalculateHODShockDetector(MeshData<Real> *md){
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;

    auto pkg = pmb->packages.Get("Hydro");

    const auto &dd_eps = pkg->Param<Real>("hydro/discontinuity_detector_epsilon");
    const auto &dd_threshold = pkg->Param<Real>("hydro/discontinuity_detector_threshold");

    // prim should hold the the second-order conversion of the
    // averaged conserved state at this time, which means
    // it already has cell-centered B available from the
    // face-averaging procedure
    const auto prim_pack = md->PackVariables(std::vector<std::string>{"prim"});
    auto indicator_pack = md->PackVariables(std::vector<std::string>{"berta24_shock_indicator"});
    auto troubled_pack  = md->PackVariables(std::vector<std::string>{"berta24_troubled"});
    
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    int kl, ku, jl, ju, il, iu;
    if (ndim > 2){
        kl = kb.s - 1;
        ku = kb.e + 1;
        jl = jb.s - 1;
        ju = jb.e + 1;
        il = ib.s - 1;
        iu = ib.e + 1;
    } else if (ndim > 1){
        kl = kb.s;
        ku = kb.e;
        jl = jb.s-1;
        ju = jb.e+1;
        il = ib.s-1;
        iu = ib.e+1;
    } else {
        kl = kb.s;
        ku = kb.e;
        jl = jb.s;
        ju = jb.e;
        il = ib.s-1;
        iu = ib.e+1;
    }
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Discontinuity detector calculation", parthenon::DevExecSpace(), 0,
        prim_pack.GetDim(5) - 1, kl, ku, jl, ju, il, iu,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            auto &indicator = indicator_pack(b);
            const auto &prim = prim_pack(b);

            const Real density_ref = prim(IDN, k, j, i);
            const Real pressure_ref = prim(IPR, k, j, i);
            const Real velocity_ref = sqrt(pressure_ref / density_ref);
            const Real magnetic_ref =
                sqrt(SQR(prim(IB1, k, j, i)) +
                     SQR(prim(IB2, k, j, i)) +
                     SQR(prim(IB3, k, j, i)));

            Real eta_x = 0.0;
            Real eta_y = 0.0;
            Real eta_z = 0.0;

            for (int n = IDN; n <= IB3; ++n) {
                Real qref;
                if (n == IDN) {
                    qref = density_ref;
                } else if (n == IPR) {
                    qref = pressure_ref;
                } else if (n >= IV1 && n <= IV3) {
                    qref = velocity_ref;
                } else {
                    qref = magnetic_ref;
                }

                const Real eta_qx = CalculateHODRatio(
                    prim(n, k, j, i - 2), prim(n, k, j, i - 1),
                    prim(n, k, j, i), prim(n, k, j, i + 1),
                    prim(n, k, j, i + 2), qref, dd_eps);
                eta_x = fmax(eta_x, eta_qx);

                if (ndim > 1) {
                    const Real eta_qy = CalculateHODRatio(
                        prim(n, k, j - 2, i), prim(n, k, j - 1, i),
                        prim(n, k, j, i), prim(n, k, j + 1, i),
                        prim(n, k, j + 2, i), qref, dd_eps);
                    eta_y = fmax(eta_y, eta_qy);
                }

                if (ndim > 2) {
                    const Real eta_qz = CalculateHODRatio(
                        prim(n, k - 2, j, i), prim(n, k - 1, j, i),
                        prim(n, k, j, i), prim(n, k + 1, j, i),
                        prim(n, k + 2, j, i), qref, dd_eps);
                    eta_z = fmax(eta_z, eta_qz);
                }
            }

            indicator(0, k, j, i) =
                sqrt(SQR(eta_x) + SQR(eta_y) + SQR(eta_z));
        });

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Troubled array filling", parthenon::DevExecSpace(), 0,
        prim_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto &indicator = indicator_pack(b);
            auto &troubled  = troubled_pack(b);

            // troubled means that Berta's theta = 0.0 in that cell !
            if (ndim > 2){
                if (indicator(0,k,j,i) >= dd_threshold || indicator(0,k,j,i+1) >= dd_threshold || indicator(0,k,j+1,i) >= dd_threshold || indicator(0,k+1,j,i) >= dd_threshold || indicator(0,k,j,i-1) >= dd_threshold || indicator(0,k,j-1,i) >= dd_threshold || indicator(0,k-1,j,i) >= dd_threshold){
                    troubled(0,k,j,i) = 1.0;
                } else {
                    troubled(0,k,j,i) = 0.0;
                }
            } else if (ndim > 1) {
                if (indicator(0,k,j,i) >= dd_threshold || indicator(0,k,j,i+1) >= dd_threshold || indicator(0,k,j+1,i) >= dd_threshold || indicator(0,k,j,i-1) >= dd_threshold || indicator(0,k,j-1,i) >= dd_threshold){
                    troubled(0,k,j,i) = 1.0;
                } else {
                    troubled(0,k,j,i) = 0.0;
                }
            } else {
                if (indicator(0,k,j,i) >= dd_threshold || indicator(0,k,j,i+1) >= dd_threshold || indicator(0,k,j,i-1) >= dd_threshold){
                    troubled(0,k,j,i) = 1.0;
                } else {
                    troubled(0,k,j,i) = 0.0;
                }
            }

        });


    return TaskStatus::complete;
}

template <bool pointwise, Reconstruction edge_recon = Reconstruction::plm>
TaskStatus Assemble_HLLD_Edge_EMF_Impl(MeshData<Real> *md) {
    static_assert(pointwise || edge_recon == Reconstruction::plm ||
                      edge_recon == Reconstruction::weno3,
                  "Unsupported reconstruction for averaged UCT edge states.");
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;

    const auto &detector =
        pmb->packages.Get("Hydro")
            ->Param<std::string>("hydro/discontinuity_detector");

    const bool use_order_reduction = detector != "none";

    auto cons_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
    // the edge EMFs will be stored in the fluxes of Bface

    auto B_pack = md->PackVariablesAndFluxes(
        std::vector<std::string>{pointwise ? "Bface_point" : "Bface"});

    const auto &uct_hlld_pack = md->PackVariables(std::vector<std::string>{"uct_hlld"});

    auto troubled_pack  = md->PackVariables(std::vector<std::string>{"berta24_troubled"});

    
    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);


    // loops need to run over the interior domain + 1 so 
    // they reach the end of the mesh
    // for z-directed edges (Ez_edges)
    int kl, ku, jl, ju, il, iu;
    if (pointwise && ndim > 2){
        kl = kb.s - 1;
        ku = kb.e + 1;
        jl = jb.s - 1;
        ju = jb.e + 1;
        il = ib.s - 1;
        iu = ib.e + 1;
    }
    else {
        kl = kb.s;
        ku = kb.e;
        jl = jb.s;
        ju = jb.e;
        il = ib.s;
        iu = ib.e;
    }
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Assemble Ez_edges", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kl, ku, jb.s, jb.e+1, ib.s, ib.e+1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            auto &Bface = B_pack(b);
            const auto &uct_hlld = uct_hlld_pack(b);
            const auto &troubled = troubled_pack(b);

            const bool bad =
                use_order_reduction &&
                (troubled(0, k, j, i - 1) == 1.0 ||
                 troubled(0, k, j, i) == 1.0 ||
                 troubled(0, k, j - 1, i) == 1.0 ||
                 troubled(0, k, j - 1, i - 1) == 1.0);


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
            if (pointwise && !bad){ 
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
            } else if (pointwise && bad) {
                PLM(Bface(TE::F2, 0, k, j, i-2),
                    Bface(TE::F2, 0, k, j, i-1),
                    Bface(TE::F2, 0, k, j, i  ),
                    ByW, unused);

                PLM(Bface(TE::F2, 0, k, j, i-1),
                    Bface(TE::F2, 0, k, j, i),
                    Bface(TE::F2, 0, k, j, i+1),
                    unused, ByE);

                PLM(Bface(TE::F1, 0, k, j-2, i),
                    Bface(TE::F1, 0, k, j-1, i),
                    Bface(TE::F1, 0, k, j, i  ),
                    BxS, unused);

                PLM(Bface(TE::F1, 0, k, j-1, i),
                    Bface(TE::F1, 0, k, j, i),
                    Bface(TE::F1, 0, k, j+1, i),
                    unused, BxN);
                
                // reconstruct velocities
                PLM(uct_hlld(TE::F2, VBART2, k, j, i-2),
                    uct_hlld(TE::F2, VBART2, k, j, i-1),
                    uct_hlld(TE::F2, VBART2, k, j, i  ),
                    vxW, unused);

                PLM(uct_hlld(TE::F2, VBART2, k, j, i-1),
                    uct_hlld(TE::F2, VBART2, k, j, i),
                    uct_hlld(TE::F2, VBART2, k, j, i+1),
                    unused, vxE);
                
                PLM(uct_hlld(TE::F1, VBART1, k, j-2, i),
                    uct_hlld(TE::F1, VBART1, k, j-1, i),
                    uct_hlld(TE::F1, VBART1, k, j, i  ),
                    vyS, unused);

                PLM(uct_hlld(TE::F1, VBART1, k, j-1, i),
                    uct_hlld(TE::F1, VBART1, k, j, i),
                    uct_hlld(TE::F1, VBART1, k, j+1, i),
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
            cons_pack.GetDim(5) - 1, kb.s, kb.e+1, jl, ju, ib.s, ib.e+1,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                auto &Bface = B_pack(b);
                const auto &uct_hlld = uct_hlld_pack(b);
                const auto &troubled = troubled_pack(b);

                const bool bad =
                    use_order_reduction &&
                    (troubled(0, k, j, i - 1) == 1.0 ||
                     troubled(0, k, j, i) == 1.0 ||
                     troubled(0, k - 1, j, i) == 1.0 ||
                     troubled(0, k - 1, j, i - 1) == 1.0);

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
                if (pointwise && !bad) {
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
            cons_pack.GetDim(5) - 1, kb.s, kb.e+1, jb.s, jb.e+1, il, iu,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                auto &Bface = B_pack(b);
                const auto &uct_hlld = uct_hlld_pack(b);
                const auto &troubled = troubled_pack(b);

                const bool bad =
                    use_order_reduction &&
                    (troubled(0, k, j - 1, i) == 1.0 ||
                     troubled(0, k, j, i) == 1.0 ||
                     troubled(0, k - 1, j, i) == 1.0 ||
                     troubled(0, k - 1, j - 1, i) == 1.0);

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
                if (pointwise && !bad) {
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

TaskStatus averageToPoint(MeshData<Real> *md){
    /*
    This function approximates volume- and face-averaged
    quantities with centered point-wise quantities
    */
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    const auto &detector =
        pmb->packages.Get("Hydro")
            ->Param<std::string>("hydro/discontinuity_detector");

    const bool use_order_reduction = detector != "none";

    auto cons_pack = md->PackVariables(std::vector<std::string>{"cons"});
    auto Bface_pack = md->PackVariables(std::vector<std::string>{"Bface"});


    auto cons_point_pack = md->PackVariables(std::vector<std::string>{"cons_point"});
    auto Bface_point_pack = md->PackVariables(std::vector<std::string>{"Bface_point"});

    const auto troubled_pack = md->PackVariables(std::vector<std::string>{"berta24_troubled"});

    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Vol-Average to Point", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e, 0, 4,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const int n) {
            auto &cons_point = cons_point_pack(b);
            const auto &cons = cons_pack(b);
            const auto &troubled = troubled_pack(b);
            // is the cell bad or good?
            const bool bad = use_order_reduction && troubled(0, k, j, i) == 1.0;

            if (bad){
                cons_point(n, k, j, i) = cons(n, k, j, i);
            } else {
                Real dxQ = cons(n, k, j, i-1) - 2.0*cons(n, k, j, i) + cons(n, k, j, i+1);
                Real dyQ = ndim > 1 ? cons(n, k, j-1, i) - 2.0*cons(n, k, j, i) + cons(n, k, j+1, i) : 0.0;
                Real dzQ = ndim>2  ? cons(n, k-1, j, i) - 2.0*cons(n, k, j, i) + cons(n, k+1, j, i) : 0.0;
                Real dQ  = dxQ + dyQ + dzQ;
                cons_point(n, k, j, i) = cons(n, k, j, i) - (dQ / 24.0);
            }

            // Magnetic components transverse to all active dimensions are evolved
            // cell-centrally rather than through CT. In 2D this is B3; in 1D
            // these are B2 and B3.
            if (n == IEN) {
                for (int nb = IB1 + ndim; nb <= IB3; ++nb) {
                    if (bad){
                        cons_point(nb, k, j, i) = cons(nb, k, j, i);
                    } else {
                        const Real dxB = cons(nb, k, j, i-1) -
                                        2.0*cons(nb, k, j, i) +
                                        cons(nb, k, j, i+1);
                        const Real dyB = ndim > 1
                                            ? cons(nb, k, j-1, i) -
                                                2.0*cons(nb, k, j, i) +
                                                cons(nb, k, j+1, i)
                                            : 0.0;
                        cons_point(nb, k, j, i) =
                            cons(nb, k, j, i) - ((dxB + dyB) / 24.0);
                    }
                }
            }

        });


    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Face-x - Average to Point", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s-1, ib.e+2,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            auto &Bface_point = Bface_point_pack(b);
            const auto &Bface = Bface_pack(b);
            const auto &troubled = troubled_pack(b);
            // is the cell bad or good?
            const bool bad =
                use_order_reduction &&
                (troubled(0, k, j, i) == 1.0 || troubled(0, k, j, i - 1) == 1.0);

            if (bad){
                Bface_point(TE::F1, 0, k, j, i) = Bface(TE::F1, 0, k, j, i);
            } else {
                Real dyFacex = Bface(TE::F1, 0, k, j-1, i) - 2.0*Bface(TE::F1, 0, k, j, i) + Bface(TE::F1, 0, k, j+1, i);
                Real dzFacex = ndim > 2 ? Bface(TE::F1, 0, k-1, j, i) - 2.0*Bface(TE::F1, 0, k, j, i) + Bface(TE::F1, 0, k+1, j, i) : 0.0;

                Real dxTBx = dyFacex + dzFacex;

                Bface_point(TE::F1, 0, k, j, i) = Bface(TE::F1, 0, k, j, i) - (dxTBx / 24.0);
            }
        });

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Face-y - Average to Point", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s-1, jb.e+2, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            auto &Bface_point = Bface_point_pack(b);
            const auto &Bface = Bface_pack(b);
            const auto &troubled = troubled_pack(b);
            // is the cell bad or good?
            const bool bad =
                use_order_reduction &&
                (troubled(0, k, j, i) == 1.0 || troubled(0, k, j - 1, i) == 1.0);

            if (bad){
                Bface_point(TE::F2, 0, k, j, i) = Bface(TE::F2, 0, k, j, i);
            } else {
                Real dxFacey = Bface(TE::F2, 0, k, j, i-1) - 2.0*Bface(TE::F2, 0, k, j, i) + Bface(TE::F2, 0, k, j, i+1);
                Real dzFacey = ndim > 2 ? Bface(TE::F2, 0, k-1, j, i) - 2.0*Bface(TE::F2, 0, k, j, i) + Bface(TE::F2, 0, k+1, j, i) : 0.0;

                Real dyTBy = dxFacey + dzFacey;

                Bface_point(TE::F2, 0, k, j, i) = Bface(TE::F2, 0, k, j, i) - (dyTBy / 24.0);
            }
        });

    if (ndim > 2){
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "Face-z - Average to Point", parthenon::DevExecSpace(), 0,
            cons_pack.GetDim(5) - 1, kb.s-1, kb.e+2, jb.s, jb.e, ib.s, ib.e,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                auto &Bface_point = Bface_point_pack(b);
                const auto &Bface = Bface_pack(b);
                const auto &troubled = troubled_pack(b);
                // is the cell bad or good?
                const bool bad =
                    use_order_reduction &&
                    (troubled(0, k, j, i) == 1.0 || troubled(0, k - 1, j, i) == 1.0);

                if (bad){
                    Bface_point(TE::F3, 0, k, j, i) = Bface(TE::F3, 0, k, j, i);
                } else {
                    Real dxFacez = Bface(TE::F3, 0, k, j, i-1) - 2.0*Bface(TE::F3, 0, k, j, i) + Bface(TE::F3, 0, k, j, i+1);
                    Real dyFacez = Bface(TE::F3, 0, k, j-1, i) - 2.0*Bface(TE::F3, 0, k, j, i) + Bface(TE::F3, 0, k, j+1, i);

                    Real dzTBz = dxFacez + dyFacez;

                    Bface_point(TE::F3, 0, k, j, i) = Bface(TE::F3, 0, k, j, i) - (dzTBz / 24.0);
                }
            });
    }


    return TaskStatus::complete;
}

TaskStatus pointToAverage(MeshData<Real> *md){
    /*
    This function approximates point-wise quantities with
    volume- and face-averaged quantities at 4th order
    */
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;

    const auto &detector =
        pmb->packages.Get("Hydro")
            ->Param<std::string>("hydro/discontinuity_detector");

    const bool use_order_reduction = detector != "none";

    auto cons_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
    auto Bface_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"Bface"});

    auto consflux_point_pack = md->PackVariables(std::vector<std::string>{"consflux_point"});
    auto Bface_point_pack = md->PackVariablesAndFluxes(std::vector<std::string>{"Bface_point"});
    auto troubled_pack = md->PackVariables(std::vector<std::string>{"berta24_troubled"});


    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);


    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Flux-x - Point to Average", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e, ib.s, ib.e+1, 0, 7,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const int n) {
            const auto &consflux_point = consflux_point_pack(b);
            auto &cons = cons_pack(b);
            const auto &troubled = troubled_pack(b);
            const bool bad =
                use_order_reduction &&
                (troubled(0, k, j, i - 1) == 1.0 ||
                 troubled(0, k, j, i) == 1.0);

            if (bad) {
                cons.flux(X1DIR, n, k, j, i) = consflux_point(TE::F1, n, k, j, i);
            } else {
                Real dyFacex = consflux_point(TE::F1, n, k, j-1, i) - 2.0*consflux_point(TE::F1, n, k, j, i) + consflux_point(TE::F1, n, k, j+1, i);
                Real dzFacex = ndim > 2 ? consflux_point(TE::F1, n, k-1, j, i) - 2.0*consflux_point(TE::F1, n, k, j, i) + consflux_point(TE::F1, n, k+1, j, i) : 0.0;

                Real dxTBx = dyFacex + dzFacex;

                cons.flux(X1DIR, n, k, j, i) = consflux_point(TE::F1, n, k, j, i) + (dxTBx / 24.0);
            }

        });

    if (ndim > 1){
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Face-y - Point to Average", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e+1, ib.s, ib.e, 0, 7,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const int n) {
            const auto &consflux_point = consflux_point_pack(b);
            auto &cons = cons_pack(b);
            const auto &troubled = troubled_pack(b);
            const bool bad =
                use_order_reduction &&
                (troubled(0, k, j - 1, i) == 1.0 ||
                 troubled(0, k, j, i) == 1.0);

            if (bad){
                cons.flux(X2DIR, n, k, j, i) = consflux_point(TE::F2, n, k, j, i);
            } else {
                Real dxFacey = consflux_point(TE::F2, n, k, j, i-1) - 2.0*consflux_point(TE::F2, n, k, j, i) + consflux_point(TE::F2, n, k, j, i+1);
                Real dzFacey = ndim > 2 ? consflux_point(TE::F2, n, k-1, j, i) - 2.0*consflux_point(TE::F2, n, k, j, i) + consflux_point(TE::F2, n, k+1, j, i) : 0.0;

                Real dyTBy = dxFacey + dzFacey;

                cons.flux(X2DIR, n, k, j, i) = consflux_point(TE::F2, n, k, j, i) + (dyTBy / 24.0);
            }

        });
    }

    if (ndim > 2){
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "Face-z - Point to Average", parthenon::DevExecSpace(), 0,
            cons_pack.GetDim(5) - 1, kb.s, kb.e+1, jb.s, jb.e, ib.s, ib.e, 0, 7,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i, const int n) {
                const auto &consflux_point = consflux_point_pack(b);
                auto &cons = cons_pack(b);
                const auto &troubled = troubled_pack(b);
                const bool bad =
                    use_order_reduction &&
                    (troubled(0, k - 1, j, i) == 1.0 ||
                     troubled(0, k, j, i) == 1.0);

                if (bad) {
                    cons.flux(X3DIR, n, k, j, i) = consflux_point(TE::F3, n, k, j, i);
                } else {
                    Real dxFacez = consflux_point(TE::F3, n, k, j, i-1) - 2.0*consflux_point(TE::F3, n, k, j, i) + consflux_point(TE::F3, n, k, j, i+1);
                    Real dyFacez = consflux_point(TE::F3, n, k, j-1, i) - 2.0*consflux_point(TE::F3, n, k, j, i) + consflux_point(TE::F3, n, k, j+1, i);

                    Real dzTBz = dxFacez + dyFacez;

                    cons.flux(X3DIR, n, k, j, i) = consflux_point(TE::F3, n, k, j, i) + (dzTBz / 24.0);
                }

            });
    }

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Edge-z - Point to Average", parthenon::DevExecSpace(), 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e+1, ib.s, ib.e+1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            const auto &Bface_point = Bface_point_pack(b);
            auto &Bface = Bface_pack(b);
            const auto &troubled = troubled_pack(b);
            const bool bad =
                use_order_reduction &&
                (troubled(0, k, j, i - 1) == 1.0 ||
                    troubled(0, k, j, i) == 1.0 ||
                    troubled(0, k, j - 1, i) == 1.0 ||
                    troubled(0, k, j - 1, i - 1) == 1.0);

            const Real emfz_z =
                    Bface_point.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i);
            if (bad) {
                Bface.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i) = emfz_z;
            } else {
                Real dzEdgez;
                if (ndim > 2){
                    const Real emfz_zm1 =
                        Bface_point.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k-1, j, i);
                    const Real emfz_zp1 =
                        Bface_point.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k+1, j, i);

                    dzEdgez = emfz_zm1 - 2.0*emfz_z + emfz_zp1;
                } else {
                    dzEdgez = 0.0;
                }

                Bface.template flux<parthenon::TopologicalType::Edge>(X3DIR, 0, k, j, i) = emfz_z + (dzEdgez / 24.0);
            }
        });

    if (ndim > 2){
        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "Edge-y - Point to Average", parthenon::DevExecSpace(), 0,
            cons_pack.GetDim(5) - 1, kb.s, kb.e+1, jb.s, jb.e, ib.s, ib.e+1,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                const auto &Bface_point = Bface_point_pack(b);
                auto &Bface = Bface_pack(b);
                const auto &troubled = troubled_pack(b);
                const bool bad =
                    use_order_reduction &&
                    (troubled(0, k, j, i - 1) == 1.0 ||
                        troubled(0, k, j, i) == 1.0 ||
                        troubled(0, k - 1, j, i) == 1.0 ||
                        troubled(0, k - 1, j, i - 1) == 1.0);
                
                const Real emfy_y =
                    Bface_point.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i);
                
                if (bad) {
                    Bface.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i) = emfy_y;
                } else {
                    const Real emfy_ym1 =
                        Bface_point.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j-1, i);
                    const Real emfy_yp1 =
                        Bface_point.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j+1, i);



                    Real dyEdgey = emfy_ym1 - 2.0*emfy_y + emfy_yp1;

                    Bface.template flux<parthenon::TopologicalType::Edge>(X2DIR, 0, k, j, i) = emfy_y + (dyEdgey / 24.0);
                }
            });

        parthenon::par_for(
            DEFAULT_LOOP_PATTERN, "Edge-x - Point to Average", parthenon::DevExecSpace(), 0,
            cons_pack.GetDim(5) - 1, kb.s, kb.e+1, jb.s, jb.e+1, ib.s, ib.e,
            KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
                const auto &Bface_point = Bface_point_pack(b);
                auto &Bface = Bface_pack(b);
                const auto &troubled = troubled_pack(b);
                const bool bad =
                    use_order_reduction &&
                    (troubled(0, k, j - 1, i) == 1.0 ||
                        troubled(0, k, j, i) == 1.0 ||
                        troubled(0, k - 1, j, i) == 1.0 ||
                        troubled(0, k - 1, j - 1, i) == 1.0);

                const Real emfx_x =
                    Bface_point.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i);
                
                if (bad) {
                    Bface.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i) = emfx_x;
                } else {
                    const Real emfx_xm1 =
                        Bface_point.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i-1);
                    const Real emfx_xp1 =
                        Bface_point.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i+1);

                    Real dxEdgex = emfx_xm1 - 2.0*emfx_x + emfx_xp1;

                    Bface.template flux<parthenon::TopologicalType::Edge>(X1DIR, 0, k, j, i) = emfx_x + (dxEdgex / 24.0);
                }
            });
    }

    return TaskStatus::complete;
}


TaskStatus CenterPointMagField(MeshData<Real> *md) {
    auto pmb = md->GetBlockData(0)->GetBlockPointer();
    const int ndim = pmb->pmy_mesh->ndim;
    const auto &detector =
        pmb->packages.Get("Hydro")
            ->Param<std::string>("hydro/discontinuity_detector");

    const bool use_order_reduction = detector != "none";

    auto cons_point_pack =
        md->PackVariables(std::vector<std::string>{"cons_point"});
    auto Bface_point_pack =
        md->PackVariables(std::vector<std::string>{"Bface_point"});
    const auto Bface_pack =
        md->PackVariables(std::vector<std::string>{"Bface"});

    const auto troubled_pack = md->PackVariables(std::vector<std::string>{"berta24_troubled"});

    IndexRange ib = md->GetBlockData(0)->GetBoundsI(IndexDomain::interior);
    IndexRange jb = md->GetBlockData(0)->GetBoundsJ(IndexDomain::interior);
    IndexRange kb = md->GetBlockData(0)->GetBoundsK(IndexDomain::interior);

    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Center point magnetic field",
        parthenon::DevExecSpace(), 0, cons_point_pack.GetDim(5) - 1,
        kb.s, kb.e, jb.s, jb.e, ib.s, ib.e,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
            auto &cons_point = cons_point_pack(b);
            const auto &Bface_point = Bface_point_pack(b);
            const auto &Bface = Bface_pack(b);
            const auto &troubled = troubled_pack(b);
            const bool bad = use_order_reduction && troubled(0, k, j, i) == 1.0;

            if (bad) {
                cons_point(IB1, k,j,i) = 
                          0.5 *(Bface(TE::F1, 0, k, j, i) +
                                Bface(TE::F1, 0, k, j, i+1));
                if (ndim > 1) {
                    cons_point(IB2, k, j, i) =
                        0.5 * (Bface(TE::F2, 0, k, j, i) +
                               Bface(TE::F2, 0, k, j+1, i));
                }
                if (ndim > 2) {
                    cons_point(IB3, k, j, i) =
                        0.5 * (Bface(TE::F3, 0, k, j, i) +
                                Bface(TE::F3, 0, k+1, j, i));
                }
            } else {
                cons_point(IB1, k, j, i) =
                    (9.0 * (Bface_point(TE::F1, 0, k, j, i) +
                            Bface_point(TE::F1, 0, k, j, i+1)) -
                    Bface_point(TE::F1, 0, k, j, i-1) -
                    Bface_point(TE::F1, 0, k, j, i+2)) / 16.0;

                if (ndim > 1) {
                    cons_point(IB2, k, j, i) =
                        (9.0 * (Bface_point(TE::F2, 0, k, j, i) +
                                Bface_point(TE::F2, 0, k, j+1, i)) -
                        Bface_point(TE::F2, 0, k, j-1, i) -
                        Bface_point(TE::F2, 0, k, j+2, i)) / 16.0;
                }

                if (ndim > 2) {
                    cons_point(IB3, k, j, i) =
                        (9.0 * (Bface_point(TE::F3, 0, k, j, i) +
                                Bface_point(TE::F3, 0, k+1, j, i)) -
                        Bface_point(TE::F3, 0, k-1, j, i) -
                        Bface_point(TE::F3, 0, k+2, j, i)) / 16.0;
                }
            }
        });

    return TaskStatus::complete;
}


TaskStatus PointConsToPrim(MeshData<Real> *md) {
    const auto &eos = md->GetBlockData(0)
                          ->GetBlockPointer()
                          ->packages.Get("Hydro")
                          ->Param<AdiabaticCTMHDEOS>("eos");
    eos.PointConservedToPrimitive(md);
    return TaskStatus::complete;
}


} // namespace Hydro::UCTHLLDMHD
