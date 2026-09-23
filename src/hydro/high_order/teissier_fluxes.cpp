//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD
// code. Copyright (c) 2020-2021, Athena-Parthenon Collaboration. All rights
// reserved. Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"
#include "../../eos/adiabatic_ctmhd.hpp"
#include "../../eos/adiabatic_glmmhd.hpp"
#include "../../eos/adiabatic_hydro.hpp"
#include "../../recon/weno5_classic.hpp"
#include "../../recon/weno7_classic.hpp"
#include "../../recon/weno9_classic.hpp"
#include "../rsolvers/rsolvers.hpp"
#include "teissier_fluxes.hpp"
#include "teissier_transforms.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;


namespace Hydro::HIGHORDER {


template <int ORDER, int DIR, typename FacePack>
KOKKOS_INLINE_FUNCTION UCTMHDPointConserved
LoadPointConserved(const FacePack &face_avg, const int ndim, const int k,
                   const int j, const int i) {
  constexpr auto face =
      DIR == X1DIR ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);
  UCTMHDPointConserved u;
  u.d = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IDN), ndim, k, j, i);
  u.m1 = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IM1), ndim, k, j, i);
  u.m2 = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IM2), ndim, k, j, i);
  u.m3 = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IM3), ndim, k, j, i);
  u.e = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IEN), ndim, k, j, i);
  u.b1 = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IB1), ndim, k, j, i);
  u.b2 = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IB2), ndim, k, j, i);
  u.b3 = FaceAverageToPoint<ORDER, DIR>(face_avg(face, IB3), ndim, k, j, i);
  return u;
}

template <int ORDER, int DIR, typename FacePack, typename FluxPack, typename SpeedPack,
          typename ElectricPack>
KOKKOS_INLINE_FUNCTION void CalculateLLFPointFace(
    const FacePack &face_avg_left, const FacePack &face_avg_right,
    FluxPack &consflux_point, SpeedPack &llf_speed,
    ElectricPack &eface_point, const AdiabaticCTMHDEOS &eos,
    const int nhydro, const int nscalars, const int ndim, const int k,
    const int j, const int i) {
  constexpr auto face =
      DIR == X1DIR ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);

  UCTMHDPointConserved u =
      LoadPointConserved<ORDER, DIR>(face_avg_left, ndim, k, j, i);
  const Real bnormal = DIR == X1DIR ? u.b1 : (DIR == X2DIR ? u.b2 : u.b3);
  UCTMHDPointPrimitive wl;
  UCTMHDConsToPrimPoint(u, wl, eos);

  u = LoadPointConserved<ORDER, DIR>(face_avg_right, ndim, k, j, i);
  if constexpr (DIR == X1DIR) {
    u.b1 = bnormal;
  } else if constexpr (DIR == X2DIR) {
    u.b2 = bnormal;
  } else {
    u.b3 = bnormal;
  }
  UCTMHDPointPrimitive wr;
  UCTMHDConsToPrimPoint(u, wr, eos);

  Real speed;
  const auto flux =
      Riemann<Fluid::ucthlldmhd, RiemannSolver::llf>::SolveFace<DIR>(wl, wr,
                                                                    speed, eos);
  consflux_point(face, IDN, k, j, i) = flux.d;
  consflux_point(face, IM1, k, j, i) = flux.m1;
  consflux_point(face, IM2, k, j, i) = flux.m2;
  consflux_point(face, IM3, k, j, i) = flux.m3;
  consflux_point(face, IEN, k, j, i) = flux.e;
  consflux_point(face, IB1, k, j, i) = flux.b1;
  consflux_point(face, IB2, k, j, i) = flux.b2;
  consflux_point(face, IB3, k, j, i) = flux.b3;
  llf_speed(face, 0, k, j, i) = speed;

  if constexpr (DIR == X1DIR) {
    eface_point(face, ELEFT_T1, k, j, i) = wl.v1 * wl.b3 - wl.v3 * wl.b1;
    eface_point(face, ELEFT_T2, k, j, i) = wl.v2 * wl.b1 - wl.v1 * wl.b2;
    eface_point(face, ERIGHT_T1, k, j, i) = wr.v1 * wr.b3 - wr.v3 * wr.b1;
    eface_point(face, ERIGHT_T2, k, j, i) = wr.v2 * wr.b1 - wr.v1 * wr.b2;
  } else if constexpr (DIR == X2DIR) {
    eface_point(face, ELEFT_T1, k, j, i) = wl.v2 * wl.b1 - wl.v1 * wl.b2;
    eface_point(face, ELEFT_T2, k, j, i) = wl.v3 * wl.b2 - wl.v2 * wl.b3;
    eface_point(face, ERIGHT_T1, k, j, i) = wr.v2 * wr.b1 - wr.v1 * wr.b2;
    eface_point(face, ERIGHT_T2, k, j, i) = wr.v3 * wr.b2 - wr.v2 * wr.b3;
  } else {
    eface_point(face, ELEFT_T1, k, j, i) = wl.v3 * wl.b2 - wl.v2 * wl.b3;
    eface_point(face, ELEFT_T2, k, j, i) = wl.v1 * wl.b3 - wl.v3 * wl.b1;
    eface_point(face, ERIGHT_T1, k, j, i) = wr.v3 * wr.b2 - wr.v2 * wr.b3;
    eface_point(face, ERIGHT_T2, k, j, i) = wr.v1 * wr.b3 - wr.v3 * wr.b1;
  }

  for (int n = nhydro; n < nhydro + nscalars; ++n) {
    const Real scalar_l = FaceAverageToPoint<ORDER, DIR>(face_avg_left(face, n), ndim,
                                                  k, j, i) /
                          wl.d;
    const Real scalar_r = FaceAverageToPoint<ORDER, DIR>(face_avg_right(face, n), ndim,
                                                  k, j, i) /
                          wr.d;
    consflux_point(face, n, k, j, i) =
        flux.d * ((flux.d >= 0.0) ? scalar_l : scalar_r);
  }
}


TaskStatus ReconstructFaceAveragesTeissier(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  auto pkg = pmb->packages.Get("Hydro");
  const int order = pkg->Param<int>("convergence_order");
  const int transverse_halo = 2 * (order / 2 - 1);
  int il, iu, jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;

  // AtoP reads ORDER/2-1 transverse neighbors. Point-face quantities extend
  // through that same number of transverse layers for the subsequent PtoA,
  // so reconstructed face averages need twice that halo.
  if (pmb->block_size.nx(X2DIR) > 1) {
    jl = jb.s - transverse_halo;
    ju = jb.e + transverse_halo;
    if (pmb->block_size.nx(X3DIR) > 1) {
      kl = kb.s - transverse_halo;
      ku = kb.e + transverse_halo;
    }
  }

  auto cons_in =
      md->PackVariables(std::vector<std::string>{"cons"});
  MeshBlockPack<VariablePack<Real>> Bface_pack;
  MeshBlockPack<VariablePack<Real>> face_avg_left_pack;
  MeshBlockPack<VariablePack<Real>> face_avg_right_pack;
  
  Bface_pack =
        md->PackVariables(std::vector<std::string>{"Bface"});
  face_avg_left_pack =
        md->PackVariables(std::vector<std::string>{"face_avg_left"});
  face_avg_right_pack =
        md->PackVariables(std::vector<std::string>{"face_avg_right"});

  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");

  auto num_scratch_vars = nhydro + nscalars;

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 2;

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Teissier x1 face reconstruction", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku, jl, ju,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        const auto &cons = cons_in(b);
        const auto &Bface = Bface_pack(b);
        auto &face_avg_left  = face_avg_left_pack(b);
        auto &face_avg_right = face_avg_right_pack(b);
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        // get reconstructed state on faces
        if (order == 8) {
          Reconstruct<Reconstruction::weno9js, X1DIR>(member, k, j, ib.s - 1,
                                                      ib.e + 1, cons, wl, wr);
        } else if (order == 6) {
          Reconstruct<Reconstruction::weno7js, X1DIR>(member, k, j, ib.s - 1,
                                                      ib.e + 1, cons, wl, wr);
        } else {
          Reconstruct<Reconstruction::weno5js, X1DIR>(member, k, j, ib.s - 1,
                                                      ib.e + 1, cons, wl, wr);
        }
        // Sync all threads in the team so that scratch memory is consistent
        member.team_barrier();
        // force the evolved Bface vars to sit on both sides of the reconstructed face
        parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            wl(IB1, i) = Bface(TE::F1, 0, k, j, i);
            wr(IB1, i) = Bface(TE::F1, 0, k, j, i);
        });
        member.team_barrier();
  // need to loop over n and i
        parthenon::par_for_inner(member, 0, num_scratch_vars-1, ib.s, ib.e + 1, [&](const int n, const int i) {
            face_avg_left(TE::F1, n, k, j, i)  = wl(n, i);
            face_avg_right(TE::F1, n, k, j, i) = wr(n, i);
        });

        member.team_barrier();

      });

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 2;
    // The order-dependent transverse halo is wide enough for the point solve.
    il = ib.s - transverse_halo;
    iu = ib.e + transverse_halo;
    kl = kb.s;
    ku = kb.e;
    if (pmb->block_size.nx(X3DIR) > 1) {
      kl = kb.s - transverse_halo;
      ku = kb.e + transverse_halo;
    }

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Teissier x2 face reconstruction", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku, jb.s - 1,
        jb.e + 1,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k,
                      const int j) {
          const auto &cons = cons_in(b);
          const auto &Bface = Bface_pack(b);
          auto &face_avg_left = face_avg_left_pack(b);
          auto &face_avg_right = face_avg_right_pack(b);
          parthenon::ScratchPad2D<Real> ql(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> qr(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);

          // A reconstruction centered on j produces the left state at face
          // j+1 and the right state at face j. Assigning one center to each
          // team preserves the rolling algorithm's work without a serial j loop.
          if (order == 8) {
            Reconstruct<Reconstruction::weno9js, X2DIR>(member, k, j, il,
                                                         iu, cons, ql, qr);
          } else if (order == 6) {
            Reconstruct<Reconstruction::weno7js, X2DIR>(member, k, j, il,
                                                         iu, cons, ql, qr);
          } else {
            Reconstruct<Reconstruction::weno5js, X2DIR>(member, k, j, il,
                                                         iu, cons, ql, qr);
          }
          member.team_barrier();

          parthenon::par_for_inner(
              member, 0, num_scratch_vars - 1, il, iu,
              [&](const int n, const int i) {
                if (j <= jb.e) {
                  const int jf = j + 1;
                  face_avg_left(TE::F2, n, k, jf, i) =
                      (n == IB2) ? Bface(TE::F2, 0, k, jf, i) : ql(n, i);
                }
                if (j >= jb.s) {
                  face_avg_right(TE::F2, n, k, j, i) =
                      (n == IB2) ? Bface(TE::F2, 0, k, j, i) : qr(n, i);
                }
              });
        });
  }
  //--------------------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->ndim >= 3) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 2;
    // The order-dependent transverse halo is wide enough for the point solve.
    il = ib.s - transverse_halo;
    iu = ib.e + transverse_halo;
    jl = jb.s - transverse_halo;
    ju = jb.e + transverse_halo;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Teissier x3 face reconstruction", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, kb.s - 1, kb.e + 1, jl,
        ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k,
                      const int j) {
          const auto &cons = cons_in(b);
          const auto &Bface = Bface_pack(b);
          auto &face_avg_left = face_avg_left_pack(b);
          auto &face_avg_right = face_avg_right_pack(b);
          parthenon::ScratchPad2D<Real> ql(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> qr(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);

          // A reconstruction centered on k produces the left state at face
          // k+1 and the right state at face k. Each team writes unique left
          // and right face-state elements.
          if (order == 8) {
            Reconstruct<Reconstruction::weno9js, X3DIR>(member, k, j, il,
                                                         iu, cons, ql, qr);
          } else if (order == 6) {
            Reconstruct<Reconstruction::weno7js, X3DIR>(member, k, j, il,
                                                         iu, cons, ql, qr);
          } else {
            Reconstruct<Reconstruction::weno5js, X3DIR>(member, k, j, il,
                                                         iu, cons, ql, qr);
          }
          member.team_barrier();

          parthenon::par_for_inner(
              member, 0, num_scratch_vars - 1, il, iu,
              [&](const int n, const int i) {
                if (k <= kb.e) {
                  const int kf = k + 1;
                  face_avg_left(TE::F3, n, kf, j, i) =
                      (n == IB3) ? Bface(TE::F3, 0, kf, j, i) : ql(n, i);
                }
                if (k >= kb.s) {
                  face_avg_right(TE::F3, n, k, j, i) =
                      (n == IB3) ? Bface(TE::F3, 0, k, j, i) : qr(n, i);
                }
              });
        });
  }


  return TaskStatus::complete;
}

// convert face averages to point-values
// Riemann solve with those point values
template <int ORDER, int DIR, typename FacePack, typename FluxPack, typename AuxPack,
          typename VbartPack, typename BPointPack>
KOKKOS_INLINE_FUNCTION void CalculateHLLDPointFace(
    const FacePack &face_avg_left, const FacePack &face_avg_right,
    FluxPack &consflux_point, AuxPack &uct_hlld, VbartPack &vbart,
    BPointPack &Bface_point, const AdiabaticCTMHDEOS &eos, const int nhydro,
    const int nscalars, const int ndim, const int k, const int j, const int i) {
  constexpr auto face =
      DIR == X1DIR ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);

  UCTMHDPointConserved u =
      LoadPointConserved<ORDER, DIR>(face_avg_left, ndim, k, j, i);
  const Real bnormal = DIR == X1DIR ? u.b1 : (DIR == X2DIR ? u.b2 : u.b3);
  UCTMHDPointPrimitive wl;
  UCTMHDConsToPrimPoint(u, wl, eos);

  u = LoadPointConserved<ORDER, DIR>(face_avg_right, ndim, k, j, i);
  if constexpr (DIR == X1DIR) {
    u.b1 = bnormal;
  } else if constexpr (DIR == X2DIR) {
    u.b2 = bnormal;
  } else {
    u.b3 = bnormal;
  }
  UCTMHDPointPrimitive wr;
  UCTMHDConsToPrimPoint(u, wr, eos);

  const auto result =
      Riemann<Fluid::ucthlldmhd, RiemannSolver::hlld>::SolvePointFace<DIR>(
          wl, wr, eos);
  consflux_point(face, IDN, k, j, i) = result.flux.d;
  consflux_point(face, IM1, k, j, i) = result.flux.m1;
  consflux_point(face, IM2, k, j, i) = result.flux.m2;
  consflux_point(face, IM3, k, j, i) = result.flux.m3;
  consflux_point(face, IEN, k, j, i) = result.flux.e;
  consflux_point(face, IB1, k, j, i) = result.flux.b1;
  consflux_point(face, IB2, k, j, i) = result.flux.b2;
  consflux_point(face, IB3, k, j, i) = result.flux.b3;
  Bface_point(face, 0, k, j, i) = bnormal;

  uct_hlld(face, AL, k, j, i) = result.aux.aL;
  uct_hlld(face, AR, k, j, i) = result.aux.aR;
  uct_hlld(face, DL, k, j, i) = result.aux.dL;
  uct_hlld(face, DR, k, j, i) = result.aux.dR;
  vbart(face, 0, k, j, i) = result.aux.vT1;
  vbart(face, 1, k, j, i) = result.aux.vT2;

  for (int n = nhydro; n < nhydro + nscalars; ++n) {
    const Real scalar_l = FaceAverageToPoint<ORDER, DIR>(face_avg_left(face, n), ndim,
                                                  k, j, i) /
                          wl.d;
    const Real scalar_r = FaceAverageToPoint<ORDER, DIR>(face_avg_right(face, n), ndim,
                                                  k, j, i) /
                          wr.d;
    consflux_point(face, n, k, j, i) =
        result.flux.d * ((result.flux.d >= 0.0) ? scalar_l : scalar_r);
  }
}

TaskStatus CalculatePointFluxesTeissierHLLDPoint(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto face_avg_left_pack =
      md->PackVariables(std::vector<std::string>{"face_avg_left"});
  auto face_avg_right_pack =
      md->PackVariables(std::vector<std::string>{"face_avg_right"});
  auto consflux_point_pack =
      md->PackVariables(std::vector<std::string>{"consflux_point"});
  auto uct_hlld_pack = md->PackVariables(std::vector<std::string>{"uct_hlld"});
  auto vbart_pack =
      md->PackVariables(std::vector<std::string>{"teissier_hlld_vbart"});
  auto Bface_point_pack =
      md->PackVariables(std::vector<std::string>{"Bface_point"});

  const auto pkg = pmb->packages.Get("Hydro");
  const int nhydro = pkg->Param<int>("nhydro");
  const int nscalars = pkg->Param<int>("nscalars");
  const auto &eos = pkg->Param<AdiabaticCTMHDEOS>("eos");
  const int block_end = face_avg_left_pack.GetDim(5) - 1;

  int jl = jb.s;
  int ju = jb.e;
  int kl = kb.s;
  int ku = kb.e;
  if (ndim > 1) {
    jl = jb.s - 1;
    ju = jb.e + 1;
  }
  if (ndim > 2) {
    kl = kb.s - 1;
    ku = kb.e + 1;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculatePointFluxesTeissierHLLDPoint x1",
      DevExecSpace(), 0, block_end, kl, ku, jl, ju, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &face_avg_left = face_avg_left_pack(b);
        const auto &face_avg_right = face_avg_right_pack(b);
        auto &consflux_point = consflux_point_pack(b);
        auto &uct_hlld = uct_hlld_pack(b);
        auto &vbart = vbart_pack(b);
        auto &Bface_point = Bface_point_pack(b);
        CalculateHLLDPointFace<4, X1DIR>(
            face_avg_left, face_avg_right, consflux_point, uct_hlld, vbart,
            Bface_point, eos, nhydro, nscalars, ndim, k, j, i);
      });

  if (ndim >= 2) {
    int x2_il = ib.s - 1;
    int x2_iu = ib.e + 1;
    int x2_kl = kb.s;
    int x2_ku = kb.e;
    if (ndim > 2) {
      x2_kl = kb.s - 1;
      x2_ku = kb.e + 1;
    }
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CalculatePointFluxesTeissierHLLDPoint x2",
        DevExecSpace(), 0, block_end, x2_kl, x2_ku, jb.s, jb.e + 1, x2_il,
        x2_iu,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          auto &consflux_point = consflux_point_pack(b);
          auto &uct_hlld = uct_hlld_pack(b);
          auto &vbart = vbart_pack(b);
          auto &Bface_point = Bface_point_pack(b);
          CalculateHLLDPointFace<4, X2DIR>(
              face_avg_left, face_avg_right, consflux_point, uct_hlld, vbart,
              Bface_point, eos, nhydro, nscalars, ndim, k, j, i);
        });
  }

  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CalculatePointFluxesTeissierHLLDPoint x3",
        DevExecSpace(), 0, block_end, kb.s, kb.e + 1, jb.s - 1, jb.e + 1,
        ib.s - 1, ib.e + 1,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          auto &consflux_point = consflux_point_pack(b);
          auto &uct_hlld = uct_hlld_pack(b);
          auto &vbart = vbart_pack(b);
          auto &Bface_point = Bface_point_pack(b);
          CalculateHLLDPointFace<4, X3DIR>(
              face_avg_left, face_avg_right, consflux_point, uct_hlld, vbart,
              Bface_point, eos, nhydro, nscalars, ndim, k, j, i);
        });
  }

  return TaskStatus::complete;
}

// convert face averages to point-values
// Riemann solve with those point values
template <int ORDER, int DIR, typename FacePack, typename FluxPack, typename AuxPack,
          typename VbartBPack>
KOKKOS_INLINE_FUNCTION void CalculateHLLDLineFace(
    const FacePack &face_avg_left, const FacePack &face_avg_right,
    FluxPack &consflux_point, AuxPack &uct_hlld, VbartBPack &vbartB,
    const AdiabaticCTMHDEOS &eos, const int nhydro,
    const int nscalars, const int ndim, const int k, const int j, const int i) {
  constexpr auto face =
      DIR == X1DIR ? TE::F1 : (DIR == X2DIR ? TE::F2 : TE::F3);

  UCTMHDPointConserved u =
      LoadPointConserved<ORDER, DIR>(face_avg_left, ndim, k, j, i);
  const Real bnormal = DIR == X1DIR ? u.b1 : (DIR == X2DIR ? u.b2 : u.b3);
  UCTMHDPointPrimitive wl;
  UCTMHDConsToPrimPoint(u, wl, eos);

  u = LoadPointConserved<ORDER, DIR>(face_avg_right, ndim, k, j, i);
  if constexpr (DIR == X1DIR) {
    u.b1 = bnormal;
  } else if constexpr (DIR == X2DIR) {
    u.b2 = bnormal;
  } else {
    u.b3 = bnormal;
  }
  UCTMHDPointPrimitive wr;
  UCTMHDConsToPrimPoint(u, wr, eos);

  const auto result =
      Riemann<Fluid::ucthlldmhd, RiemannSolver::hlld>::SolvePointFace<DIR>(
          wl, wr, eos);
  consflux_point(face, IDN, k, j, i) = result.flux.d;
  consflux_point(face, IM1, k, j, i) = result.flux.m1;
  consflux_point(face, IM2, k, j, i) = result.flux.m2;
  consflux_point(face, IM3, k, j, i) = result.flux.m3;
  consflux_point(face, IEN, k, j, i) = result.flux.e;
  consflux_point(face, IB1, k, j, i) = result.flux.b1;
  consflux_point(face, IB2, k, j, i) = result.flux.b2;
  consflux_point(face, IB3, k, j, i) = result.flux.b3;

  uct_hlld(face, AL, k, j, i) = result.aux.aL;
  uct_hlld(face, AR, k, j, i) = result.aux.aR;
  uct_hlld(face, DL, k, j, i) = result.aux.dL;
  uct_hlld(face, DR, k, j, i) = result.aux.dR;
  vbartB(face, 0, k, j, i) = result.aux.vT1 * bnormal;
  vbartB(face, 1, k, j, i) = result.aux.vT2 * bnormal;

  for (int n = nhydro; n < nhydro + nscalars; ++n) {
    const Real scalar_l = FaceAverageToPoint<ORDER, DIR>(face_avg_left(face, n), ndim,
                                                  k, j, i) /
                          wl.d;
    const Real scalar_r = FaceAverageToPoint<ORDER, DIR>(face_avg_right(face, n), ndim,
                                                  k, j, i) /
                          wr.d;
    consflux_point(face, n, k, j, i) =
        result.flux.d * ((result.flux.d >= 0.0) ? scalar_l : scalar_r);
  }
}

template <int ORDER>
TaskStatus CalculatePointFluxesTeissierHLLDLine(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto face_avg_left_pack =
      md->PackVariables(std::vector<std::string>{"face_avg_left"});
  auto face_avg_right_pack =
      md->PackVariables(std::vector<std::string>{"face_avg_right"});
  auto consflux_point_pack =
      md->PackVariables(std::vector<std::string>{"consflux_point"});
  auto uct_hlld_pack = md->PackVariables(std::vector<std::string>{"uct_hlld"});
  auto vbartB_pack =
      md->PackVariables(std::vector<std::string>{"teissier_hlld_vbartB_point"});

  const auto pkg = pmb->packages.Get("Hydro");
  constexpr int point_halo = ORDER / 2 - 1;
  const int nhydro = pkg->Param<int>("nhydro");
  const int nscalars = pkg->Param<int>("nscalars");
  const auto &eos = pkg->Param<AdiabaticCTMHDEOS>("eos");
  const int block_end = face_avg_left_pack.GetDim(5) - 1;

  int jl = jb.s;
  int ju = jb.e;
  int kl = kb.s;
  int ku = kb.e;
  if (ndim > 1) {
    jl = jb.s - point_halo;
    ju = jb.e + point_halo;
  }
  if (ndim > 2) {
    kl = kb.s - point_halo;
    ku = kb.e + point_halo;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "CalculatePointFluxesTeissierHLLDLine x1",
      DevExecSpace(), 0, block_end, kl, ku, jl, ju, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &face_avg_left = face_avg_left_pack(b);
        const auto &face_avg_right = face_avg_right_pack(b);
        auto &consflux_point = consflux_point_pack(b);
        auto &uct_hlld = uct_hlld_pack(b);
        auto &vbartB = vbartB_pack(b);
        CalculateHLLDLineFace<ORDER, X1DIR>(
            face_avg_left, face_avg_right, consflux_point, uct_hlld, vbartB,
            eos, nhydro, nscalars, ndim, k, j, i);
      });

  if (ndim >= 2) {
    int x2_il = ib.s - point_halo;
    int x2_iu = ib.e + point_halo;
    int x2_kl = kb.s;
    int x2_ku = kb.e;
    if (ndim > 2) {
      x2_kl = kb.s - point_halo;
      x2_ku = kb.e + point_halo;
    }
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CalculatePointFluxesTeissierHLLDLine x2",
        DevExecSpace(), 0, block_end, x2_kl, x2_ku, jb.s, jb.e + 1, x2_il,
        x2_iu,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          auto &consflux_point = consflux_point_pack(b);
          auto &uct_hlld = uct_hlld_pack(b);
          auto &vbartB = vbartB_pack(b);
          CalculateHLLDLineFace<ORDER, X2DIR>(
              face_avg_left, face_avg_right, consflux_point, uct_hlld, vbartB,
              eos, nhydro, nscalars, ndim, k, j, i);
        });
  }

  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "CalculatePointFluxesTeissierHLLDLine x3",
        DevExecSpace(), 0, block_end, kb.s, kb.e + 1,
        jb.s - point_halo, jb.e + point_halo,
        ib.s - point_halo, ib.e + point_halo,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          auto &consflux_point = consflux_point_pack(b);
          auto &uct_hlld = uct_hlld_pack(b);
          auto &vbartB = vbartB_pack(b);
          CalculateHLLDLineFace<ORDER, X3DIR>(
              face_avg_left, face_avg_right, consflux_point, uct_hlld, vbartB,
              eos, nhydro, nscalars, ndim, k, j, i);
        });
  }

  return TaskStatus::complete;
}

template <int ORDER>
TaskStatus CalculatePointFluxesTeissierLLF(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  const IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  const IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  const IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);

  auto face_avg_left_pack =
      md->PackVariables(std::vector<std::string>{"face_avg_left"});
  auto face_avg_right_pack =
      md->PackVariables(std::vector<std::string>{"face_avg_right"});
  auto consflux_point_pack =
      md->PackVariables(std::vector<std::string>{"consflux_point"});
  auto llf_speed_pack =
      md->PackVariables(std::vector<std::string>{"teissier_llf_speed"});
  auto eface_point_pack =
      md->PackVariables(std::vector<std::string>{"teissier_eface_point"});

  const auto pkg = pmb->packages.Get("Hydro");
  constexpr int point_halo = ORDER / 2 - 1;
  const int nhydro = pkg->Param<int>("nhydro");
  const int nscalars = pkg->Param<int>("nscalars");
  const auto &eos = pkg->Param<AdiabaticCTMHDEOS>("eos");
  const int block_end = face_avg_left_pack.GetDim(5) - 1;

  int jl = jb.s;
  int ju = jb.e;
  int kl = kb.s;
  int ku = kb.e;
  if (ndim > 1) {
    jl = jb.s - point_halo;
    ju = jb.e + point_halo;
  }
  if (ndim > 2) {
    kl = kb.s - point_halo;
    ku = kb.e + point_halo;
  }

  parthenon::par_for(
      DEFAULT_LOOP_PATTERN, "Teissier x1 thread-per-face AtoP + LLF",
      DevExecSpace(), 0, block_end, kl, ku, jl, ju, ib.s, ib.e + 1,
      KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
        const auto &face_avg_left = face_avg_left_pack(b);
        const auto &face_avg_right = face_avg_right_pack(b);
        auto &consflux_point = consflux_point_pack(b);
        auto &llf_speed = llf_speed_pack(b);
        auto &eface_point = eface_point_pack(b);
        CalculateLLFPointFace<ORDER, X1DIR>(face_avg_left, face_avg_right,
                                     consflux_point, llf_speed, eface_point,
                                     eos, nhydro, nscalars, ndim, k, j, i);
      });

  if (ndim >= 2) {
    int x2_il = ib.s - point_halo;
    int x2_iu = ib.e + point_halo;
    int x2_kl = kb.s;
    int x2_ku = kb.e;
    if (ndim > 2) {
      x2_kl = kb.s - point_halo;
      x2_ku = kb.e + point_halo;
    }
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier x2 thread-per-face AtoP + LLF",
        DevExecSpace(), 0, block_end, x2_kl, x2_ku, jb.s, jb.e + 1,
        x2_il, x2_iu,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          auto &consflux_point = consflux_point_pack(b);
          auto &llf_speed = llf_speed_pack(b);
          auto &eface_point = eface_point_pack(b);
          CalculateLLFPointFace<ORDER, X2DIR>(face_avg_left, face_avg_right,
                                       consflux_point, llf_speed, eface_point,
                                       eos, nhydro, nscalars, ndim, k, j, i);
        });
  }

  if (ndim >= 3) {
    parthenon::par_for(
        DEFAULT_LOOP_PATTERN, "Teissier x3 thread-per-face AtoP + LLF",
        DevExecSpace(), 0, block_end, kb.s, kb.e + 1,
        jb.s - point_halo, jb.e + point_halo,
        ib.s - point_halo, ib.e + point_halo,
        KOKKOS_LAMBDA(const int b, const int k, const int j, const int i) {
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          auto &consflux_point = consflux_point_pack(b);
          auto &llf_speed = llf_speed_pack(b);
          auto &eface_point = eface_point_pack(b);
          CalculateLLFPointFace<ORDER, X3DIR>(face_avg_left, face_avg_right,
                                       consflux_point, llf_speed, eface_point,
                                       eos, nhydro, nscalars, ndim, k, j, i);
        });
  }

  return TaskStatus::complete;
}

TaskStatus CalculatePointFluxesTeissier(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto pkg = pmb->packages.Get("Hydro");
  const auto &uct_solver = pkg->Param<std::string>("uct_solver");
  const int order = pkg->Param<int>("convergence_order");

  if (uct_solver == "hlld_point") {
    PARTHENON_REQUIRE(order == 4,
                      "Teissier hlld_point is only supported at fourth order.");
    return CalculatePointFluxesTeissierHLLDPoint(md);
  }
  if (uct_solver == "hlld_line") {
    if (order == 4) return CalculatePointFluxesTeissierHLLDLine<4>(md);
    if (order == 6) return CalculatePointFluxesTeissierHLLDLine<6>(md);
    if (order == 8) return CalculatePointFluxesTeissierHLLDLine<8>(md);
  }
  if (uct_solver == "llf_line") {
    if (order == 4) return CalculatePointFluxesTeissierLLF<4>(md);
    if (order == 6) return CalculatePointFluxesTeissierLLF<6>(md);
    if (order == 8) return CalculatePointFluxesTeissierLLF<8>(md);
  }
  PARTHENON_FAIL("Unsupported Teissier order/UCT solver in point-face calculation: " +
                 uct_solver);
}

} // namespace Hydro::HIGHORDER
