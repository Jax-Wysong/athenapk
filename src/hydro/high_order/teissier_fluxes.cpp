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
#include "../rsolvers/rsolvers.hpp"
#include "teissier_fluxes.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;


namespace Hydro::HIGHORDER {

struct Scratch4DAdapter {
  parthenon::ScratchPad2D<Real> data;

  KOKKOS_INLINE_FUNCTION
  Real &operator()(const int n, const int /*k*/, const int /*j*/,
                   const int i) const {
    return data(n, i);
  }
};


TaskStatus ReconstructFaceAveragesTeissier(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int il, iu, jl, ju, kl, ku;
  jl = jb.s, ju = jb.e, kl = kb.s, ku = kb.e;
 
  // Face AtoP needs one transverse neighbor. The wider pointwise edge stencil is
  // supplied later by exchanging the completed point-face fields.
    if (pmb->block_size.nx(X2DIR) > 1) {
      if (pmb->block_size.nx(X3DIR) == 1) // 2D
        jl = jb.s - 1, ju = jb.e + 1, kl = kb.s, ku = kb.e;
      else // 3D
        jl = jb.s - 1, ju = jb.e + 1, kl = kb.s - 1, ku = kb.e + 1;
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

  auto pkg = pmb->packages.Get("Hydro");
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
        Reconstruct<Reconstruction::weno5js, X1DIR>(member, k, j, ib.s - 1, ib.e + 1, cons, wl, wr);
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
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 3;
    // set the loop limits
    // One transverse layer is sufficient for the subsequent face AtoP.
    il = ib.s - 1, iu = ib.e + 1, kl = kb.s, ku = kb.e;
    if (pmb->block_size.nx(X3DIR) == 1) // 2D
    kl = kb.s, ku = kb.e;
    else // 3D
    kl = kb.s - 1, ku = kb.e + 1;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Teissier x2 face reconstruction", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, kl, ku,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
        const auto &cons = cons_in(b);
        const auto &Bface = Bface_pack(b);
        auto &face_avg_left  = face_avg_left_pack(b);
        auto &face_avg_right = face_avg_right_pack(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          for (int j = jb.s - 1; j <= jb.e + 1; ++j) {
            // reconstruct L/R states at j
            Reconstruct<Reconstruction::weno5js, X2DIR>(member, k, j, il, iu, cons, wlb, wr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (j > jb.s - 1) {
                parthenon::par_for_inner(member, il, iu, [&](const int i) {
                  wl(IB2, i) = Bface(TE::F2, 0, k, j, i);
                  wr(IB2, i) = Bface(TE::F2, 0, k, j, i);
                });
                member.team_barrier();
                // need to loop over n and i
                parthenon::par_for_inner(member, 0, num_scratch_vars-1, il, iu, [&](const int n, const int i) {
                    face_avg_left(TE::F2, n, k, j, i)  = wl(n, i);
                    face_avg_right(TE::F2, n, k, j, i) = wr(n, i);
                });
                member.team_barrier();
            }
            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
          }
        });
  }
  //--------------------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->ndim >= 3) {
    // set the loop limits
      il = ib.s - 1, iu = ib.e + 1, jl = jb.s - 1, ju = jb.e + 1;


    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Teissier x3 face reconstruction", DevExecSpace(), scratch_size_in_bytes,
        scratch_level, 0, cons_in.GetDim(5) - 1, jl, ju,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int j) {
        const auto &cons = cons_in(b);
        const auto &Bface = Bface_pack(b);
        auto &face_avg_left  = face_avg_left_pack(b);
        auto &face_avg_right = face_avg_right_pack(b);
          parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                           num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wlb(member.team_scratch(scratch_level),
                                            num_scratch_vars, nx1);
          for (int k = kb.s - 1; k <= kb.e + 1; ++k) {
            // reconstruct L/R states at j
            Reconstruct<Reconstruction::weno5js, X3DIR>(member, k, j, il, iu, cons, wlb, wr);
            // Sync all threads in the team so that scratch memory is consistent
            member.team_barrier();

            if (k > kb.s - 1) {
                parthenon::par_for_inner(member, il, iu, [&](const int i) {
                  wl(IB3, i) = Bface(TE::F3, 0, k, j, i);
                  wr(IB3, i) = Bface(TE::F3, 0, k, j, i);
                });
                member.team_barrier();
                // need to loop over n and i
                parthenon::par_for_inner(member, 0, num_scratch_vars-1, il, iu, [&](const int n, const int i) {
                    face_avg_left(TE::F3, n, k, j, i)  = wl(n, i);
                    face_avg_right(TE::F3, n, k, j, i) = wr(n, i);
                });
                member.team_barrier();
            }
            // swap the arrays for the next step
            auto *tmp = wl.data();
            wl.assign_data(wlb.data());
            wlb.assign_data(tmp);
          }
        });
  }


  return TaskStatus::complete;
}

// convert face averages to point-values
// Riemann solve with those point values
template <RiemannSolver solver>
TaskStatus CalculatePointFluxesTeissierImpl(MeshData<Real> *md) {
  static_assert(solver == RiemannSolver::hlld || solver == RiemannSolver::llf,
                "The Teissier point-face task supports only HLLD and LLF");
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const int ndim = pmb->pmy_mesh->ndim;
  IndexRange ib = pmb->cellbounds.GetBoundsI(IndexDomain::interior);
  IndexRange jb = pmb->cellbounds.GetBoundsJ(IndexDomain::interior);
  IndexRange kb = pmb->cellbounds.GetBoundsK(IndexDomain::interior);
  int il, iu;

  auto cons_pack =
      md->PackVariablesAndFluxes(std::vector<std::string>{"cons"});
  MeshBlockPack<VariablePack<Real>> uct_hlld_pack;
  MeshBlockPack<VariablePack<Real>> face_avg_left_pack;
  MeshBlockPack<VariablePack<Real>> face_avg_right_pack;
  MeshBlockPack<VariablePack<Real>> consflux_point_pack;
  MeshBlockPack<VariablePack<Real>> Bface_point_pack;
  MeshBlockPack<VariablePack<Real>> teissier_llf_speed_pack;
  MeshBlockPack<VariablePack<Real>> teissier_eface_point_pack;

  auto pkg = pmb->packages.Get("Hydro");
  const auto nhydro = pkg->Param<int>("nhydro");
  const auto nscalars = pkg->Param<int>("nscalars");
  face_avg_left_pack =
        md->PackVariables(std::vector<std::string>{"face_avg_left"});
  face_avg_right_pack =
        md->PackVariables(std::vector<std::string>{"face_avg_right"});
  consflux_point_pack =
        md->PackVariables(std::vector<std::string>{"consflux_point"});
  if constexpr (solver == RiemannSolver::hlld) {
    uct_hlld_pack =
        md->PackVariables(std::vector<std::string>{"uct_hlld"});
    Bface_point_pack =
        md->PackVariables(std::vector<std::string>{"Bface_point"});
  } else {
    teissier_llf_speed_pack =
        md->PackVariables(std::vector<std::string>{"teissier_llf_speed"});
    teissier_eface_point_pack =
        md->PackVariables(std::vector<std::string>{"teissier_eface_point"});
  }


  auto num_scratch_vars = nhydro + nscalars;

  const auto &eos = pkg->Param<AdiabaticCTMHDEOS>("eos");
  auto riemann = Riemann<Fluid::ucthlldmhd, solver>();
  constexpr Real c_h = 0.0;

  const int scratch_level =
      pkg->Param<int>("scratch_level"); // 0 is actual scratch (tiny); 1 is HBM
  const int nx1 = pmb->cellbounds.ncellsi(IndexDomain::entire);

  size_t scratch_size_in_bytes =
      parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 4;

  parthenon::par_for_outer(
      DEFAULT_OUTER_LOOP_PATTERN, "Teissier x1 AtoP + Riemann Solver", DevExecSpace(), scratch_size_in_bytes,
      scratch_level, 0, cons_pack.GetDim(5) - 1, kb.s, kb.e, jb.s, jb.e,
      KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k, const int j) {
        // NVCC does not allow an extended device lambda's first implicit capture
        // to occur inside an if constexpr branch. Force shallow captures here;
        // unused solver-specific packs are never dereferenced.
        (void)Bface_point_pack;
        (void)uct_hlld_pack;
        (void)teissier_llf_speed_pack;
        (void)teissier_eface_point_pack;
        (void)riemann;
        (void)eos;
        (void)c_h;
        auto &consflux_point = consflux_point_pack(b);
        const auto &face_avg_left  = face_avg_left_pack(b);
        const auto &face_avg_right = face_avg_right_pack(b);
        parthenon::ScratchPad2D<Real> wl(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> wr(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> pointL(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        parthenon::ScratchPad2D<Real> pointR(member.team_scratch(scratch_level),
                                         num_scratch_vars, nx1);
        // step 1:
        // convert L/R ijk averages to points
        // step 2:
        // point cons->prim for wl and wr
        // step 3:
        // riemann solve 'em
        
        // step 1
        // qij = (7/6) <qij> - (1/24) (<q_{i+1, j}> + <q_{i-1,j}> + <q_{i, j+1}> + <q_{i,j-1}>)
        parthenon::par_for_inner(member, 0, num_scratch_vars-1, ib.s, ib.e+1, [&](const int n, const int i) {
        auto ql = face_avg_left(TE::F1, n);
        auto qr = face_avg_right(TE::F1, n);
        
        if (ndim == 1){
            pointL(n, i) = ql(k, j, i);
            pointR(n, i) = qr(k, j, i);    
        }
        else if (ndim == 2){ // Teissier24 (A.30)
            pointR(n, i) = (13.0/12.0) * qr(k,j,i) - (1.0/24.0) * (qr(k,j+1,i) + qr(k,j-1,i));
            pointL(n, i) = (13.0/12.0) * ql(k,j,i) - (1.0/24.0) * (ql(k,j+1,i) + ql(k,j-1,i));
        } else if (ndim == 3) { // Teissier24 (33)
            pointR(n, i) = (7.0/6.0) * qr(k,j,i) - (1.0/24.0) * (qr(k+1,j,i) + qr(k-1,j,i) + qr(k,j+1,i) + qr(k,j-1,i));
            pointL(n, i) = (7.0/6.0) * ql(k,j,i) - (1.0/24.0) * (ql(k+1,j,i) + ql(k-1,j,i) + ql(k,j+1,i) + ql(k,j-1,i));
        }
        });
        member.team_barrier();

        // The normal field is single-valued at a CT face. Preserve that property
        // explicitly after the face-average-to-point conversion and retain the point
        // value for the edge UCT reconstruction.
        parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
          const Real bnormal = pointL(IB1, i);
          pointL(IB1, i) = bnormal;
          pointR(IB1, i) = bnormal;
          if constexpr (solver == RiemannSolver::hlld) {
            auto &Bface_point = Bface_point_pack(b);
            Bface_point(TE::F1, 0, k, j, i) = bnormal;
          }
        });
        member.team_barrier();

        // step 2
        Scratch4DAdapter pointqL4{pointL};
        Scratch4DAdapter pointqR4{pointR};
        Scratch4DAdapter wl4{wl};
        Scratch4DAdapter wr4{wr};

        parthenon::par_for_inner(
            member, ib.s, ib.e + 1,
            [&](const int i) {
            eos.ConsToPrim(pointqL4, wl4, nhydro, nscalars, 0, 0, i);
            eos.ConsToPrim(pointqR4, wr4, nhydro, nscalars, 0, 0, i);
            });

        member.team_barrier();
        
        // step 3:
        if constexpr (solver == RiemannSolver::hlld) {
          auto &uct_hlld = uct_hlld_pack(b);
          riemann.template Solve<true>(member, k, j, ib.s, ib.e + 1, IV1, wl, wr,
                                      consflux_point, uct_hlld, eos, c_h);
        } else {
          auto &llf_speed = teissier_llf_speed_pack(b);
          auto &eface_point = teissier_eface_point_pack(b);
          riemann.template Solve<true>(member, k, j, ib.s, ib.e + 1, IV1, wl, wr,
                                       consflux_point, llf_speed, eos, c_h);
          member.team_barrier();
          parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            // The two tangential electric fields on an x1 face are Ey and Ez.
            eface_point(TE::F1, ELEFT_T1, k, j, i) =
                wl(IV1, i) * wl(IB3, i) - wl(IV3, i) * wl(IB1, i);
            eface_point(TE::F1, ELEFT_T2, k, j, i) =
                wl(IV2, i) * wl(IB1, i) - wl(IV1, i) * wl(IB2, i);
            eface_point(TE::F1, ERIGHT_T1, k, j, i) =
                wr(IV1, i) * wr(IB3, i) - wr(IV3, i) * wr(IB1, i);
            eface_point(TE::F1, ERIGHT_T2, k, j, i) =
                wr(IV2, i) * wr(IB1, i) - wr(IV1, i) * wr(IB2, i);
          });
        }

        member.team_barrier();

        // Passive scalars are advected with the point-valued mass flux. They
        // must remain in point-flux storage until the common PtoA operation.
        for (auto n = nhydro; n < nhydro + nscalars; ++n) {
          parthenon::par_for_inner(member, ib.s, ib.e + 1, [&](const int i) {
            const Real mass_flux =
                consflux_point(TE::F1, IDN, k, j, i);
            consflux_point(TE::F1, n, k, j, i) =
                mass_flux * ((mass_flux >= 0.0) ? wl(n, i) : wr(n, i));
          });
        }
        member.team_barrier();

      });

  //--------------------------------------------------------------------------------------
  // j-direction
  if (pmb->pmy_mesh->ndim >= 2) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 4;
    il = ib.s, iu = ib.e;

    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Teissier x2 AtoP + Riemann Solver",
        DevExecSpace(), scratch_size_in_bytes, scratch_level, 0,
        cons_pack.GetDim(5) - 1, kb.s, kb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int k) {
          (void)Bface_point_pack;
          (void)uct_hlld_pack;
          (void)teissier_llf_speed_pack;
          (void)teissier_eface_point_pack;
          (void)riemann;
          (void)eos;
          (void)c_h;
          auto &consflux_point = consflux_point_pack(b);
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          parthenon::ScratchPad2D<Real> wl(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> pointL(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> pointR(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          Scratch4DAdapter pointqL4{pointL};
          Scratch4DAdapter pointqR4{pointR};
          Scratch4DAdapter wl4{wl};
          Scratch4DAdapter wr4{wr};

          for (int j = jb.s; j <= jb.e + 1; ++j) {
            parthenon::par_for_inner(
                member, 0, num_scratch_vars - 1, il, iu,
                [&](const int n, const int i) {
                  const auto ql = face_avg_left(TE::F2, n);
                  const auto qr = face_avg_right(TE::F2, n);
                  if (ndim == 2) {
                    pointL(n, i) =
                        (13.0 / 12.0) * ql(k, j, i) -
                        (ql(k, j, i - 1) + ql(k, j, i + 1)) / 24.0;
                    pointR(n, i) =
                        (13.0 / 12.0) * qr(k, j, i) -
                        (qr(k, j, i - 1) + qr(k, j, i + 1)) / 24.0;
                  } else {
                    pointL(n, i) =
                        (7.0 / 6.0) * ql(k, j, i) -
                        (ql(k, j, i - 1) + ql(k, j, i + 1) +
                         ql(k - 1, j, i) + ql(k + 1, j, i)) /
                            24.0;
                    pointR(n, i) =
                        (7.0 / 6.0) * qr(k, j, i) -
                        (qr(k, j, i - 1) + qr(k, j, i + 1) +
                         qr(k - 1, j, i) + qr(k + 1, j, i)) /
                            24.0;
                  }
                });
            member.team_barrier();

            parthenon::par_for_inner(member, il, iu, [&](const int i) {
              const Real bnormal = pointL(IB2, i);
              pointL(IB2, i) = bnormal;
              pointR(IB2, i) = bnormal;
              if constexpr (solver == RiemannSolver::hlld) {
                auto &Bface_point = Bface_point_pack(b);
                Bface_point(TE::F2, 0, k, j, i) = bnormal;
              }
            });
            member.team_barrier();

            parthenon::par_for_inner(member, il, iu, [&](const int i) {
              eos.ConsToPrim(pointqL4, wl4, nhydro, nscalars, 0, 0, i);
              eos.ConsToPrim(pointqR4, wr4, nhydro, nscalars, 0, 0, i);
            });
            member.team_barrier();

            if constexpr (solver == RiemannSolver::hlld) {
              auto &uct_hlld = uct_hlld_pack(b);
              riemann.template Solve<true>(member, k, j, il, iu, IV2, wl, wr,
                                           consflux_point, uct_hlld, eos, c_h);
            } else {
              auto &llf_speed = teissier_llf_speed_pack(b);
              auto &eface_point = teissier_eface_point_pack(b);
              riemann.template Solve<true>(member, k, j, il, iu, IV2, wl, wr,
                                           consflux_point, llf_speed, eos, c_h);
              member.team_barrier();
              parthenon::par_for_inner(member, il, iu, [&](const int i) {
                // The two tangential electric fields on an x2 face are Ez and Ex.
                eface_point(TE::F2, ELEFT_T1, k, j, i) =
                    wl(IV2, i) * wl(IB1, i) - wl(IV1, i) * wl(IB2, i);
                eface_point(TE::F2, ELEFT_T2, k, j, i) =
                    wl(IV3, i) * wl(IB2, i) - wl(IV2, i) * wl(IB3, i);
                eface_point(TE::F2, ERIGHT_T1, k, j, i) =
                    wr(IV2, i) * wr(IB1, i) - wr(IV1, i) * wr(IB2, i);
                eface_point(TE::F2, ERIGHT_T2, k, j, i) =
                    wr(IV3, i) * wr(IB2, i) - wr(IV2, i) * wr(IB3, i);
              });
            }
            member.team_barrier();

            for (auto n = nhydro; n < nhydro + nscalars; ++n) {
              parthenon::par_for_inner(member, il, iu, [&](const int i) {
                const Real mass_flux =
                    consflux_point(TE::F2, IDN, k, j, i);
                consflux_point(TE::F2, n, k, j, i) =
                    mass_flux * ((mass_flux >= 0.0) ? wl(n, i) : wr(n, i));
              });
            }
            member.team_barrier();
          }
        });
  }
  //--------------------------------------------------------------------------------------
  // k-direction
  if (pmb->pmy_mesh->ndim >= 3) {
    scratch_size_in_bytes =
        parthenon::ScratchPad2D<Real>::shmem_size(num_scratch_vars, nx1) * 4;
    il = ib.s, iu = ib.e;
    parthenon::par_for_outer(
        DEFAULT_OUTER_LOOP_PATTERN, "Teissier x3 AtoP + Riemann Solver",
        DevExecSpace(), scratch_size_in_bytes, scratch_level, 0,
        cons_pack.GetDim(5) - 1, jb.s, jb.e,
        KOKKOS_LAMBDA(parthenon::team_mbr_t member, const int b, const int j) {
          (void)Bface_point_pack;
          (void)uct_hlld_pack;
          (void)teissier_llf_speed_pack;
          (void)teissier_eface_point_pack;
          (void)riemann;
          (void)eos;
          (void)c_h;
          auto &consflux_point = consflux_point_pack(b);
          const auto &face_avg_left = face_avg_left_pack(b);
          const auto &face_avg_right = face_avg_right_pack(b);
          parthenon::ScratchPad2D<Real> wl(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> wr(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> pointL(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          parthenon::ScratchPad2D<Real> pointR(
              member.team_scratch(scratch_level), num_scratch_vars, nx1);
          Scratch4DAdapter pointqL4{pointL};
          Scratch4DAdapter pointqR4{pointR};
          Scratch4DAdapter wl4{wl};
          Scratch4DAdapter wr4{wr};

          for (int k = kb.s; k <= kb.e + 1; ++k) {
            parthenon::par_for_inner(
                member, 0, num_scratch_vars - 1, il, iu,
                [&](const int n, const int i) {
                  const auto ql = face_avg_left(TE::F3, n);
                  const auto qr = face_avg_right(TE::F3, n);
                  pointL(n, i) =
                      (7.0 / 6.0) * ql(k, j, i) -
                      (ql(k, j, i - 1) + ql(k, j, i + 1) +
                       ql(k, j - 1, i) + ql(k, j + 1, i)) /
                          24.0;
                  pointR(n, i) =
                      (7.0 / 6.0) * qr(k, j, i) -
                      (qr(k, j, i - 1) + qr(k, j, i + 1) +
                       qr(k, j - 1, i) + qr(k, j + 1, i)) /
                          24.0;
                });
            member.team_barrier();

            parthenon::par_for_inner(member, il, iu, [&](const int i) {
              const Real bnormal = pointL(IB3, i);
              pointL(IB3, i) = bnormal;
              pointR(IB3, i) = bnormal;
              if constexpr (solver == RiemannSolver::hlld) {
                auto &Bface_point = Bface_point_pack(b);
                Bface_point(TE::F3, 0, k, j, i) = bnormal;
              }
            });
            member.team_barrier();

            parthenon::par_for_inner(member, il, iu, [&](const int i) {
              eos.ConsToPrim(pointqL4, wl4, nhydro, nscalars, 0, 0, i);
              eos.ConsToPrim(pointqR4, wr4, nhydro, nscalars, 0, 0, i);
            });
            member.team_barrier();

            if constexpr (solver == RiemannSolver::hlld) {
              auto &uct_hlld = uct_hlld_pack(b);
              riemann.template Solve<true>(member, k, j, il, iu, IV3, wl, wr,
                                           consflux_point, uct_hlld, eos, c_h);
            } else {
              auto &llf_speed = teissier_llf_speed_pack(b);
              auto &eface_point = teissier_eface_point_pack(b);
              riemann.template Solve<true>(member, k, j, il, iu, IV3, wl, wr,
                                           consflux_point, llf_speed, eos, c_h);
              member.team_barrier();
              parthenon::par_for_inner(member, il, iu, [&](const int i) {
                // The two tangential electric fields on an x3 face are Ex and Ey.
                eface_point(TE::F3, ELEFT_T1, k, j, i) =
                    wl(IV3, i) * wl(IB2, i) - wl(IV2, i) * wl(IB3, i);
                eface_point(TE::F3, ELEFT_T2, k, j, i) =
                    wl(IV1, i) * wl(IB3, i) - wl(IV3, i) * wl(IB1, i);
                eface_point(TE::F3, ERIGHT_T1, k, j, i) =
                    wr(IV3, i) * wr(IB2, i) - wr(IV2, i) * wr(IB3, i);
                eface_point(TE::F3, ERIGHT_T2, k, j, i) =
                    wr(IV1, i) * wr(IB3, i) - wr(IV3, i) * wr(IB1, i);
              });
            }
            member.team_barrier();

            for (auto n = nhydro; n < nhydro + nscalars; ++n) {
              parthenon::par_for_inner(member, il, iu, [&](const int i) {
                const Real mass_flux =
                    consflux_point(TE::F3, IDN, k, j, i);
                consflux_point(TE::F3, n, k, j, i) =
                    mass_flux * ((mass_flux >= 0.0) ? wl(n, i) : wr(n, i));
              });
            }
            member.team_barrier();
          }
        });
  }


  return TaskStatus::complete;
}

TaskStatus CalculatePointFluxesTeissier(MeshData<Real> *md) {
  auto pmb = md->GetBlockData(0)->GetBlockPointer();
  const auto pkg = pmb->packages.Get("Hydro");
  const auto &uct_solver = pkg->Param<std::string>("uct_solver");

  if (uct_solver == "hlld_point") {
    return CalculatePointFluxesTeissierImpl<RiemannSolver::hlld>(md);
  }
  if (uct_solver == "llf_line") {
    return CalculatePointFluxesTeissierImpl<RiemannSolver::llf>(md);
  }
  PARTHENON_FAIL("Unsupported Teissier UCT solver in point-face calculation: " +
                 uct_solver);
}

} // namespace Hydro::HIGHORDER
