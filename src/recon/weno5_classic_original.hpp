//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef RECONSTRUCT_WENO5_CLASSIC_HPP_
#define RECONSTRUCT_WENO5_CLASSIC_HPP_
//! \file weno5_classic.hpp
//  \brief WENO-5 reconstruction for a Cartesian-like coordinate with uniform spacing.
//
// REFERENCES:
// Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
// non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)

#include <algorithm> // max()
#include <math.h>

#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \fn WENO5()
//  \brief Reconstructs 5th-order polynomial in cell i to compute ql(i+1) and qr(i).
//  Works for any dimension by passing in the appropriate q_im2,...,q _ip2.

KOKKOS_INLINE_FUNCTION
void WENO5JS(const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, Real &ql_ip1, Real &qr_i) {
  // Smooth WENO weights: Note that these are from Del Zanna et al. 2007 (A.18)
  
  const Real epsilon = 1E-26;

  Real beta[3]; // (2.63) 
  beta[0] = (13.0/12.0)*SQR(q_i - 2*q_ip1 + q_ip2) + (1.0/4.0)*SQR(3*q_i - 4*q_ip1 + q_ip2);
  beta[1] = (13.0/12.0)*SQR(q_im1 - 2*q_i + q_ip1) + (1.0/4.0)*SQR(q_im1 - q_ip1);
  beta[2] = (13.0/12.0)*SQR(q_im2 - 2*q_im1 + q_i) + (1.0/4.0)*SQR(q_im2 - 4*q_im1 + 3*q_i);

  Real indicator[3]; // fraction part of (2.59) 
  indicator[0] = 1 / SQR(epsilon + beta[0]);
  indicator[1] = 1 / SQR(epsilon + beta[1]);
  indicator[2] = 1 / SQR(epsilon + beta[2]);

  // compute qL_ip1
  Real f[3]; // polynomial based on constants c_{r,j} in Table 2.1 
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] =  2*q_i   + 5*q_ip1 - q_ip2;
  f[1] = -1*q_im1 + 5*q_i   + 2*q_ip1;
  f[2] =  2*q_im2 - 7*q_im1 + 11*q_i;

  Real alpha[3];
  alpha[0] = 0.3 * (1.0 + SQR(indicator[0]));
  alpha[1] = 0.6 * (1.0 + SQR(indicator[1]));
  alpha[2] = 0.1 * (1.0 + SQR(indicator[2]));

  // ENO Weights
  // alpha[0] = 0.3;
  // alpha[1] = 0.6;
  // alpha[2] = 0.1;  

  Real alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

  ql_ip1 = (f[0] * alpha[0] + f[1] * alpha[1] + f[2] * alpha[2]) / alpha_sum;

  // compute qR_i
  // Factor of 1/6 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] =  2*q_i   + 5*q_im1 - q_im2;
  f[1] = -1*q_ip1 + 5*q_i   + 2*q_im1;
  f[2] =  2*q_ip2 - 7*q_ip1 + 11*q_i;

  alpha[0] = 0.3 * (1.0 + SQR(indicator[2]));
  alpha[1] = 0.6 * (1.0 + SQR(indicator[1]));
  alpha[2] = 0.1 * (1.0 + SQR(indicator[0]));

  alpha_sum = 6.0 * (alpha[0] + alpha[1] + alpha[2]);

  qr_i = (f[0] * alpha[0] + f[1] * alpha[1] + f[2] * alpha[2]) / alpha_sum;

}

//! \fn Reconstruct<Reconstruction::wenoz, int DIR>()
//  \brief Wrapper function for WENOZ reconstruction
//  In X1DIR call over [is-1,ie+1] to get BOTH L/R states over [is,ie]
//  In X2DIR call over [js-1,je+1] to get BOTH L/R states over [js,je]
//  In X3DIR call over [ks-1,ke+1] to get BOTH L/R states over [ks,ke]
//  Note that in the CalculateFlux function ql and qr contain stencils in i-direction that
//  have been cached for the appropriate k, j (and plus 1) values. Thus, in x1dir ql needs
//  to be offset by i+1 but for the other direction the offset has been set outside in the
//  cached stencil.
template <Reconstruction recon, int XNDIR>
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::weno5js, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        // ql is ql_ip1 and qr is qr_i
        WENO5JS(q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
              q(n, k, j, i + 2), ql(n, i + 1), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        // ql is ql_jp1 and qr is qr_j
        WENO5JS(q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
              q(n, k, j + 2, i), ql(n, i), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        // ql is ql_kp1 and qr is qr_k
        WENO5JS(q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
              q(n, k + 2, j, i), ql(n, i), qr(n, i));
      } else {
        PARTHENON_FAIL("Unknow direction for PPM reconstruction.")
      }
    });
  }
}

#endif // RECONSTRUCT_WENO5_CLASSIC_HPP_