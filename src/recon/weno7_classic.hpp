//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef RECONSTRUCT_WENO7_CLASSIC_HPP_
#define RECONSTRUCT_WENO7_CLASSIC_HPP_
//! \file weno7_classic.hpp
//  \brief WENO-7 reconstruction for a Cartesian-like coordinate with uniform spacing.
//
// REFERENCES:
// Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
// non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)

#include <algorithm> // max()
#include <math.h>

#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \fn WENO7()
//  \brief Reconstructs 7th-order polynomial in cell i to compute ql(i+1) and qr(i).
//  Works for any dimension by passing in the appropriate q_im3,...,q _ip3.

KOKKOS_INLINE_FUNCTION
void WENO7JS(const Real &q_im3, const Real &q_im2, const Real &q_im1, const Real &q_i, const Real &q_ip1,
           const Real &q_ip2, const Real &q_ip3, Real &ql_ip1, Real &qr_i) {
  // taken from Carolyn Wendeln's Spartha code
  const Real epsilon = 1E-6;

  Real beta[4]; // flipped from MATLAB code where beta[0] is beta_3
  // beta[0] = (15943*v_i*v_i)/960 - (36709*v_i*v_ip1)/480 + (28429*v_i*v_ip2)/480 - (7663*v_i*v_ip3)/480 + (86447*v_ip1*v_ip1)/960 - (68407*v_ip1*v_ip2)/480 + (6223*v_ip1*v_ip3)/160 + (55247*v_ip2*v_ip2)/960 - (15269*v_ip2*v_ip3)/480 + (1421*v_ip3*v_ip3)/320;
  // beta[1] = (22847*v_i*v_i)/960 - (9389*v_i*v_im1)/480  - (18367*v_i*v_ip1)/480 + (4909*v_i*v_ip2)/480 + (1421*v_im1*v_im1)/320  + (2303*v_im1*v_ip1)/160  - (1783*v_im1*v_ip2)/480 + (15887*v_ip1*v_ip1)/960 - (4429*v_ip1*v_ip2)/480  + (1303*v_ip2*v_ip2)/960;
  // beta[2] = (10847*v_i*v_i)/960 - (10327*v_i*v_im1)/480 + (2909*v_i*v_im2)/480  - (1143*v_i*v_ip1)/160 + (10847*v_im1*v_im1)/960 - (1143*v_im1*v_im2)/160  + (2909*v_im1*v_ip1)/480 + (1303*v_im2*v_im2)/960  - (261*v_im2*v_ip1)/160   + (1303*v_ip1*v_ip1)/960;
  // beta[3] = (1421*v_i*v_i)/320  - (9389*v_i*v_im1)/480  + (2303*v_i*v_im2)/160  - (1783*v_i*v_im3)/480 + (22847*v_im1*v_im1)/960 - (18367*v_im1*v_im2)/480 + (4909*v_im1*v_im3)/480 + (15887*v_im2*v_im2)/960 - (4429*v_im2*v_im3)/480  + (1303*v_im3*v_im3)/960;

  // Balsara and Shu (III.a.)
  beta[0] =   q_i*(2107*q_i   - 9402*q_ip1 + 7042*q_ip2 - 1854*q_ip3 ) + q_ip1*(11003*q_ip1 - 17246*q_ip2 + 4642*q_ip3 ) + q_ip2*( 7043*q_ip2 - 3882*q_ip3 ) +  547*q_ip3*q_ip3;
  beta[1] = q_im1*( 547*q_im1 - 2522*q_i   + 1922*q_ip1 -  494*q_ip2 ) +   q_i*( 3443*q_i   -  5966*q_ip1 + 1602*q_ip2 ) + q_ip1*( 2843*q_ip1 - 1642*q_ip2 ) +  267*q_ip2*q_ip2;
  beta[2] = q_im2*( 267*q_im2 - 1642*q_im1 + 1602*q_i   -  494*q_ip1 ) + q_im1*( 2843*q_im1 -  5966*q_i   + 1922*q_ip1 ) +   q_i*( 3443*q_i   - 2522*q_ip1 ) +  547*q_ip1*q_ip1;
  beta[3] = q_im3*( 547*q_im3 - 3882*q_im2 + 4642*q_im1 - 1854*q_i   ) + q_im2*( 7043*q_im2 - 17246*q_im1 + 7042*q_i   ) + q_im1*(11003*q_im1 - 9402*q_i   ) + 2107*q_i*q_i;

  Real indicator[4]; // fraction part of (2.59) 
  indicator[0] = 1 / SQR(epsilon + beta[0]);
  indicator[1] = 1 / SQR(epsilon + beta[1]);
  indicator[2] = 1 / SQR(epsilon + beta[2]);
  indicator[3] = 1 / SQR(epsilon + beta[3]);

  // indicator[0] = 1.0 / std::pow(epsilon + beta[0], 4);
  // indicator[1] = 1.0 / std::pow(epsilon + beta[1], 4);
  // indicator[2] = 1.0 / std::pow(epsilon + beta[2], 4);
  // indicator[3] = 1.0 / std::pow(epsilon + beta[3], 4);

  // compute qL_ip1
  Real f[4]; // polynomial based on constants c_{r,j} in Table 2.1 
  // Factor of 1/12 in coefficients of f[] array applied to alpha_sum to reduce divisions
  f[0] =  3*q_i   + 13*q_ip1 -  5*q_ip2 +   q_ip3;
  f[1] =   -q_im1 +  7*q_i   +  7*q_ip1 -   q_ip2;
  f[2] =    q_im2 -  5*q_im1 + 13*q_i   + 3*q_ip1;
  f[3] = -3*q_im3 + 13*q_im2 - 23*q_im1 + 25*q_i;

  Real alpha[4]; // (2.59)
  alpha[0] = indicator[0] *  4.0 / 35.0;
  alpha[1] = indicator[1] * 18.0 / 35.0;
  alpha[2] = indicator[2] * 12.0 / 35.0;
  alpha[3] = indicator[3] *  1.0 / 35.0;   

  Real alpha_sum = 12.0 * (alpha[0] + alpha[1] + alpha[2] + alpha[3]);

  ql_ip1 = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2] + alpha[3] * f[3]) / alpha_sum; // (2.52) 

  // compute qR_i
  f[0] =  3*q_i   + 13*q_im1 -  5*q_im2 +   q_im3;
  f[1] =   -q_ip1 +  7*q_i   +  7*q_im1 -   q_im2;
  f[2] =    q_ip2 -  5*q_ip1 + 13*q_i   + 3*q_im1;
  f[3] = -3*q_ip3 + 13*q_ip2 - 23*q_ip1 + 25*q_i;

  alpha[0] = indicator[3] *  4.0 / 35.0;
  alpha[1] = indicator[2] * 18.0 / 35.0;
  alpha[2] = indicator[1] * 12.0 / 35.0;
  alpha[3] = indicator[0] *  1.0 / 35.0;

  alpha_sum = 12.0 * (alpha[0] + alpha[1] + alpha[2] + alpha[3]);

  qr_i = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2] + alpha[3] * f[3]) / alpha_sum; // (2.52) 

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
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::weno7js, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        // ql is ql_ip1 and qr is qr_i
        WENO7JS(q(n, k, j, i - 3), q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
              q(n, k, j, i + 2), q(n, k, j, i + 3), ql(n, i + 1), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        // ql is ql_jp1 and qr is qr_j
        WENO7JS(q(n, k, j - 3, i), q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
              q(n, k, j + 2, i), q(n, k, j + 3, i), ql(n, i), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        // ql is ql_kp1 and qr is qr_k
        WENO7JS(q(n, k - 3, j, i), q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
              q(n, k + 2, j, i), q(n, k + 3, j, i), ql(n, i), qr(n, i));
      } else {
        PARTHENON_FAIL("Unknow direction for WENO7 reconstruction.")
      }
    });
  }
}

#endif // RECONSTRUCT_WENO7_CLASSIC_HPP_
