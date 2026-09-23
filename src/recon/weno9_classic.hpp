//========================================================================================
// AthenaXXX astrophysical plasma code
// Copyright(C) 2020 James M. Stone <jmstone@ias.edu> and the Athena code team
// Licensed under the 3-clause BSD License (the "LICENSE")
//========================================================================================
#ifndef RECONSTRUCT_WENO9_CLASSIC_HPP_
#define RECONSTRUCT_WENO9_CLASSIC_HPP_
//! \file weno9_classic.hpp
//  \brief WENO-9 reconstruction for a Cartesian-like coordinate with uniform spacing.
//
// REFERENCES:
// Borges R., Carmona M., Costa B., Don W.S. , "An improved weighted essentially
// non-oscillatory scheme for hyperbolic conservation laws" , JCP, 227, 3191 (2008)

#include <algorithm> // max()
#include <math.h>

#include <parthenon/parthenon.hpp>

using parthenon::ScratchPad2D;

//----------------------------------------------------------------------------------------
//! \fn WENO9JS()
//  \brief Reconstructs 9th-order polynomial in cell i to compute ql(i+1) and qr(i).
//  Works for any dimension by passing in the appropriate q_im4,...,q_ip4.

KOKKOS_INLINE_FUNCTION
void WENO9JS(const Real &q_im4, const Real &q_im3, const Real &q_im2,
             const Real &q_im1, const Real &q_i, const Real &q_ip1,
             const Real &q_ip2, const Real &q_ip3, const Real &q_ip4,
             Real &ql_ip1, Real &qr_i) {
    
    const Real epsilon = 1E-6;

    Real beta[5]; // flipped from MATLAB code where beta[0] is beta_4
    // beta[0] = (3693653*v_i*v_i)/80640 - (2834627*v_i*v_ip1)/10080 + (2246389*v_i*v_ip2)/6720 - (1850819*v_i*v_ip3)/10080 + (1569797*v_i*v_ip4)/40320 + (8907527*v_ip1*v_ip1)/20160 - (3594209*v_ip1*v_ip2)/3360 + (5988821*v_ip1*v_ip3)/10080 - (639547*v_ip1*v_ip4)/5040  + (551713*v_ip2*v_ip2)/840   - (824853*v_ip2*v_ip3)/1120  + (1063739*v_ip2*v_ip4)/6720 + (4190927*v_ip3*v_ip3)/20160 - (226313*v_ip3*v_ip4)/2520  + (785153*v_ip4*v_ip4)/80640;  
    // beta[1] = (1906127*v_i*v_i)/20160 - (18406*v_i*v_im1)/315     - (776359*v_i*v_ip1)/3360  + (1281461*v_i*v_ip2)/10080 - (269519*v_i*v_ip3)/10080  + (785153*v_im1*v_im1)/80640  + (151953*v_im1*v_ip1)/2240  - (183637*v_im1*v_ip2)/5040   + (304757*v_im1*v_ip3)/40320 + (123103*v_ip1*v_ip1)/840   - (554809*v_ip1*v_ip2)/3360  + (78943*v_ip1*v_ip3)/2240   + (957767*v_ip2*v_ip2)/20160  - (207527*v_ip2*v_ip3)/10080 + (182453*v_ip3*v_ip3)/80640;
    // beta[2] = (22439*v_i*v_i)/420     - (84463*v_i*v_im1)/1120    + (124409*v_i*v_im2)/6720  - (214259*v_i*v_ip1)/3360   + (92839*v_i*v_ip2)/6720    + (574727*v_im1*v_im1)/20160  - (151877*v_im1*v_im2)/10080 + (426341*v_im1*v_ip1)/10080  - (2782*v_im1*v_ip2)/315     + (182453*v_im2*v_im2)/80640 - (100889*v_im2*v_ip1)/10080 + (82157*v_im2*v_ip2)/40320  + (410927*v_ip1*v_ip1)/20160  - (46801*v_ip1*v_ip2)/5040   + (91313*v_ip2*v_ip2)/80640;
    // beta[3] = (574727*v_i*v_i)/20160  - (84463*v_i*v_im1)/1120    + (426341*v_i*v_im2)/10080 - (2782*v_i*v_im3)/315      - (151877*v_i*v_ip1)/10080  + (22439*v_im1*v_im1)/420     - (214259*v_im1*v_im2)/3360  + (92839*v_im1*v_im3)/6720    + (124409*v_im1*v_ip1)/6720  + (410927*v_im2*v_im2)/20160 - (46801*v_im2*v_im3)/5040   - (100889*v_im2*v_ip1)/10080 + (91313*v_im3*v_im3)/80640   + (82157*v_im3*v_ip1)/40320  + (182453*v_ip1*v_ip1)/80640;
    // beta[4] = (785153*v_i*v_i)/80640  - (18406*v_i*v_im1)/315     + (151953*v_i*v_im2)/2240  - (183637*v_i*v_im3)/5040   + (304757*v_i*v_im4)/40320  + (1906127*v_im1*v_im1)/20160 - (776359*v_im1*v_im2)/3360  + (1281461*v_im1*v_im3)/10080 - (269519*v_im1*v_im4)/10080 + (123103*v_im2*v_im2)/840   - (554809*v_im2*v_im3)/3360  + (78943*v_im2*v_im4)/2240   + (957767*v_im3*v_im3)/20160  - (207527*v_im3*v_im4)/10080 + (182453*v_im4*v_im4)/80640;

    beta[0] =   q_i*(107918*q_i   - 649501*q_ip1 + 758823*q_ip2 - 411487*q_ip3 + 86329*q_ip4) + q_ip1*(1020563*q_ip1 - 2462076*q_ip2 + 1358458*q_ip3 - 288007*q_ip4) + q_ip2*(1521393*q_ip2 - 1704396*q_ip3 + 364863*q_ip4) + q_ip3*( 482963*q_ip3 - 208501*q_ip4) +  22658*q_ip4*q_ip4;
    beta[1] = q_im1*( 22658*q_im1 - 140251*q_i   + 165153*q_ip1 -  88297*q_ip2 + 18079*q_ip3) +   q_i*( 242723*q_i   -  611976*q_ip1 +  337018*q_ip2 -  70237*q_ip3) + q_ip1*( 406293*q_ip1 -  464976*q_ip2 +  99213*q_ip3) + q_ip2*( 138563*q_ip2 -  60871*q_ip3) +   6908*q_ip3*q_ip3;
    beta[2] = q_im2*(  6908*q_im2 -  51001*q_im1 +  67923*q_i   -  38947*q_ip1 +  8209*q_ip2) + q_im1*( 104963*q_im1 -  299076*q_i   +  179098*q_ip1 -  38947*q_ip2) +   q_i*( 231153*q_i   -  299076*q_ip1 +  67923*q_ip2) + q_ip1*( 104963*q_ip1 -  51001*q_ip2) +   6908*q_ip2*q_ip2;
    beta[3] = q_im3*(  6908*q_im3 -  60871*q_im2 +  99213*q_im1 -  70237*q_i   + 18079*q_ip1) + q_im2*( 138563*q_im2 -  464976*q_im1 +  337018*q_i   -  88297*q_ip1) + q_im1*( 406293*q_im1 -  611976*q_i   + 165153*q_ip1) +   q_i*( 242723*q_i   - 140251*q_ip1) +  22658*q_ip1*q_ip1;
    beta[4] = q_im4*( 22658*q_im4 - 208501*q_im3 + 364863*q_im2 - 288007*q_im1 + 86329*q_i)   + q_im3*( 482963*q_im3 - 1704396*q_im2 + 1358458*q_im1 - 411487*q_i)   + q_im2*(1521393*q_im2 - 2462076*q_im1 + 758823*q_i)   + q_im1*(1020563*q_im1 - 649501*q_i)   + 107918*q_i*q_i;
    
    Real indicator[5]; // fraction part of (2.59) 
    indicator[0] = 1 / SQR(epsilon + beta[0]);
    indicator[1] = 1 / SQR(epsilon + beta[1]);
    indicator[2] = 1 / SQR(epsilon + beta[2]);
    indicator[3] = 1 / SQR(epsilon + beta[3]);
    indicator[4] = 1 / SQR(epsilon + beta[4]);

    // indicator[0] = 1.0 / std::pow(epsilon + beta[0], 4);
    // indicator[1] = 1.0 / std::pow(epsilon + beta[1], 4);
    // indicator[2] = 1.0 / std::pow(epsilon + beta[2], 4);
    // indicator[3] = 1.0 / std::pow(epsilon + beta[3], 4);
    // indicator[4] = 1.0 / std::pow(epsilon + beta[4], 4);

    // compute qL_ip1
    Real f[5]; // polynomial based on constants c_{r,j} in Table 2.1 
    // Factor of 1/60 in coefficients of f[] array applied to alpha_sum to reduce divisions
    f[0] = 12*q_i   + 77*q_ip1 -  43*q_ip2 +  17*q_ip3 -   3*q_ip4;
    f[1] = -3*q_im1 + 27*q_i   +  47*q_ip1 -  13*q_ip2 +   2*q_ip3;
    f[2] =  2*q_im2 - 13*q_im1 +  47*q_i   +  27*q_ip1 -   3*q_ip2;
    f[3] = -3*q_im3 + 17*q_im2 -  43*q_im1 +  77*q_i   +  12*q_ip1;
    f[4] = 12*q_im4 - 63*q_im3 + 137*q_im2 - 163*q_im1 + 137*q_i;

    Real alpha[5]; // (2.59)
    alpha[0] = indicator[0] *  5.0 / 126.0;
    alpha[1] = indicator[1] * 40.0 / 126.0;
    alpha[2] = indicator[2] * 60.0 / 126.0;
    alpha[3] = indicator[3] * 20.0 / 126.0;
    alpha[4] = indicator[4] *  1.0 / 126.0;

    Real alpha_sum = 60.0 * (alpha[0] + alpha[1] + alpha[2] + alpha[3] + alpha[4]);

    ql_ip1 = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2] + alpha[3] * f[3] + alpha[4] * f[4]) / alpha_sum; // (2.52) 

    f[0] = 12*q_i   + 77*q_im1 -  43*q_im2 +  17*q_im3 -   3*q_im4;
    f[1] = -3*q_ip1 + 27*q_i   +  47*q_im1 -  13*q_im2 +   2*q_im3;
    f[2] =  2*q_ip2 - 13*q_ip1 +  47*q_i   +  27*q_im1 -   3*q_im2;
    f[3] = -3*q_ip3 + 17*q_ip2 -  43*q_ip1 +  77*q_i   +  12*q_im1;
    f[4] = 12*q_ip4 - 63*q_ip3 + 137*q_ip2 - 163*q_ip1 + 137*q_i;

    alpha[0] = indicator[4] *  5.0 / 126.0;
    alpha[1] = indicator[3] * 40.0 / 126.0;
    alpha[2] = indicator[2] * 60.0 / 126.0;
    alpha[3] = indicator[1] * 20.0 / 126.0;
    alpha[4] = indicator[0] *  1.0 / 126.0;

    alpha_sum = 60.0 * (alpha[0] + alpha[1] + alpha[2] + alpha[3] + alpha[4]);

    qr_i = (alpha[0] * f[0] + alpha[1] * f[1] + alpha[2] * f[2] + alpha[3] * f[3] + alpha[4] * f[4]) / alpha_sum; // (2.52) 

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
KOKKOS_INLINE_FUNCTION typename std::enable_if<recon == Reconstruction::weno9js, void>::type
Reconstruct(parthenon::team_mbr_t const &member, const int k, const int j, const int il,
            const int iu, const parthenon::VariablePack<Real> &q, ScratchPad2D<Real> &ql,
            ScratchPad2D<Real> &qr) {
  const auto nvar = q.GetDim(4);
  for (auto n = 0; n < nvar; ++n) {
    parthenon::par_for_inner(member, il, iu, [&](const int i) {
      if constexpr (XNDIR == parthenon::X1DIR) {
        // ql is ql_ip1 and qr is qr_i
        WENO9JS(q(n, k, j, i - 4), q(n, k, j, i - 3), q(n, k, j, i - 2), q(n, k, j, i - 1), q(n, k, j, i), q(n, k, j, i + 1),
              q(n, k, j, i + 2), q(n, k, j, i + 3), q(n, k, j, i + 4), ql(n, i + 1), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X2DIR) {
        // ql is ql_jp1 and qr is qr_j
        WENO9JS(q(n, k, j - 4, i), q(n, k, j - 3, i), q(n, k, j - 2, i), q(n, k, j - 1, i), q(n, k, j, i), q(n, k, j + 1, i),
              q(n, k, j + 2, i), q(n, k, j + 3, i), q(n, k, j + 4, i), ql(n, i), qr(n, i));
      } else if constexpr (XNDIR == parthenon::X3DIR) {
        // ql is ql_kp1 and qr is qr_k
        WENO9JS(q(n, k - 4, j, i), q(n, k - 3, j, i), q(n, k - 2, j, i), q(n, k - 1, j, i), q(n, k, j, i), q(n, k + 1, j, i),
              q(n, k + 2, j, i), q(n, k + 3, j, i), q(n, k + 4, j, i), ql(n, i), qr(n, i));
      } else {
        PARTHENON_FAIL("Unknown direction for WENO9 reconstruction.")
      }
    });
  }
}

#endif // RECONSTRUCT_WENO9_CLASSIC_HPP_
