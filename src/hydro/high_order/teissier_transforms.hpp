#ifndef HYDRO_HIGH_ORDER_TEISSIER_TRANSFORMS_HPP_
#define HYDRO_HIGH_ORDER_TEISSIER_TRANSFORMS_HPP_
//========================================================================================
// AthenaPK - a performance portable block structured AMR astrophysical MHD code.
// Copyright (c) 2021, Athena-Parthenon Collaboration. All rights reserved.
// Licensed under the BSD 3-Clause License (the "LICENSE").
//========================================================================================

// Parthenon headers
#include <parthenon/package.hpp>

// AthenaPK headers
#include "../../main.hpp"

using namespace parthenon::package::prelude;
using TE = parthenon::TopologicalElement;

namespace Hydro::HIGHORDER {

template <int ORDER, int DIR, typename View>
KOKKOS_INLINE_FUNCTION Real FaceAverageToPoint(const View &q, const int ndim,
                                                const int k, const int j,
                                                const int i) 
{
    if constexpr (ORDER == 4){                                                
        if constexpr (DIR == X1DIR) {
        if (ndim == 1) return q(k, j, i);
        if (ndim == 2) {
            return (13.0 / 12.0) * q(k, j, i) -
                    (1.0 / 24.0) * (q(k, j + 1, i) + q(k, j - 1, i));
        }
        return (7.0 / 6.0) * q(k, j, i) -
                (1.0 / 24.0) * (q(k + 1, j, i) + q(k - 1, j, i) +
                                q(k, j + 1, i) + q(k, j - 1, i));
        } else if constexpr (DIR == X2DIR) {
        if (ndim == 2) {
            return (13.0 / 12.0) * q(k, j, i) -
                    (q(k, j, i - 1) + q(k, j, i + 1)) / 24.0;
        }
        return (7.0 / 6.0) * q(k, j, i) -
                (q(k, j, i - 1) + q(k, j, i + 1) + q(k - 1, j, i) +
                q(k + 1, j, i)) /
                    24.0;
        } else {
        return (7.0 / 6.0) * q(k, j, i) -
                (q(k, j, i - 1) + q(k, j, i + 1) + q(k, j - 1, i) +
                q(k, j + 1, i)) /
                    24.0;
        }
    } else if constexpr (ORDER == 6) {
        if constexpr (DIR == X1DIR) {
        if (ndim == 1) return q(k, j, i);
        if (ndim == 2) {
            return (1067.0/960.0) * q(k, j, i) - 
                    (29.0/480.0) * (q(k, j + 1, i) + q(k, j - 1, i)) +
                    (3.0/640.0) * (q(k, j + 2, i) + q(k, j - 2, i));
        }
        // else ndim == 3
        return (1771.0/1440.0) * q(k,j,i) - 
                (23.0/360.0) * (q(k, j + 1, i) + q(k, j - 1, i) + q(k + 1, j, i) + q(k - 1, j, i)) +
                (3.0/640.0) * (q(k, j + 2, i) +  q(k, j - 2, i) + q(k + 2, j, i) +  q(k - 2, j, i)) +
                (1.0/576.0) * (q(k + 1, j + 1, i) + q(k + 1, j - 1, i) + q(k - 1, j + 1, i) + q(k - 1, j - 1, i));
        } else if constexpr (DIR == X2DIR) {
        if (ndim == 2) {
            return (1067.0/960.0) * q(k, j, i) - 
                    (29.0/480.0) * (q(k, j, i + 1) + q(k, j, i - 1)) +
                    (3.0/640.0) * (q(k, j, i + 2) + q(k, j, i - 2));
        }
        // else ndim == 3
        return (1771.0/1440.0) * q(k,j,i) - 
                (23.0/360.0) * (q(k, j, i + 1) + q(k, j, i - 1) + q(k + 1, j, i) + q(k - 1, j, i)) +
                (3.0/640.0) * (q(k, j, i + 2) +  q(k, j, i - 2) + q(k + 2, j, i) +  q(k - 2, j, i)) +
                (1.0/576.0) * (q(k + 1, j, i + 1) + q(k + 1, j, i - 1) + q(k - 1, j, i + 1) + q(k - 1, j, i - 1));
        } else {
        // else xdir == ndim == 3
        return (1771.0/1440.0) * q(k,j,i) - 
                (23.0/360.0) * (q(k, j + 1, i) + q(k, j - 1, i) + q(k, j, i + 1) + q(k, j, i - 1)) +
                (3.0/640.0) * (q(k, j + 2, i) +  q(k, j - 2, i) + q(k, j, i + 2) +  q(k, j, i - 2)) +
                (1.0/576.0) * (q(k, j + 1, i + 1) + q(k, j - 1, i + 1) + q(k, j + 1, i - 1) + q(k, j - 1, i - 1));
        }
    } else if constexpr (ORDER == 8) {
        if constexpr (DIR == X1DIR) {
        if (ndim == 1) return q(k, j, i);
        if (ndim == 2) {
            return (30251.0/26880.0) * q(k, j, i) -
                   (7621.0/107520.0) * (q(k, j + 1, i) + q(k, j - 1, i)) +
                   (159.0/17920.0)   * (q(k, j + 2, i) + q(k, j - 2, i)) -
                   (5.0/7168.0)      * (q(k, j + 3, i) + q(k, j - 3, i));
        }
        // else ndim == 3
        return (25451.0/20160.0) * q(k,j,i) - 
                (4973.0/64512.0) * (q(k, j + 1, i) + q(k, j - 1, i) + q(k + 1, j, i) + q(k - 1, j, i)) +
                (83.0/8960.0) * (q(k, j + 2, i) +  q(k, j - 2, i) + q(k + 2, j, i) +  q(k - 2, j, i)) -
                (5.0/7168.0) * (q(k, j + 3, i) +  q(k, j - 3, i) + q(k + 3, j, i) +  q(k - 3, j, i)) +
                (19.0/5760.0) * (q(k + 1, j + 1, i) + q(k + 1, j - 1, i) + q(k - 1, j + 1, i) + q(k - 1, j - 1, i)) -
                (1.0/5120.0) * (q(k + 2, j + 1, i) + q(k + 2, j - 1, i) + q(k - 2, j + 1, i) + q(k - 2, j - 1, i) + q(k + 1, j + 2, i) + q(k + 1, j - 2, i) + q(k - 1, j + 2, i) + q(k - 1, j - 2, i));
        } else if constexpr (DIR == X2DIR) {
        if (ndim == 2) {
            return (30251.0/26880.0) * q(k, j, i) -
                   (7621.0/107520.0) * (q(k, j, i + 1) + q(k, j, i - 1)) +
                   (159.0/17920.0)   * (q(k, j, i + 2) + q(k, j, i - 2)) -
                   (5.0/7168.0)      * (q(k, j, i + 3) + q(k, j, i - 3));
        }
        // else ndim == 3
        return (25451.0/20160.0) * q(k,j,i) - 
                (4973.0/64512.0) * (q(k, j, i + 1) + q(k, j, i - 1) + q(k + 1, j, i) + q(k - 1, j, i)) +
                (83.0/8960.0) * (q(k, j, i + 2) +  q(k, j, i - 2) + q(k + 2, j, i) +  q(k - 2, j, i)) -
                (5.0/7168.0) * (q(k, j, i + 3) +  q(k, j, i - 3) + q(k + 3, j, i) +  q(k - 3, j, i)) +
                (19.0/5760.0) * (q(k + 1, j, i + 1) + q(k + 1, j, i - 1) + q(k - 1, j, i + 1) + q(k - 1, j, i - 1)) -
                (1.0/5120.0) * (q(k + 2, j, i + 1) + q(k + 2, j, i - 1) + q(k - 2, j, i + 1) + q(k - 2, j, i - 1) + q(k + 1, j, i + 2) + q(k + 1, j, i - 2) + q(k - 1, j, i + 2) + q(k - 1, j, i - 2));
        } else {
        // else xdir == ndim == 3
        return (25451.0/20160.0) * q(k,j,i) - 
                (4973.0/64512.0) * (q(k, j + 1, i) + q(k, j - 1, i) + q(k, j, i + 1) + q(k, j, i - 1)) +
                (83.0/8960.0) * (q(k, j + 2, i) +  q(k, j - 2, i) + q(k, j, i + 2) +  q(k, j, i - 2)) -
                (5.0/7168.0) * (q(k, j + 3, i) +  q(k, j - 3, i) + q(k, j, i + 3) +  q(k, j, i - 3)) +
                (19.0/5760.0) * (q(k, j + 1, i + 1) + q(k, j - 1, i + 1) + q(k, j + 1, i - 1) + q(k, j - 1, i - 1)) -
                (1.0/5120.0) * (q(k, j + 1, i + 2) + q(k, j - 1, i + 2) + q(k, j + 1, i - 2) + q(k, j - 1, i - 2) + q(k, j + 2, i + 1) + q(k, j - 2, i + 1) + q(k, j + 2, i - 1) + q(k, j - 2, i - 1));
        }
    }
}

template <int ORDER, int DIR, typename View>
KOKKOS_INLINE_FUNCTION Real PointToFaceAverage(const View &pf, 
                                               const int ndim,
                                               const int n, 
                                               const int k, const int j, const int i)
{
    if constexpr (ORDER == 4) {
        if constexpr (DIR == X1DIR) {
            if (ndim == 1){
                return pf(TE::F1, n, k, j, i);
            } else if (ndim == 2){
                return (11.0/12.0) * pf(TE::F1, n, k, j, i) + 
                        (1.0/24.0) * (pf(TE::F1, n, k, j+1, i) + pf(TE::F1, n, k, j-1, i));
            } // else ndim == 3
            return (5.0/6.0) * pf(TE::F1, n, k, j, i) +
                    (1.0/24.0) * (pf(TE::F1, n, k + 1, j, i) + pf(TE::F1, n, k - 1, j, i) + pf(TE::F1, n, k, j + 1, i) + pf(TE::F1, n , k, j - 1, i));
        } else if constexpr (DIR == X2DIR) {
            if (ndim == 2){
                return (11.0/12.0) * pf(TE::F2, n, k, j, i) + 
                        (1.0/24.0) * (pf(TE::F2, n, k, j, i+1) + pf(TE::F2, n, k, j, i-1));
            } // else ndim == 3
            return (5.0/6.0) * pf(TE::F2, n, k, j, i) +
                    (1.0/24.0) * (pf(TE::F2, n, k + 1, j, i) + pf(TE::F2, n, k - 1, j, i) + pf(TE::F2, n, k, j, i + 1) + pf(TE::F2, n , k, j, i - 1));
        } else if constexpr (DIR == X3DIR) {
            // ndim == DIR == 3
            return (5.0/6.0) * pf(TE::F3, n, k, j, i) +
                    (1.0/24.0) * (pf(TE::F3, n, k, j, i + 1) + pf(TE::F3, n, k, j, i - 1) + pf(TE::F3, n, k, j + 1, i) + pf(TE::F3, n , k, j - 1, i));
        }
    } else if constexpr (ORDER == 6) {
        if constexpr (DIR == X1DIR) {
            if (ndim == 1){
                return pf(TE::F1, n, k, j, i);
            } else if (ndim == 2){
                return (863.0/960.0) * pf(TE::F1, n, k, j, i) +
                        (77.0/1440.0) * (pf(TE::F1, n, k, j + 1, i) + pf(TE::F1, n, k, j - 1, i)) -
                        (17.0/5760.0) * (pf(TE::F1, n, k, j + 2, i) + pf(TE::F1, n, k, j - 2, i));
            } // else ndim == 3
            return (1159.0/1440.0) * pf(TE::F1, n, k, j, i) +
                    (1.0/20.0) * (pf(TE::F1, n, k + 1, j, i) + pf(TE::F1, n, k - 1, j, i) + pf(TE::F1, n, k, j + 1, i) + pf(TE::F1, n, k, j - 1, i)) -
                    (17.0/5760.0) * (pf(TE::F1, n, k + 2, j, i) + pf(TE::F1, n, k - 2, j, i) + pf(TE::F1, n, k, j + 2, i) + pf(TE::F1, n, k, j - 2, i)) +
                    (1.0/576.0) * (pf(TE::F1, n, k + 1, j + 1, i) + pf(TE::F1, n, k + 1, j - 1, i) + pf(TE::F1, n, k - 1, j + 1, i) + pf(TE::F1, n, k - 1, j - 1, i));
        } else if constexpr (DIR == X2DIR) {
            if (ndim == 2){
                return (863.0/960.0) * pf(TE::F2, n, k, j, i) +
                        (77.0/1440.0) * (pf(TE::F2, n, k, j, i + 1) + pf(TE::F2, n, k, j, i - 1)) -
                        (17.0/5760.0) * (pf(TE::F2, n, k, j, i + 2) + pf(TE::F2, n, k, j, i - 2));
            } // else ndim == 3
            return (1159.0/1440.0) * pf(TE::F2, n, k, j, i) +
                    (1.0/20.0) * (pf(TE::F2, n, k + 1, j, i) + pf(TE::F2, n, k - 1, j, i) + pf(TE::F2, n, k, j, i + 1) + pf(TE::F2, n, k, j, i - 1)) -
                    (17.0/5760.0) * (pf(TE::F2, n, k + 2, j, i) + pf(TE::F2, n, k - 2, j, i) + pf(TE::F2, n, k, j, i + 2) + pf(TE::F2, n, k, j, i - 2)) +
                    (1.0/576.0) * (pf(TE::F2, n, k + 1, j, i + 1) + pf(TE::F2, n, k + 1, j, i - 1) + pf(TE::F2, n, k - 1, j, i + 1) + pf(TE::F2, n, k - 1, j, i - 1));
        } else if constexpr (DIR == X3DIR) {
            // ndim == DIR == 3
            return (1159.0/1440.0) * pf(TE::F3, n, k, j, i) +
                    (1.0/20.0) * (pf(TE::F3, n, k, j, i + 1) + pf(TE::F3, n, k, j, i - 1) + pf(TE::F3, n, k, j + 1, i) + pf(TE::F3, n, k, j - 1, i)) -
                    (17.0/5760.0) * (pf(TE::F3, n, k, j, i + 2) + pf(TE::F3, n, k, j, i - 2) + pf(TE::F3, n, k, j + 2, i) + pf(TE::F3, n, k, j - 2, i)) +
                    (1.0/576.0) * (pf(TE::F3, n, k, j + 1, i + 1) + pf(TE::F3, n, k, j - 1, i + 1) + pf(TE::F3, n, k, j + 1, i - 1) + pf(TE::F3, n, k, j - 1, i - 1));
        }
    } else if constexpr (ORDER == 8) {
        if constexpr (DIR == X1DIR) {
        if (ndim == 1) return pf(TE::F1, n, k, j, i);
        if (ndim == 2) {
            return (215641.0/241920.0) * pf(TE::F1, n, k, j, i) +
                   (6361.0/107520.0) * (pf(TE::F1, n, k, j + 1, i) + pf(TE::F1, n, k, j - 1, i)) -
                   (281.0/53760.0) * (pf(TE::F1, n, k, j + 2, i) + pf(TE::F1, n, k, j - 2, i)) +
                   (367.0/967680.0) * (pf(TE::F1, n, k, j + 3, i) + pf(TE::F1, n, k, j - 3, i));
        }
        // else ndim == 3
        return (47939.0/60480.0) * pf(TE::F1, n, k,j,i) + 
                (52223.0/967680.0) * (pf(TE::F1, n, k, j + 1, i) + pf(TE::F1, n, k, j - 1, i) + pf(TE::F1, n, k + 1, j, i) + pf(TE::F1, n, k - 1, j, i)) -
                (241.0/48384.0) * (pf(TE::F1, n, k, j + 2, i) +  pf(TE::F1, n, k, j - 2, i) + pf(TE::F1, n, k + 2, j, i) +  pf(TE::F1, n, k - 2, j, i)) +
                (367.0/967680.0) * (pf(TE::F1, n, k, j + 3, i) +  pf(TE::F1, n, k, j - 3, i) + pf(TE::F1, n, k + 3, j, i) +  pf(TE::F1, n, k - 3, j, i)) +
                (47.0/17280.0) * (pf(TE::F1, n, k + 1, j + 1, i) + pf(TE::F1, n, k + 1, j - 1, i) + pf(TE::F1, n, k - 1, j + 1, i) + pf(TE::F1, n, k - 1, j - 1, i)) -
                (17.0/138240.0) * (pf(TE::F1, n, k + 2, j + 1, i) + pf(TE::F1, n, k + 2, j - 1, i) + pf(TE::F1, n, k - 2, j + 1, i) + pf(TE::F1, n, k - 2, j - 1, i) + pf(TE::F1, n, k + 1, j + 2, i) + pf(TE::F1, n, k + 1, j - 2, i) + pf(TE::F1, n, k - 1, j + 2, i) + pf(TE::F1, n, k - 1, j - 2, i));
        } else if constexpr (DIR == X2DIR) {
        if (ndim == 2) {
            return (215641.0/241920.0) * pf(TE::F2, n, k, j, i) +
                   (6361.0/107520.0) * (pf(TE::F2, n, k, j, i + 1) + pf(TE::F2, n, k, j, i - 1)) -
                   (281.0/53760.0) * (pf(TE::F2, n, k, j, i + 2) + pf(TE::F2, n, k, j, i - 2)) +
                   (367.0/967680.0) * (pf(TE::F2, n, k, j, i + 3) + pf(TE::F2, n, k, j, i - 3));
        }
        // else ndim == 3
        return (47939.0/60480.0) * pf(TE::F2, n, k,j,i) + 
                (52223.0/967680.0) * (pf(TE::F2, n, k, j, i + 1) + pf(TE::F2, n, k, j, i - 1) + pf(TE::F2, n, k + 1, j, i) + pf(TE::F2, n, k - 1, j, i)) -
                (241.0/48384.0) * (pf(TE::F2, n, k, j, i + 2) +  pf(TE::F2, n, k, j, i - 2) + pf(TE::F2, n, k + 2, j, i) +  pf(TE::F2, n, k - 2, j, i)) +
                (367.0/967680.0) * (pf(TE::F2, n, k, j, i + 3) +  pf(TE::F2, n, k, j, i - 3) + pf(TE::F2, n, k + 3, j, i) +  pf(TE::F2, n, k - 3, j, i)) +
                (47.0/17280.0) * (pf(TE::F2, n, k + 1, j, i + 1) + pf(TE::F2, n, k + 1, j, i - 1) + pf(TE::F2, n, k - 1, j, i + 1) + pf(TE::F2, n, k - 1, j, i - 1)) -
                (17.0/138240.0) * (pf(TE::F2, n, k + 2, j, i + 1) + pf(TE::F2, n, k + 2, j, i - 1) + pf(TE::F2, n, k - 2, j, i + 1) + pf(TE::F2, n, k - 2, j, i - 1) + pf(TE::F2, n, k + 1, j, i + 2) + pf(TE::F2, n, k + 1, j, i - 2) + pf(TE::F2, n, k - 1, j, i + 2) + pf(TE::F2, n, k - 1, j, i - 2));
        } else {
        // else xdir == ndim == 3
        return (47939.0/60480.0) * pf(TE::F3, n, k,j,i) + 
                (52223.0/967680.0) * (pf(TE::F3, n, k, j + 1, i) + pf(TE::F3, n, k, j - 1, i) + pf(TE::F3, n, k, j, i + 1) + pf(TE::F3, n, k, j, i - 1)) -
                (241.0/48384.0) * (pf(TE::F3, n, k, j + 2, i) +  pf(TE::F3, n, k, j - 2, i) + pf(TE::F3, n, k, j, i + 2) +  pf(TE::F3, n, k, j, i - 2)) +
                (367.0/967680.0) * (pf(TE::F3, n, k, j + 3, i) +  pf(TE::F3, n, k, j - 3, i) + pf(TE::F3, n, k, j, i + 3) +  pf(TE::F3, n, k, j, i - 3)) +
                (47.0/17280.0) * (pf(TE::F3, n, k, j + 1, i + 1) + pf(TE::F3, n, k, j - 1, i + 1) + pf(TE::F3, n, k, j + 1, i - 1) + pf(TE::F3, n, k, j - 1, i - 1)) -
                (17.0/138240.0) * (pf(TE::F3, n, k, j + 1, i + 2) + pf(TE::F3, n, k, j - 1, i + 2) + pf(TE::F3, n, k, j + 1, i - 2) + pf(TE::F3, n, k, j - 1, i - 2) + pf(TE::F3, n, k, j + 2, i + 1) + pf(TE::F3, n, k, j - 2, i + 1) + pf(TE::F3, n, k, j + 2, i - 1) + pf(TE::F3, n, k, j - 2, i - 1));
        }
    }
}


} // namespace Hydro::HIGHORDER

#endif // HYDRO_HIGH_ORDER_TEISSIER_TRANSFORMS_HPP_
