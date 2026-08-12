Driven MHD turbulence: GLM and UCT-HLLD comparison
===================================================

Comparison of four AthenaPK driven-turbulence configurations: native GLM-MHD with
WENO3 and WENO5, second-order UCT-HLLD with WENO3 and RK2, and the fourth-order
Berta UCT-HLLD scheme with pointwise WENOZ and RK4. All four runs use the same 30
forced Fourier modes, Ornstein--Uhlenbeck random seed, purely solenoidal forcing,
and time-correlation parameters. The native GLM runs retain VL2 and first-order
flux correction; the UCT runs do not use first-order flux correction.

The energy histories show the volume means

.. math::

   \langle E_{\mathrm{kin}}\rangle =
   \frac{1}{V}\int \frac{1}{2}\rho |\mathbf{v}|^2\,dV,

and

.. math::

   \langle E_{\mathrm{mag}}\rangle =
   \frac{1}{V}\int \frac{1}{2}|\mathbf{B}|^2\,dV.

The spatial diagnostics compare density, magnetic-field magnitude, and velocity
magnitude on the final ``z=0.5`` slice. Each diagnostic contains a two-by-two panel
comparison, with shared color limits derived from the combined first and
ninety-ninth percentiles of all four runs.

This preliminary setup follows the supplied input deck: ``rho0=1``, ``p0=1``,
``b0=0.01``, ``accel_rms=1``, ``corr_time=1``, and ``kpeak=2``. These parameters are
retained pending confirmation of the differing Mach-number and initial-beta labels in
the reference presentation.
