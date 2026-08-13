Driven MHD turbulence comparison
================================

The initial reference comparison uses native GLM-MHD with the classical WENO3JS and
WENO5JS implementations supplied by Claire Wendeln. Both cases use HLLD, RK3, CFL
0.1, and no first-order flux correction. The physical setup, forcing parameters, and
mode list come from the supplied ``turbulence_wendeln.in`` input deck.

The energy histories show the volume means

.. math::

   \langle E_{\mathrm{kin}}\rangle =
   \frac{1}{V}\int \frac{1}{2}\rho |\mathbf{v}|^2\,dV,

and

.. math::

   \langle E_{\mathrm{mag}}\rangle =
   \frac{1}{V}\int \frac{1}{2}|\mathbf{B}|^2\,dV.

The spatial diagnostics compare density, magnetic-field magnitude, and velocity
magnitude on the final ``z=0.5`` slice. Panels use shared color limits derived from
the combined first and ninety-ninth percentiles of all configured runs.

The reference setup uses ``rho0=1``, ``p0=1``, ``b0=0.01``, a sinusoidal zero-net-flux
field (``b_config=2``), ``accel_rms=0.125``, ``corr_time=1``, and ``kpeak=2``.

For each configured spectrum snapshot, the workflow calls the unmodified flow-analysis
route from ``pgrete/energy-transfer-analysis`` (branch ``back-to-mpi4py-fft``, commit
``d1577f91f96db2a8ff2d7e3c64375becce014647``). The plotted kinetic spectrum is based
on :math:`|\mathcal{F}(\sqrt{\rho}\,\mathbf{v})|^2`, and the magnetic spectrum on
:math:`|\mathcal{F}(\mathbf{B})|^2`. Following Wendeln's supplied notebook, each
snapshot spectrum is normalized by its integral before the mean and standard deviation
are calculated. The plots show the compensated spectra :math:`k^{5/3}\widetilde E(k)`
for :math:`0 < k < N/2`.
