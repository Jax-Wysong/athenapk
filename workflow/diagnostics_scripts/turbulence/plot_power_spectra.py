"""Plot time-averaged compensated turbulence power spectra.

The spectrum selection, per-snapshot normalization, averaging, Nyquist cutoff, and
k^(5/3) compensation follow the final power-spectrum section of the notebook supplied
by Carolyn Wendeln.
"""

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    description="Plot compensated kinetic and magnetic turbulence power spectra."
)
parser.add_argument(
    "--case",
    action="append",
    required=True,
    metavar="LABEL=DIRECTORY",
    help="Repeat once per numerical case; DIRECTORY contains flow-<dump>-out.hdf5.",
)
parser.add_argument(
    "--dump-id",
    action="append",
    required=True,
    type=int,
    help="Flow-analysis dump ID to include; repeat for each snapshot.",
)
parser.add_argument("--resolution", required=True, type=int)
parser.add_argument("--kinetic", required=True, help="Output kinetic-spectrum PNG")
parser.add_argument("--magnetic", required=True, help="Output magnetic-spectrum PNG")
args = parser.parse_args()


def parse_cases(specifications):
    cases = []
    labels = set()
    for specification in specifications:
        if "=" not in specification:
            parser.error(f"invalid --case {specification!r}; expected LABEL=DIRECTORY")
        label, directory = specification.split("=", 1)
        if not label or not directory:
            parser.error(f"invalid --case {specification!r}; expected LABEL=DIRECTORY")
        if label in labels:
            parser.error(f"duplicate --case label {label!r}")
        labels.add(label)
        cases.append((label, Path(directory).expanduser()))
    return cases


cases = parse_cases(args.case)
dump_ids = sorted(set(args.dump_id))
if args.resolution <= 0:
    parser.error("--resolution must be positive")


def trapezoid(values, coordinates):
    if hasattr(np, "trapezoid"):
        return np.trapezoid(values, x=coordinates)
    return np.trapz(values, x=coordinates)


def mean_spectrum(directory, dataset):
    spectra = []
    wavenumbers = None
    sources = []
    for dump_id in dump_ids:
        filename = directory / f"flow-{dump_id:05d}-out.hdf5"
        if not filename.is_file():
            raise FileNotFoundError(f"Missing flow-analysis file: {filename}")
        with h5py.File(filename, "r") as handle:
            if dataset not in handle:
                raise RuntimeError(f"Missing {dataset!r} in {filename}")
            values = np.asarray(handle[dataset], dtype=float)
        if values.ndim != 2 or values.shape[0] < 2:
            raise RuntimeError(
                f"Unexpected shape {values.shape} for {dataset!r} in {filename}"
            )

        this_k = values[0]
        this_spectrum = values[1]
        if wavenumbers is None:
            wavenumbers = this_k
        elif not np.allclose(wavenumbers, this_k, rtol=1.0e-12, atol=1.0e-12):
            raise RuntimeError(f"Inconsistent wavenumber bins in {filename}")

        normalization = trapezoid(this_spectrum, this_k)
        if not np.isfinite(normalization) or normalization <= 0.0:
            raise RuntimeError(
                f"Non-positive spectrum normalization {normalization} in {filename}"
            )
        spectra.append(this_spectrum / normalization)
        sources.append(filename)

    spectra = np.asarray(spectra)
    return wavenumbers, np.mean(spectra, axis=0), np.std(spectra, axis=0), sources


styles = ("-", "--", "-.", ":")
colors = plt.get_cmap("tab10")


def plot_quantity(dataset, output, ylabel, title):
    figure, axis = plt.subplots(figsize=(8.0, 5.4), constrained_layout=True)
    plotted = []
    for index, (label, directory) in enumerate(cases):
        k, spectrum, deviation, sources = mean_spectrum(directory, dataset)
        mask = (k != 0.0) & (k < args.resolution / 2.0)
        x = k[mask]
        compensation = x ** (5.0 / 3.0)
        y = spectrum[mask] * compensation
        y_lower = np.maximum((spectrum[mask] - deviation[mask]) * compensation, 0.0)
        y_upper = (spectrum[mask] + deviation[mask]) * compensation
        color = colors(index % 10)
        axis.plot(
            x,
            y,
            linewidth=2.5,
            linestyle=styles[index % len(styles)],
            color=color,
            label=label,
        )
        axis.fill_between(x, y_lower, y_upper, alpha=0.1, color=color)
        plotted.append((label, sources))

    reference_min = 4.0
    reference_max = min(40.0, 0.95 * args.resolution / 2.0)
    if reference_max > reference_min:
        x_reference = np.array([reference_min, reference_max])
        y_reference_start = 2.0e-1
        y_reference = y_reference_start * (x_reference / x_reference[0]) ** (-5.0 / 3.0)
        axis.plot(
            x_reference,
            y_reference * x_reference ** (5.0 / 3.0),
            "k--",
            linewidth=2.5,
            label=r"$k^{-5/3}$",
        )

    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set_xlabel(r"wavenumber $k$")
    axis.set_ylabel(ylabel)
    axis.set_title(title)
    axis.grid(which="major", color="lightgray", linestyle="-", linewidth=0.8)
    axis.legend()

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)
    print(f"Saved {output_path}")
    for label, sources in plotted:
        print(f"{label}: used {len(sources)} spectra")
        for source in sources:
            print(f"  {source}")


plot_quantity(
    "rhoU/PowSpec/Full",
    args.kinetic,
    r"$k^{5/3}\widetilde{E}_{\mathrm{kin}}(k)$",
    "Kinetic Power Spectra",
)
plot_quantity(
    "B/PowSpec/Full",
    args.magnetic,
    r"$k^{5/3}\widetilde{E}_{\mathrm{mag}}(k)$",
    "Magnetic Power Spectra",
)

