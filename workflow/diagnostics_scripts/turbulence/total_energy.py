"""Plot mean kinetic plus magnetic energy for turbulence runs.

This diagnostic intentionally excludes thermal/internal energy.  The plotted quantity is

    <E_kin + E_mag> = <0.5 rho |v|^2> + <0.5 |B|^2>,

using the volume-averaged history columns written by the turbulence problem.
"""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    description="Plot mean kinetic-plus-magnetic energy from turbulence HST files."
)
parser.add_argument(
    "--case",
    action="append",
    required=True,
    metavar="LABEL=DIRECTORY",
    help="Repeat once for each numerical case to compare.",
)
parser.add_argument("--plot", required=True, help="Output energy-history PNG")
args = parser.parse_args()


def parse_cases(specifications):
    parsed = []
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
        parsed.append((label, Path(directory).expanduser()))
    return parsed


def find_history(directory):
    files = sorted(directory.glob("*.hst"))
    if len(files) != 1:
        raise RuntimeError(
            f"Expected exactly one HST file in {directory}, found {len(files)}"
        )
    return files[0]


def load_history(directory):
    filename = find_history(directory)
    names = None
    with filename.open("r", encoding="utf-8") as stream:
        for line in stream:
            parsed = re.findall(r"\[\d+\]=([^\s]+)", line)
            if parsed:
                names = parsed

    if names is None:
        raise RuntimeError(f"Could not find an indexed column header in {filename}")

    values = np.loadtxt(filename, comments="#", ndmin=2)
    if values.shape[1] != len(names):
        raise RuntimeError(
            f"HST column mismatch in {filename}: {values.shape[1]} values but "
            f"{len(names)} names"
        )

    history = {name: values[:, index] for index, name in enumerate(names)}
    required = {"time", "mean_Ekin", "mean_Emag"}
    missing = required.difference(history)
    if missing:
        raise RuntimeError(f"Missing HST columns in {filename}: {sorted(missing)}")

    finite = np.ones(values.shape[0], dtype=bool)
    for name in required:
        finite &= np.isfinite(history[name])
    if not np.all(finite):
        raise RuntimeError(f"Non-finite turbulence history values in {filename}")

    # Restarted runs append to the HST and can introduce a backward jump in time.
    # Retain only the ultimately accepted, strictly increasing trajectory.
    keep = []
    times = history["time"]
    for index, time in enumerate(times):
        while keep and times[keep[-1]] >= time:
            keep.pop()
        keep.append(index)
    keep = np.asarray(keep, dtype=int)
    history = {name: column[keep] for name, column in history.items()}

    if history["time"].size > 1 and not np.all(np.diff(history["time"]) > 0.0):
        raise RuntimeError(
            f"Could not reconstruct a strictly increasing history from {filename}"
        )
    return filename, history


cases = parse_cases(args.case)
colors = plt.get_cmap("tab10")
line_styles = ("-", "--", "-.", ":")

figure, axis = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
for index, (label, directory) in enumerate(cases):
    filename, history = load_history(directory)
    total = history["mean_Ekin"] + history["mean_Emag"]
    axis.plot(
        history["time"],
        total,
        color=colors(index % 10),
        linestyle=line_styles[index % len(line_styles)],
        linewidth=2.2,
        label=label,
    )
    print(
        f"{label}: final time={history['time'][-1]:.10g}, "
        f"final <Ekin+Emag>={total[-1]:.10e}, source={filename}"
    )

axis.set_title("Mean Kinetic + Magnetic Energy")
axis.set_xlabel("Time")
axis.set_ylabel(r"$\langle E_{\mathrm{kin}} + E_{\mathrm{mag}}\rangle$")
axis.set_xlim(left=0.0)
axis.set_ylim(bottom=0.0)
axis.grid(alpha=0.25)
axis.legend()

plot_output = Path(args.plot).expanduser()
plot_output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(plot_output, dpi=180)
plt.close(figure)
print(f"Saved kinetic-plus-magnetic energy history: {plot_output}")
