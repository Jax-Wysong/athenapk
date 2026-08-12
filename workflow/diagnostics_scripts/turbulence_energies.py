"""Compare GLM and UCT-HLLD turbulence energy histories."""

import argparse
from pathlib import Path
import re

import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    description="Plot mean kinetic and magnetic energy from turbulence HST files."
)
parser.add_argument(
    "--case",
    action="append",
    required=True,
    metavar="LABEL=DIRECTORY",
    help="Repeat once for each numerical case to compare.",
)
parser.add_argument("--plot", required=True, help="Output energy-history PNG")
parser.add_argument("--summary", required=True, help="Output text summary")
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
        parsed.append((label, directory))
    return parsed


cases = parse_cases(args.case)

required_columns = {
    "time",
    "Ms",
    "Ma",
    "plasma_beta",
    "mean_Ekin",
    "mean_Emag",
}


def find_history(directory):
    files = sorted(Path(directory).expanduser().glob("*.hst"))
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
    missing = required_columns.difference(history)
    if missing:
        raise RuntimeError(f"Missing HST columns in {filename}: {sorted(missing)}")

    finite = np.ones(values.shape[0], dtype=bool)
    for name in required_columns:
        finite &= np.isfinite(history[name])
    if not np.all(finite):
        raise RuntimeError(f"Non-finite turbulence history values in {filename}")

    # If a restart appended a duplicate time, retain the final occurrence.
    reverse_unique = np.unique(history["time"][::-1], return_index=True)[1]
    keep = np.sort(values.shape[0] - 1 - reverse_unique)
    history = {name: column[keep] for name, column in history.items()}
    return filename, history


histories = {}
history_files = {}
for label, directory in cases:
    filename, history = load_history(directory)
    history_files[label] = filename
    histories[label] = history

color_cycle = plt.get_cmap("tab10")
line_cycle = ("-", "--", "-.", ":")
styles = {
    label: {
        "color": color_cycle(index % 10),
        "linestyle": line_cycle[index % len(line_cycle)],
    }
    for index, (label, _) in enumerate(cases)
}

figure, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
for label, history in histories.items():
    style = {
        **styles[label],
        "linewidth": 2.2,
        "label": label,
    }
    axes[0].plot(history["time"], history["mean_Ekin"], **style)
    axes[1].plot(history["time"], history["mean_Emag"], **style)

axes[0].set_title("Mean Kinetic Energy")
axes[0].set_ylabel(r"$\langle E_{\mathrm{kin}}\rangle$")
axes[1].set_title("Mean Magnetic Energy")
axes[1].set_ylabel(r"$\langle E_{\mathrm{mag}}\rangle$")
for axis in axes:
    axis.set_xlabel("Time")
    axis.grid(alpha=0.25)
    axis.legend()
    axis.set_xlim(left=0.0)
    axis.set_ylim(bottom=0.0)

plot_output = Path(args.plot).expanduser()
plot_output.parent.mkdir(parents=True, exist_ok=True)
figure.savefig(plot_output, dpi=180)
plt.close(figure)


def stats(values, mask):
    return float(np.mean(values[mask])), float(np.std(values[mask]))


summary_lines = [
    "AthenaPK driven-MHD turbulence comparison",
    "",
    "Definitions:",
    "  mean_Ekin = <0.5 rho |v|^2>_volume",
    "  mean_Emag = <0.5 |B|^2>_volume",
    "  Late-time interval = final third of each run",
    "",
]
for label, history in histories.items():
    time = history["time"]
    late_start = time[-1] - (time[-1] - time[0]) / 3.0
    late = time >= late_start
    ekin_mean, ekin_std = stats(history["mean_Ekin"], late)
    emag_mean, emag_std = stats(history["mean_Emag"], late)
    ms_mean, ms_std = stats(history["Ms"], late)
    ma_mean, ma_std = stats(history["Ma"], late)
    beta_mean, beta_std = stats(history["plasma_beta"], late)

    summary_lines.extend(
        [
            f"{label}",
            f"  source: {history_files[label]}",
            f"  final time: {time[-1]:.10g}",
            f"  initial mean_Ekin: {history['mean_Ekin'][0]:.10e}",
            f"  initial mean_Emag: {history['mean_Emag'][0]:.10e}",
            f"  late mean_Ekin: {ekin_mean:.10e} +/- {ekin_std:.10e}",
            f"  late mean_Emag: {emag_mean:.10e} +/- {emag_std:.10e}",
            f"  late Ms: {ms_mean:.10e} +/- {ms_std:.10e}",
            f"  late Ma: {ma_mean:.10e} +/- {ma_std:.10e}",
            f"  late plasma_beta: {beta_mean:.10e} +/- {beta_std:.10e}",
            "",
        ]
    )

summary_output = Path(args.summary).expanduser()
summary_output.parent.mkdir(parents=True, exist_ok=True)
summary_output.write_text("\n".join(summary_lines), encoding="utf-8")

print(f"Saved energy histories: {plot_output}")
print(f"Saved summary: {summary_output}")
