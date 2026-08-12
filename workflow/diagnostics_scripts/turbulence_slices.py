"""Plot matched final midplane slices for GLM and UCT-HLLD turbulence runs."""

import argparse
from pathlib import Path

import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import yt


yt.set_log_level(50)

parser = argparse.ArgumentParser(
    description="Make matched GLM/UCT-HLLD turbulence slices through z=0.5."
)
parser.add_argument("--glm-weno3-dir", required=True)
parser.add_argument("--glm-weno5-dir", required=True)
parser.add_argument("--uct2-dir", required=True)
parser.add_argument("--uct4-dir", required=True)
parser.add_argument("--density", required=True, help="Output density PNG")
parser.add_argument("--bmag", required=True, help="Output magnetic-magnitude PNG")
parser.add_argument("--velocity", required=True, help="Output velocity-magnitude PNG")
parser.add_argument("--expected-final-time", type=float, required=True)
parser.add_argument("--time-tolerance", type=float, default=1.0e-8)
parser.add_argument("--percentile-min", type=float, default=1.0)
parser.add_argument("--percentile-max", type=float, default=99.0)
args = parser.parse_args()

cases = (
    ("GLM WENO3", args.glm_weno3_dir),
    ("GLM WENO5", args.glm_weno5_dir),
    ("UCT-HLLD 2nd", args.uct2_dir),
    ("UCT-HLLD Berta4", args.uct4_dir),
)

if args.time_tolerance <= 0.0:
    parser.error("--time-tolerance must be positive")
if not 0.0 <= args.percentile_min < args.percentile_max <= 100.0:
    parser.error("percentile limits must satisfy 0 <= min < max <= 100")

density_field = ("gas", "density")
bmag_field = ("gas", "magnetic_field_magnitude")
velocity_field = ("gas", "velocity_magnitude")


def final_dataset(directory):
    files = list(Path(directory).expanduser().glob("*.phdf"))
    if not files:
        raise FileNotFoundError(f"No PHDF files found in {directory}")

    timed = []
    for filename in files:
        ds = yt.load(str(filename))
        timed.append((float(ds.current_time.to_value()), "final" in filename.name, filename))
    timed.sort(key=lambda item: (item[0], item[1]))
    time, _, filename = timed[-1]
    if abs(time - args.expected_final_time) > args.time_tolerance:
        raise RuntimeError(
            f"Latest PHDF in {directory} is at t={time:.16g}; expected "
            f"t={args.expected_final_time:.16g}"
        )
    return time, filename, yt.load(str(filename))


def require_field(ds, field):
    if field not in ds.derived_field_list and field not in ds.field_list:
        raise RuntimeError(f"Required yt field {field} is unavailable in {ds}")


def midplane_array(ds, field):
    require_field(ds, field)
    center = ds.domain_center
    z_slice = ds.slice(2, center[2])
    width = (float(ds.domain_width[0].to_value("code_length")), "code_length")
    resolution = (int(ds.domain_dimensions[0]), int(ds.domain_dimensions[1]))
    frb = z_slice.to_frb(width, resolution, center=center)
    return np.asarray(frb[field].to_value())


datasets = {}
times = {}
filenames = {}
for label, directory in cases:
    time, filename, ds = final_dataset(directory)
    times[label] = time
    filenames[label] = filename
    datasets[label] = ds


def common_limits(arrays, logarithmic):
    combined = np.concatenate([array.ravel() for array in arrays])
    combined = combined[np.isfinite(combined)]
    if logarithmic:
        combined = combined[combined > 0.0]
    if combined.size == 0:
        raise RuntimeError("No finite values available for common color limits")

    lower, upper = np.percentile(
        combined, [args.percentile_min, args.percentile_max]
    )
    if upper <= lower:
        scale = max(abs(float(lower)), 1.0)
        lower = max(float(lower) - 1.0e-6 * scale, np.finfo(float).tiny)
        upper = float(upper) + 1.0e-6 * scale
    return float(lower), float(upper)


def render_comparison(field, output, title, colorbar_label, cmap, logarithmic):
    arrays = {label: midplane_array(ds, field) for label, ds in datasets.items()}
    lower, upper = common_limits(list(arrays.values()), logarithmic)
    norm = colors.LogNorm(lower, upper) if logarithmic else colors.Normalize(lower, upper)

    figure, axes = plt.subplots(2, 2, figsize=(11, 10), constrained_layout=True)
    image = None
    for axis, (label, _) in zip(axes.flat, cases):
        ds = datasets[label]
        extent = [
            float(ds.domain_left_edge[0].to_value("code_length")),
            float(ds.domain_right_edge[0].to_value("code_length")),
            float(ds.domain_left_edge[1].to_value("code_length")),
            float(ds.domain_right_edge[1].to_value("code_length")),
        ]
        image = axis.imshow(
            arrays[label],
            origin="lower",
            extent=extent,
            cmap=cmap,
            norm=norm,
            interpolation="nearest",
            aspect="equal",
        )
        axis.set_title(f"{label}, t = {times[label]:.3f}")
        axis.set_xlabel("x")
        axis.set_ylabel("y")

    figure.suptitle(f"{title} at z = 0.5")
    colorbar = figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.88)
    colorbar.set_label(colorbar_label)

    output_path = Path(output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved {output_path} with common limits [{lower:.8g}, {upper:.8g}]")


render_comparison(
    density_field,
    args.density,
    "Driven Turbulence Density",
    r"$\rho$",
    "viridis",
    False,
)
render_comparison(
    bmag_field,
    args.bmag,
    "Driven Turbulence Magnetic-Field Magnitude",
    r"$|\mathbf{B}|$",
    "magma",
    True,
)
render_comparison(
    velocity_field,
    args.velocity,
    "Driven Turbulence Velocity Magnitude",
    r"$|\mathbf{v}|$",
    "inferno",
    False,
)

for label, _ in cases:
    print(f"{label} final source: {filenames[label]}")
