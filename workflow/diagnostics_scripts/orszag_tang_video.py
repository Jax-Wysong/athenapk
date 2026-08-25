"""Render complete multi-block Orszag-Tang diagnostics with yt."""

import argparse
import gc
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yt


yt.set_log_level(50)

parser = argparse.ArgumentParser(
    description="Make GIFs and a final pressure plot from Orszag-Tang PHDF files."
)
parser.add_argument("directory", help="Directory containing the PHDF files")
parser.add_argument(
    "-o", "--output", default="orszag_tang", help="Output GIF filename prefix"
)
parser.add_argument(
    "--pressure-final",
    help="Output PNG filename for pressure at the final saved time",
)
parser.add_argument(
    "--final-only",
    action="store_true",
    help="Write only the final pressure PNG and skip GIF generation",
)
parser.add_argument("--fps", type=int, default=4, help="Animation frames per second")
parser.add_argument(
    "--time-tolerance",
    type=float,
    default=1.0e-10,
    help="Tolerance used to identify duplicate output times",
)
args = parser.parse_args()

if args.final_only and not args.pressure_final:
    parser.error("--final-only requires --pressure-final")
if args.fps <= 0:
    parser.error("--fps must be positive")
if args.time_tolerance <= 0.0:
    parser.error("--time-tolerance must be positive")

data_dir = Path(args.directory).expanduser()
files = list(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

n_match = re.search(r"N(\d+)", files[0].name)
n_label = f"N{n_match.group(1)}" if n_match else None

output = Path(args.output).expanduser()
if not output.is_absolute():
    output = data_dir.parent / output
if n_label:
    output = output.with_name(f"{output.name}_{n_label}")

pressure_field = ("gas", "pressure")
bmag_field = ("gas", "magnetic_field_magnitude")


def physical_time(filename):
    ds = yt.load(str(filename))
    time = float(ds.current_time.to_value())
    del ds
    gc.collect()
    return time


timed_files = sorted((physical_time(filename), filename) for filename in files)

# A cadence output and the final-labeled output can represent the same time. Keep
# only one frame, preferring the explicitly final-labeled file.
snapshots = []
for time, filename in timed_files:
    if snapshots and abs(time - snapshots[-1][0]) <= args.time_tolerance:
        if "final" in filename.name.lower():
            snapshots[-1] = (time, filename)
    else:
        snapshots.append((time, filename))


def read_fields(snapshot):
    """Assemble all meshblocks into uniform y-by-x arrays using yt."""
    time, filename = snapshot
    ds = yt.load(str(filename))
    dimensions = np.asarray(ds.domain_dimensions, dtype=int)
    if dimensions[2] != 1:
        raise RuntimeError(
            f"Expected a 2D Orszag-Tang output, got dimensions {dimensions}"
        )

    grid = ds.covering_grid(
        level=0,
        left_edge=ds.domain_left_edge,
        dims=ds.domain_dimensions,
    )
    # yt stores covering-grid data as x-y-z. Transpose the z=0 plane so rows
    # correspond to y and columns correspond to x for pcolormesh.
    pressure = np.asarray(grid[pressure_field].to_value())[:, :, 0].T
    bmag = np.asarray(grid[bmag_field].to_value())[:, :, 0].T
    pressure_rot180_abs = np.abs(pressure - pressure[::-1, ::-1])

    left = ds.domain_left_edge.to_value("code_length")
    right = ds.domain_right_edge.to_value("code_length")
    x_edges = np.linspace(left[0], right[0], dimensions[0] + 1)
    y_edges = np.linspace(left[1], right[1], dimensions[1] + 1)
    del grid
    del ds
    gc.collect()
    return pressure, bmag, pressure_rot180_abs, time, x_edges, y_edges


vmin_bmag, vmax_bmag = np.inf, -np.inf
vmin_pressure, vmax_pressure = np.inf, -np.inf
vmin_pressure_rot180_abs, vmax_pressure_rot180_abs = np.inf, -np.inf
for snapshot in snapshots:
    pressure, bmag, pressure_rot180_abs, _, _, _ = read_fields(snapshot)
    vmin_bmag = min(vmin_bmag, float(bmag.min()))
    vmax_bmag = max(vmax_bmag, float(bmag.max()))
    vmin_pressure = min(vmin_pressure, float(pressure.min()))
    vmax_pressure = max(vmax_pressure, float(pressure.max()))
    vmin_pressure_rot180_abs = min(
        vmin_pressure_rot180_abs, float(pressure_rot180_abs.min())
    )
    vmax_pressure_rot180_abs = max(
        vmax_pressure_rot180_abs, float(pressure_rot180_abs.max())
    )


def make_gif(quantity_name, label, cmap, vmin, vmax, getter, suffix):
    first_pressure, first_bmag, first_diff, _, x_edges, y_edges = read_fields(
        snapshots[0]
    )
    data = getter(first_pressure, first_bmag, first_diff)

    x_range = x_edges[-1] - x_edges[0]
    y_range = y_edges[-1] - y_edges[0]
    fig, ax = plt.subplots(figsize=(6 * x_range / y_range, 6))
    ax.set_aspect("equal")

    image = ax.pcolormesh(
        x_edges, y_edges, data, cmap=cmap, vmin=vmin, vmax=vmax, shading="flat"
    )
    fig.colorbar(image, ax=ax, label=label)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    base_title = (
        f"Orszag-Tang Vortex ({n_label})" if n_label else "Orszag-Tang Vortex"
    )

    def update(snapshot):
        pressure, bmag, pressure_rot180_abs, time, _, _ = read_fields(snapshot)
        image.set_array(getter(pressure, bmag, pressure_rot180_abs).ravel())
        ax.set_title(f"{base_title}: {quantity_name}\nt = {time:.3f}")
        return [image]

    movie = animation.FuncAnimation(
        fig, update, frames=snapshots, blit=True, cache_frame_data=False
    )
    gif_output = output.with_name(f"{output.name}_{suffix}.gif")
    gif_output.parent.mkdir(parents=True, exist_ok=True)
    movie.save(gif_output, writer="pillow", fps=args.fps)
    plt.close(fig)
    print(f"saved gif: {gif_output}")


if not args.final_only:
    make_gif(
        "pressure",
        "pressure",
        "viridis",
        vmin_pressure,
        vmax_pressure,
        lambda pressure, bmag, pressure_rot180_abs: pressure,
        "pressure",
    )
    make_gif(
        "|B|",
        "|B|",
        "inferno",
        vmin_bmag,
        vmax_bmag,
        lambda pressure, bmag, pressure_rot180_abs: bmag,
        "Bmag",
    )
    make_gif(
        "|P - rot180(P)|",
        "|P - rot180(P)|",
        "magma",
        vmin_pressure_rot180_abs,
        vmax_pressure_rot180_abs,
        lambda pressure, bmag, pressure_rot180_abs: pressure_rot180_abs,
        "pressure_rot180_abs",
    )


if args.pressure_final:
    pressure, _, _, final_time, x_edges, y_edges = read_fields(snapshots[-1])
    x_range = x_edges[-1] - x_edges[0]
    y_range = y_edges[-1] - y_edges[0]
    fig, ax = plt.subplots(figsize=(6 * x_range / y_range, 6))
    ax.set_aspect("equal")
    image = ax.pcolormesh(
        x_edges,
        y_edges,
        pressure,
        cmap="viridis",
        vmin=vmin_pressure,
        vmax=vmax_pressure,
        shading="flat",
    )
    fig.colorbar(image, ax=ax, label="pressure")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    base_title = (
        f"Orszag-Tang Vortex ({n_label})" if n_label else "Orszag-Tang Vortex"
    )
    ax.set_title(f"{base_title}: pressure\nt = {final_time:.3f}")
    final_output = Path(args.pressure_final).expanduser()
    final_output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(final_output, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved final pressure plot: {final_output}")
