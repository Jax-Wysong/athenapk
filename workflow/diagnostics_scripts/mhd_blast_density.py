"""Render Berta's 3D C1 MHD blast diagnostics on the z=0 plane."""

import argparse
from pathlib import Path
import sys

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yt
from unyt.dimensions import specific_energy


yt.set_log_level(50)

# Snakemake invokes the configured environment's Python by absolute path without
# necessarily activating that environment. Locate the environment's ffmpeg directly.
environment_ffmpeg = Path(sys.executable).resolve().with_name("ffmpeg")
if environment_ffmpeg.is_file():
    matplotlib.rcParams["animation.ffmpeg_path"] = str(environment_ffmpeg)

if not animation.writers.is_available("ffmpeg"):
    raise RuntimeError(
        "Matplotlib cannot find ffmpeg. Expected it beside the plotting Python at "
        f"{environment_ffmpeg}."
    )

parser = argparse.ArgumentParser(
    description="Make z=0 movie and final diagnostics for Berta's 3D C1 blast."
)
parser.add_argument("directory", help="Directory containing PHDF files")
parser.add_argument("--movie", required=True, help="Output MP4 filename")
parser.add_argument("--gif", required=True, help="Output GIF filename")
parser.add_argument("--final", required=True, help="Output final-time PNG filename")
parser.add_argument(
    "--pressure-final", required=True, help="Output final thermal-pressure PNG"
)
parser.add_argument(
    "--kinetic-final", required=True, help="Output final specific-kinetic-energy PNG"
)
parser.add_argument(
    "--magnetic-final", required=True, help="Output final magnetic-energy PNG"
)
parser.add_argument("--fps", type=int, default=8, help="Movie frames per second")
parser.add_argument(
    "--expected-final-time",
    type=float,
    default=0.2,
    help="Expected time of the final snapshot",
)
parser.add_argument(
    "--time-tolerance",
    type=float,
    default=1.0e-10,
    help="Tolerance for duplicate and final-time comparisons",
)
parser.add_argument(
    "--density-min",
    type=float,
    default=0.0,
    help="Density minimum (default: 0.0)",
)
parser.add_argument(
    "--density-max",
    type=float,
    default=3.0,
    help="Density maximum (default: 3.0)",
)
args = parser.parse_args()

if args.fps <= 0:
    parser.error("--fps must be positive")
if args.time_tolerance <= 0.0:
    parser.error("--time-tolerance must be positive")
if args.density_min is not None and args.density_min < 0.0:
    parser.error("--density-min must be nonnegative")
if args.density_max is not None and args.density_max < 0.0:
    parser.error("--density-max must be nonnegative")

data_dir = Path(args.directory).expanduser()
movie_output = Path(args.movie).expanduser()
gif_output = Path(args.gif).expanduser()
final_output = Path(args.final).expanduser()
pressure_final_output = Path(args.pressure_final).expanduser()
kinetic_final_output = Path(args.kinetic_final).expanduser()
magnetic_final_output = Path(args.magnetic_final).expanduser()
files = list(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

density_field = ("gas", "density")
pressure_field = ("gas", "pressure")
kinetic_field = ("gas", "specific_kinetic_energy")
magnetic_field = ("gas", "magnetic_energy_density")


def _specific_kinetic_energy(field, data):
    del field
    return data[("gas", "kinetic_energy_density")] / data[("gas", "density")]


def load_dataset(filename):
    ds = yt.load(str(filename))
    if kinetic_field not in ds.derived_field_list:
        ds.add_field(
            kinetic_field,
            function=_specific_kinetic_energy,
            sampling_type="cell",
            units="auto",
            dimensions=specific_energy,
        )
    return ds


def physical_time(filename):
    ds = load_dataset(filename)
    return float(ds.current_time.to_value())


timed_files = sorted((physical_time(filename), filename) for filename in files)

# A regular-cadence output and the final-signal output can represent the same time.
# Prefer the explicitly final-labeled file and render each physical time only once.
snapshots = []
for time, filename in timed_files:
    if snapshots and abs(time - snapshots[-1][0]) <= args.time_tolerance:
        if "final" in filename.name.lower():
            snapshots[-1] = (time, filename)
    else:
        snapshots.append((time, filename))

final_time, final_filename = snapshots[-1]
if abs(final_time - args.expected_final_time) > args.time_tolerance:
    raise RuntimeError(
        f"Latest PHDF time is {final_time:.16g}, expected "
        f"{args.expected_final_time:.16g}"
    )


def find_density_range():
    density_min = np.inf
    density_max = -np.inf
    for _, filename in snapshots:
        ds = load_dataset(filename)
        density = ds.all_data()[density_field]
        density_min = min(density_min, float(density.min().to_value()))
        density_max = max(density_max, float(density.max().to_value()))
    return density_min, density_max


if args.density_min is None or args.density_max is None:
    detected_min, detected_max = find_density_range()
else:
    detected_min, detected_max = args.density_min, args.density_max

density_min = (
    args.density_min if args.density_min is not None else detected_min
)
density_max = (
    args.density_max if args.density_max is not None else detected_max
)
if density_max <= density_min:
    raise RuntimeError(f"Invalid density range [{density_min}, {density_max}]")

print(f"Using fixed linear density range [{density_min}, {density_max}]")


def make_slice(
    filename,
    time,
    field,
    *,
    cmap,
    log_scale,
    colorbar_label,
    quantity_name,
    zlim=None,
):
    ds = load_dataset(filename)
    slc = yt.SlicePlot(
        ds,
        "z",
        field,
        center=(0.0, 0.0, 0.0),
        width=((1.0, "code_length"), (1.0, "code_length")),
    )
    slc.set_cmap(field, cmap)
    slc.set_log(field, log_scale)
    if zlim is not None:
        slc.set_zlim(field, zlim[0], zlim[1])
    slc.set_colorbar_label(field, colorbar_label)
    slc.set_xlabel("x")
    slc.set_ylabel("y")
    slc.annotate_title(
        f"Berta C1 3D MHD Blast: {quantity_name} at z = 0, t = {time:.4f}"
    )
    slc.render()
    return slc


def render_frame(snapshot):
    time, filename = snapshot
    slc = make_slice(
        filename,
        time,
        density_field,
        cmap="viridis",
        log_scale=False,
        colorbar_label=r"$\rho$",
        quantity_name="density",
        zlim=(density_min, density_max),
    )
    yt_figure = slc.plots[density_field].figure
    yt_figure.canvas.draw()
    frame = np.asarray(yt_figure.canvas.buffer_rgba()).copy()
    plt.close(yt_figure)
    return frame


first_frame = render_frame(snapshots[0])
fig, ax = plt.subplots(
    figsize=(first_frame.shape[1] / 100.0, first_frame.shape[0] / 100.0)
)
image = ax.imshow(first_frame)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def update(snapshot):
    image.set_data(render_frame(snapshot))
    return (image,)


movie = animation.FuncAnimation(fig, update, frames=snapshots, blit=True)
movie_output.parent.mkdir(parents=True, exist_ok=True)
movie.save(str(movie_output), writer="ffmpeg", fps=args.fps)
print(f"Saved movie: {movie_output}")

gif_output.parent.mkdir(parents=True, exist_ok=True)
movie.save(str(gif_output), writer="pillow", fps=args.fps)
plt.close(fig)
print(f"Saved GIF: {gif_output}")

final_diagnostics = [
    (
        density_field,
        "viridis",
        False,
        r"$\rho$",
        "density",
        (density_min, density_max),
        final_output,
    ),
    (
        pressure_field,
        "inferno",
        True,
        r"$p$",
        "thermal pressure",
        None,
        pressure_final_output,
    ),
    (
        kinetic_field,
        "viridis",
        False,
        r"$v^2/2$",
        "specific kinetic energy",
        None,
        kinetic_final_output,
    ),
    (
        magnetic_field,
        "viridis",
        False,
        r"$B^2/2$",
        "magnetic energy density",
        None,
        magnetic_final_output,
    ),
]

for field, cmap, log_scale, label, name, zlim, output in final_diagnostics:
    final_slice = make_slice(
        final_filename,
        final_time,
        field,
        cmap=cmap,
        log_scale=log_scale,
        colorbar_label=label,
        quantity_name=name,
        zlim=zlim,
    )
    final_figure = final_slice.plots[field].figure
    output.parent.mkdir(parents=True, exist_ok=True)
    final_figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(final_figure)
    print(f"Saved final image: {output}")
