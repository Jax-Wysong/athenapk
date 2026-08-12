"""Render the Berta cloud-shock midplane density evolution with yt."""

import argparse
from pathlib import Path
import sys

import matplotlib
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yt


yt.set_log_level(50)

# Snakemake invokes the configured environment's Python by absolute path without
# necessarily activating that environment. In that case, the environment's bin/
# directory is not on PATH even though ffmpeg is installed beside Python.
environment_ffmpeg = Path(sys.executable).resolve().with_name("ffmpeg")
if environment_ffmpeg.is_file():
    matplotlib.rcParams["animation.ffmpeg_path"] = str(environment_ffmpeg)

if not animation.writers.is_available("ffmpeg"):
    raise RuntimeError(
        "Matplotlib cannot find ffmpeg. Expected it beside the plotting Python at "
        f"{environment_ffmpeg}."
    )

parser = argparse.ArgumentParser(
    description=(
        "Make a logarithmic-density movie and final PNG on the z=0.5 "
        "cloud-shock midplane."
    )
)
parser.add_argument("directory", help="Directory containing PHDF files")
parser.add_argument("--movie", required=True, help="Output MP4 filename")
parser.add_argument("--gif", required=True, help="Output GIF filename")
parser.add_argument("--final", required=True, help="Output final-time PNG filename")
parser.add_argument("--fps", type=int, default=8, help="Movie frames per second")
parser.add_argument(
    "--expected-final-time",
    type=float,
    default=0.06,
    help="Expected time of the final snapshot",
)
parser.add_argument(
    "--time-tolerance",
    type=float,
    default=1.0e-10,
    help="Tolerance for duplicate and final-time comparisons",
)
parser.add_argument("--rho-min", type=float, help="Optional fixed density minimum")
parser.add_argument("--rho-max", type=float, help="Optional fixed density maximum")
args = parser.parse_args()

if args.fps <= 0:
    parser.error("--fps must be positive")
if args.time_tolerance <= 0.0:
    parser.error("--time-tolerance must be positive")
if args.rho_min is not None and args.rho_min <= 0.0:
    parser.error("--rho-min must be positive for logarithmic plotting")
if args.rho_max is not None and args.rho_max <= 0.0:
    parser.error("--rho-max must be positive for logarithmic plotting")

data_dir = Path(args.directory).expanduser()
movie_output = Path(args.movie).expanduser()
gif_output = Path(args.gif).expanduser()
final_output = Path(args.final).expanduser()
files = list(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

field = ("gas", "density")


def physical_time(filename):
    ds = yt.load(str(filename))
    return float(ds.current_time.to_value())


timed_files = sorted((physical_time(filename), filename) for filename in files)

# A regular cadence output and the final-signal output can represent the same time.
# Prefer the explicitly final-labeled file and render that physical time only once.
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
    rho_min = np.inf
    rho_max = -np.inf
    for _, filename in snapshots:
        ds = yt.load(str(filename))
        rho = ds.all_data()[field]
        positive = rho[rho > 0.0]
        if positive.size == 0:
            raise RuntimeError(f"No positive density values in {filename}")
        rho_min = min(rho_min, float(positive.min().to_value()))
        rho_max = max(rho_max, float(positive.max().to_value()))
    return rho_min, rho_max


if args.rho_min is None or args.rho_max is None:
    detected_min, detected_max = find_density_range()
else:
    detected_min, detected_max = args.rho_min, args.rho_max

rho_min = args.rho_min if args.rho_min is not None else detected_min
rho_max = args.rho_max if args.rho_max is not None else detected_max
if rho_max <= rho_min:
    raise RuntimeError(f"Invalid density range [{rho_min}, {rho_max}]")

print(f"Using fixed logarithmic density range [{rho_min}, {rho_max}]")


def make_slice(filename, time):
    ds = yt.load(str(filename))
    slc = yt.SlicePlot(
        ds,
        "z",
        field,
        center=(0.5, 0.5, 0.5),
        width=((1.0, "code_length"), (1.0, "code_length")),
    )
    slc.set_cmap(field, "bwr")
    slc.set_log(field, True)
    slc.set_zlim(field, rho_min, rho_max)
    slc.set_colorbar_label(field, r"$\rho$")
    slc.set_xlabel("x")
    slc.set_ylabel("y")
    slc.annotate_title(
        f"3D Cloud-Shock Interaction: density at z = 0.5, t = {time:.4f}"
    )
    slc.render()
    return slc


def render_frame(snapshot):
    time, filename = snapshot
    slc = make_slice(filename, time)
    yt_figure = slc.plots[field].figure
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

final_slice = make_slice(final_filename, final_time)
final_figure = final_slice.plots[field].figure
final_output.parent.mkdir(parents=True, exist_ok=True)
final_figure.savefig(final_output, dpi=150, bbox_inches="tight")
plt.close(final_figure)
print(f"Saved final image: {final_output}")
