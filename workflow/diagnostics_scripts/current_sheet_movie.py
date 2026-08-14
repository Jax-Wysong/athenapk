"""Create a By animation from current-sheet PHDF outputs using yt."""

import argparse
import re
from pathlib import Path

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import yt


yt.set_log_level(50)

parser = argparse.ArgumentParser(
    description="Make a GIF from current-sheet PHDF outputs in a directory."
)
parser.add_argument("directory", help="Directory containing the PHDF files")
parser.add_argument(
    "-o", "--output", default="current_sheet", help="Output GIF filename prefix"
)
parser.add_argument("--fps", type=int, default=8, help="Animation frames per second")
args = parser.parse_args()

if args.fps <= 0:
    parser.error("--fps must be positive")

data_dir = Path(args.directory).expanduser()
files = sorted(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

n_match = re.search(r"N(\d+)", files[0].name)
n_label = f"N{n_match.group(1)}" if n_match else None

output = Path(args.output).expanduser()
if not output.is_absolute():
    output = data_dir.parent / output
if n_label:
    output = output.with_name(f"{output.name}_{n_label}")
gif_output = output.with_name(f"{output.name}_By.gif")

field = ("gas", "magnetic_field_y")
title = f"Current Sheet By ({n_label})" if n_label else "Current Sheet By"


def render_frame(filename):
    """Render the complete multi-block xy slice and return its RGBA pixels."""
    ds = yt.load(str(filename))
    slc = yt.SlicePlot(ds, "z", field)
    slc.set_cmap(field, "jet")
    slc.set_log(field, False)
    slc.set_zlim(field, -1.0, 1.0)
    slc.set_colorbar_label(field, r"$B_y$")
    slc.annotate_timestamp()
    slc.annotate_title(title)
    slc.render()

    yt_figure = slc.plots[field].figure
    yt_figure.canvas.draw()
    frame = np.asarray(yt_figure.canvas.buffer_rgba()).copy()
    plt.close(yt_figure)
    return frame


first_frame = render_frame(files[0])
fig, ax = plt.subplots(
    figsize=(first_frame.shape[1] / 100, first_frame.shape[0] / 100)
)
image = ax.imshow(first_frame)
ax.axis("off")
fig.subplots_adjust(left=0, right=1, bottom=0, top=1)


def update(filename):
    image.set_data(render_frame(filename))
    return (image,)


movie = animation.FuncAnimation(fig, update, frames=files, blit=True)
gif_output.parent.mkdir(parents=True, exist_ok=True)
movie.save(str(gif_output), writer="pillow", fps=args.fps)
plt.close(fig)
print(f"saved gif: {gif_output}")
