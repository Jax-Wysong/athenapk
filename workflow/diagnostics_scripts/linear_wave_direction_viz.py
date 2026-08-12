"""Create a density-perturbation animation for a directional linear MHD wave."""

import argparse
from pathlib import Path

import h5py
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np


parser = argparse.ArgumentParser(
    description="Make a density-perturbation GIF from single-block linear-wave PHDF files."
)
parser.add_argument("directory", help="Directory containing the PHDF files")
parser.add_argument(
    "-o",
    "--output",
    default="linear_wave_density.gif",
    help="Output GIF filename",
)
parser.add_argument("--fps", type=int, default=8, help="Animation frames per second")
parser.add_argument("--direction", required=True, help="Direction label for the title")
parser.add_argument("--wave", type=int, default=6, help="MHD characteristic mode")
parser.add_argument(
    "--background-density",
    type=float,
    default=1.0,
    help="Uniform background density to subtract",
)
args = parser.parse_args()

if args.fps <= 0:
    parser.error("--fps must be positive")

data_dir = Path(args.directory).expanduser()
files = sorted(data_dir.glob("*.phdf"))
if not files:
    raise FileNotFoundError(f"No .phdf files found in {data_dir}")

output = Path(args.output).expanduser()
if not output.is_absolute():
    output = data_dir.parent / output
if output.suffix.lower() != ".gif":
    output = output.with_suffix(".gif")

block = 0
density_component = 0
x3_index = 0


def read_density_perturbation(filename):
    with h5py.File(filename, "r") as f:
        if "prim" not in f:
            raise KeyError(
                f"{filename} does not contain 'prim'; output variables must include prim"
            )
        prim = f["prim"]
        if prim.ndim != 5 or prim.shape[0] != 1:
            raise ValueError(
                f"Unexpected prim shape {prim.shape} in {filename}; "
                "the directional visualization expects one MeshBlock"
            )
        density = prim[block, density_component, x3_index, :, :]
        time = f["Info"].attrs["Time"]
    return density - args.background_density, time


with h5py.File(files[0], "r") as f:
    x1 = f["VolumeLocations"]["x"][block, :]
    x2 = f["VolumeLocations"]["y"][block, :]

drho_max = 0.0
for filename in files:
    drho, _ = read_density_perturbation(filename)
    drho_max = max(drho_max, float(np.max(np.abs(drho))))

if drho_max == 0.0:
    raise ValueError("Density perturbation is identically zero in all output files")

drho0, time0 = read_density_perturbation(files[0])

fig, ax = plt.subplots(figsize=(6.5, 6.0))
ax.set_aspect("equal")
image = ax.pcolormesh(
    x1,
    x2,
    drho0,
    shading="nearest",
    cmap="RdBu_r",
    vmin=-drho_max,
    vmax=drho_max,
)
fig.colorbar(image, ax=ax, label=r"$\rho-\rho_0$")
ax.set_xlabel(r"$x_1$")
ax.set_ylabel(r"$x_2$")
base_title = f"Linear MHD wave mode {args.wave}: {args.direction}"
title = ax.set_title(base_title + "\n" + rf"$t={time0:.3f}$")


def update(filename):
    drho, time = read_density_perturbation(filename)
    image.set_array(drho.ravel())
    title.set_text(base_title + "\n" + rf"$t={time:.3f}$")
    return image, title


movie = animation.FuncAnimation(fig, update, frames=files, blit=True)
movie.save(output, writer="pillow", fps=args.fps)
print(f"saved gif: {output}")
plt.close(fig)
