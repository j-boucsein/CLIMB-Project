#!/usr/bin/env python3
"""Light-cone picture of the spectra stitching (companion figure to augmented_spectra.pdf).

Ten simulation boxes -- one per short spectrum in augmented_spectra.pdf, same order, redshifts
and colours -- are glued together along the line of sight, each with a random transverse
offset, and a single straight ray is traced through all of them.

The 3D scene is drawn with a parallel (orthographic) projection and the painter's algorithm
instead of mplot3d, so the result is a clean vector PDF (only the particle clouds are rasterised).

Usage:  python lightcone_boxes.py   ->  lightcone_boxes.pdf / lightcone_boxes.png
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection, PolyCollection

OUT = Path(__file__).resolve().parent / "lightcone_boxes"

# ----------------------------------------------------------------------------------------------
# configuration
# ----------------------------------------------------------------------------------------------
# One entry per short spectrum of augmented_spectra.pdf, left to right. The colours are the ones
# used in that PDF (Set1, with a lighter shade when a snapshot is re-used with a new sightline).
REDSHIFTS = [2.1, 2.2, 2.3, 2.3, 2.4, 2.4, 2.5, 2.6, 2.6, 2.7]
COLORS = ["#e41a1c", "#377eb8", "#984ea3", "#c18ec9", "#ff7f00",
          "#ffb266", "#a65628", "#f781bf", "#fab7da", "#999999"]

SEED = 2                  # random transverse offsets of the boxes
HIT_RANGE = (0.2, 0.8)    # where the ray may pierce a box, as a fraction of the box size
GAP = 0.0                # space between neighbouring boxes in box sizes (0 = boxes touch)
RAY_OVERHANG = (0.9, 1.3) # how far the ray sticks out of the first / last box

SHOW_PARTICLES = True     # fill the boxes with a toy cosmic web (Zel'dovich particles)
SHOW_ENDPOINTS = True     # quasar at the far end, arrow towards the observer at the near end
SHOW_HIDDEN_EDGES = True  # dashed back edges, and dashed edge parts hidden behind a nearer box

# Viewing direction (parallel projection, so all boxes have the same size). x runs along the
# ray, y is depth, z is up. AZIM = 0 looks straight at the long side of the stack, larger values
# turn the view towards the first box; ELEV tilts it to look down onto the boxes. The image is
# rolled afterwards so that the ray is horizontal, like the wavelength axis.
AZIM = 30.0
ELEV = 20.0

FIGSIZE = (12.0, 3.4)
FACE_ALPHA = 0.13
EDGE_COLOR = "0.15"
RAY_COLOR = "#ff7f00"     # Set1 orange
RAY_WIDTH = 1.8
LABEL_SIZE = 10


# ----------------------------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------------------------
# Unit cube: vertex index = 4*i + 2*j + k for the corner (i, j, k).
CUBE = np.array([[i, j, k] for i in (0, 1) for j in (0, 1) for k in (0, 1)], dtype=float)
FACES = {  # outward normal -> vertex indices in cyclic order
    (-1, 0, 0): [0, 1, 3, 2], (1, 0, 0): [4, 6, 7, 5],
    (0, -1, 0): [0, 4, 5, 1], (0, 1, 0): [2, 3, 7, 6],
    (0, 0, -1): [0, 2, 6, 4], (0, 0, 1): [1, 5, 7, 3],
}
EDGES = [(a, b) for a in range(8) for b in range(a + 1, 8)
         if np.abs(CUBE[a] - CUBE[b]).sum() == 1]
EDGE_SAMPLES = 1000  # resolution of the solid/dashed split along an edge partly behind a box


class Camera:
    """Parallel projection seen from direction (azim, elev), followed by a roll in the image plane."""

    def __init__(self, azim, elev, up=(0.0, 0.0, 1.0)):
        a, e = np.radians([azim, elev])
        self.back = np.array([-np.sin(a) * np.cos(e), -np.cos(a) * np.cos(e), np.sin(e)])
        right = np.cross(-self.back, up)
        right /= np.linalg.norm(right)
        self.rot = np.stack([right, np.cross(right, -self.back)])
        self.roll = np.eye(2)

    def __call__(self, points):
        return np.asarray(points, dtype=float) @ self.rot.T @ self.roll.T

    def depth(self, points):
        """Distance towards the viewer (larger = closer)."""
        return np.asarray(points, dtype=float) @ self.back

    def level(self, a, b):
        """Roll the image so that the projected segment a -> b points to the right."""
        d = self(b) - self(a)
        phi = -np.arctan2(d[1], d[0])
        self.roll = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]]) @ self.roll

    def facing(self, normal):
        return np.dot(normal, self.back) > 0

    def occluded(self, points, boxes_lo, eps=1e-6):
        """True where the sight line from a point towards the viewer passes through one of the
        unit boxes with lower corners `boxes_lo` (slab test along point + t * back, t > 0)."""
        hidden = np.zeros(len(points), dtype=bool)
        with np.errstate(divide="ignore", invalid="ignore"):
            for lo in boxes_lo:
                t0, t1 = (lo - points) / self.back, (lo + 1 - points) / self.back
                t_in = np.nanmax(np.minimum(t0, t1), axis=1)
                t_out = np.nanmin(np.maximum(t0, t1), axis=1)
                hidden |= (t_out - t_in > eps) & (t_out > eps)
        return hidden


def runs(mask):
    """(start, stop, value) of the runs of equal values in a boolean array."""
    cuts = np.flatnonzero(np.diff(mask)) + 1
    starts, stops = np.r_[0, cuts], np.r_[cuts, len(mask)]
    return zip(starts, stops, mask[starts])


def toy_cosmic_web(n_grid=32, n_s=-1.5, k_cut=30.0, seed=7):
    """Lagrangian positions and Zel'dovich displacements (normalised to unit rms density)."""
    rng = np.random.default_rng(seed)
    k1 = 2 * np.pi * np.fft.fftfreq(n_grid, d=1.0 / n_grid)
    kvec = np.meshgrid(k1, k1, k1, indexing="ij")
    k2 = sum(k ** 2 for k in kvec)
    k2[0, 0, 0] = 1.0
    amp = np.sqrt(k2 ** (n_s / 2) * np.exp(-k2 / k_cut ** 2))
    amp[0, 0, 0] = 0.0
    delta_k = np.fft.fftn(rng.standard_normal((n_grid,) * 3)) * amp
    norm = np.real(np.fft.ifftn(delta_k)).std()
    psi = np.stack([np.real(np.fft.ifftn(1j * k / k2 * delta_k)) for k in kvec]) / norm
    # jitter the lattice a little so it does not show up as a moire pattern in the voids
    q = (np.indices((n_grid,) * 3).reshape(3, -1).T + rng.uniform(0, 1, (n_grid ** 3, 3))) / n_grid
    return q, psi.reshape(3, -1).T


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------
def setup_style():
    have_roboto = any(f.name == "Roboto" for f in font_manager.fontManager.ttflist)
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": (["Roboto"] if have_roboto else []) + ["DejaVu Sans"],
        "font.weight": "light" if have_roboto else "normal",
        "pdf.fonttype": 42,
    })


def main():
    setup_style()
    rng = np.random.default_rng(SEED)
    n = len(REDSHIFTS)

    # Box i spans [x_i, x_i + 1] along the ray; its transverse offset is chosen such that the
    # ray (y = z = 0) pierces it at the random fractional position `hit[i]`.
    hit = rng.uniform(*HIT_RANGE, size=(n, 2))
    lo = np.column_stack([np.arange(n) * (1 + GAP), -hit])
    centers = lo + 0.5
    x_end = lo[-1, 0] + 1

    ray_a = np.array([-RAY_OVERHANG[0], 0.0, 0.0])
    ray_b = np.array([x_end + RAY_OVERHANG[1], 0.0, 0.0])
    cam = Camera(AZIM, ELEV)
    cam.level(ray_a, ray_b)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    fig.subplots_adjust(0, 0, 1, 1)
    ax.set_axis_off()
    ax.set_aspect("equal")

    # Painter's algorithm on whole boxes: farthest box first. Every box owns the zorder range
    # [base, base + 10); the ray pieces between boxes are slotted in between those ranges.
    # The particle clouds all go below that (zorder < 1, same order) so they never hide the ray.
    base = np.empty(n)
    base[np.argsort(cam.depth(centers))] = 10 * np.arange(n) + 10

    if SHOW_PARTICLES:
        q, psi = toy_cosmic_web()
        growth = 1.6 * (1 + REDSHIFTS[0]) / (1 + np.asarray(REDSHIFTS))

    all_xy = [cam([ray_a, ray_b])]
    for i in range(n):
        verts = lo[i] + CUBE
        xy = cam(verts)
        all_xy.append(xy)
        front = {nrm: cam.facing(np.array(nrm)) for nrm in FACES}

        for is_front, z in ((False, 0), (True, 6)):
            polys = [xy[idx] for nrm, idx in FACES.items() if front[nrm] == is_front]
            ax.add_collection(PolyCollection(polys, facecolors=COLORS[i], alpha=FACE_ALPHA,
                                             edgecolors="none", zorder=base[i] + z))

        # An edge is hidden (dashed) where it is on the back of its own box or where a nearer
        # box sits between it and the viewer; the rest of it is drawn solid. Back edges lie
        # behind the ray inside the box, the covered front edges in front of it.
        visible, back, covered = [], [], []
        others = np.delete(lo, i, axis=0)
        s = np.linspace(0, 1, EDGE_SAMPLES + 1)
        for a, b in EDGES:
            adjacent = [nrm for nrm, idx in FACES.items() if a in idx and b in idx]
            if not any(front[nrm] for nrm in adjacent):
                back.append(xy[[a, b]])
                continue
            mid = verts[a] + np.outer(0.5 * (s[:-1] + s[1:]), verts[b] - verts[a])
            for start, stop, behind in runs(cam.occluded(mid, others)):
                piece = verts[a] + np.outer(s[[start, stop]], verts[b] - verts[a])
                (covered if behind else visible).append(cam(piece))
        ax.add_collection(LineCollection(visible, colors=EDGE_COLOR, lw=1.0,
                                         capstyle="round", zorder=base[i] + 7))
        if SHOW_HIDDEN_EDGES:
            for edges, z in ((back, 1), (covered, 7)):
                ax.add_collection(LineCollection(edges, colors=EDGE_COLOR, lw=0.5, alpha=0.5,
                                                 linestyles=(0, (3, 3)), zorder=base[i] + z))

        if SHOW_PARTICLES:
            p = cam(lo[i] + (q + growth[i] * psi) % 1.0)
            ax.scatter(p[:, 0], p[:, 1], s=0.15, c=COLORS[i], alpha=0.5, lw=0,
                       rasterized=True, zorder=base[i] / 1000)

        # ray segment inside the box, drawn behind the box's front faces and edges
        seg = cam([[lo[i, 0], 0, 0], [lo[i, 0] + 1, 0, 0]])
        ax.plot(*seg.T, color=RAY_COLOR, lw=RAY_WIDTH, solid_capstyle="butt", zorder=base[i] + 4)

        # redshift label in the lower right corner of the front face (the nearer boxes all lie
        # to the left, so this corner is never covered)
        tx, ty = cam(lo[i] + [0.94, 0.0, 0.06])
        ax.text(tx, ty, f"z={REDSHIFTS[i]:.1f}", fontsize=LABEL_SIZE, ha="right", va="bottom",
                zorder=1000)

    # ray pieces in the gaps go between the zorder ranges of their two neighbours
    for i in range(n - 1):
        seg = cam([[lo[i, 0] + 1, 0, 0], [lo[i + 1, 0], 0, 0]])
        ax.plot(*seg.T, color=RAY_COLOR, lw=RAY_WIDTH, solid_capstyle="butt",
                zorder=min(base[i], base[i + 1]) + 9.5)

    # overhanging ends: in front of the end box if the viewer sees the face they leave through
    for end, box, nrm in ((ray_a, 0, (-1, 0, 0)), (ray_b, n - 1, (1, 0, 0))):
        face = np.array([lo[box, 0] + (nrm[0] > 0), 0, 0])
        z = base[box] + (9.5 if cam.facing(np.array(nrm)) else -0.5)
        seg = cam([face, end])
        if SHOW_ENDPOINTS and box == 0:
            ax.annotate("", xy=seg[1], xytext=seg[0], zorder=z,
                        arrowprops=dict(arrowstyle="-|>", color=RAY_COLOR, lw=RAY_WIDTH,
                                        mutation_scale=12, shrinkA=0, shrinkB=0))
        else:
            ax.plot(*seg.T, color=RAY_COLOR, lw=RAY_WIDTH, solid_capstyle="butt", zorder=z)

    if SHOW_ENDPOINTS:
        ax.plot(*cam(ray_b), marker="*", ms=13, color=RAY_COLOR, mec="w", mew=0.6, zorder=1000)
        for end, label in ((ray_b, "quasar"), (ray_a, "observer")):
            ax.annotate(label, cam(end), xytext=(0, -7), textcoords="offset points",
                        fontsize=LABEL_SIZE, ha="center", va="top")

    all_xy = np.concatenate(all_xy)
    pad = 0.02 * np.ptp(all_xy[:, 0])
    ax.set_xlim(all_xy[:, 0].min() - pad, all_xy[:, 0].max() + pad)
    ax.set_ylim(all_xy[:, 1].min() - pad, all_xy[:, 1].max() + pad)

    for ext in (".pdf", ".png"):
        fig.savefig(OUT.with_suffix(ext), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"saved {OUT}.pdf and {OUT}.png")


if __name__ == "__main__":
    main()
