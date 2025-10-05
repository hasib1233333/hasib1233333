#!/usr/bin/env python3
"""
1000x1000 Random-Obstacle Path Planning with No-Overlap Final Path Visualization

- Generates a 1000x1000 occupancy grid with random rectangles as obstacles.
- Ensures start and goal are connected by carving minimal corridors if needed.
- Computes distance transform, medial-axis skeleton, and shortest path over a clearance-aware skeleton graph.
- Prevents extra/overlapping red line artifacts by simplifying the path and breaking long jumps in plotting.
- Saves outputs in ./outputs/: distance_field.png, skeleton.png, final_path.png
"""

import os
import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi
from skimage.morphology import medial_axis, skeletonize
import networkx as nx

# --------------------------- Configuration ---------------------------

GRID_H = 1000
GRID_W = 1000
NUM_RECTS = 600
RECT_H_RANGE = (8, 40)
RECT_W_RANGE = (8, 40)
WALL_THICK = 6
CORRIDOR_WIDTH = 5
RNG_SEED = 42

OUT_DIR = "outputs"
OUT_DISTANCE = "distance_field.png"
OUT_SKELETON = "skeleton.png"
OUT_FINAL = "final_path.png"

@dataclass
class PlannerConfig:
    clearance_weight: float = 2.0
    length_weight: float = 1.0
    connectivity: int = 8
    smooth: bool = True

# --------------------------- Map Generation --------------------------

def generate_random_map(
    h: int,
    w: int,
    num_rects: int,
    rect_h_range: Tuple[int, int],
    rect_w_range: Tuple[int, int],
    wall_thick: int,
    corridor_width: int,
    seed: int
) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    occ = np.ones((h, w), dtype=np.uint8)

    # Outer border as obstacles
    occ[0:wall_thick, :] = 0
    occ[-wall_thick:, :] = 0
    occ[:, 0:wall_thick] = 0
    occ[:, -wall_thick:] = 0

    # Random rectangles
    for _ in range(num_rects):
        bh = int(rng.integers(rect_h_range[0], rect_h_range[1] + 1))
        bw = int(rng.integers(rect_w_range[0], rect_w_range[1] + 1))
        by = int(rng.integers(wall_thick, max(wall_thick + 1, h - bh - wall_thick)))
        bx = int(rng.integers(wall_thick, max(wall_thick + 1, w - bw - wall_thick)))
        occ[by:by + bh, bx:bx + bw] = 0

    # Start/goal
    start = (h - (wall_thick + 20), wall_thick + 20)   # (y, x)
    goal = (wall_thick + 20, w - (wall_thick + 20))    # (y, x)

    # Ensure start/goal patches are free
    sy, sx = start
    gy, gx = goal
    occ[max(0, sy - corridor_width):min(h, sy + corridor_width + 1),
        max(0, sx - corridor_width):min(w, sx + corridor_width + 1)] = 1
    occ[max(0, gy - corridor_width):min(h, gy + corridor_width + 1),
        max(0, gx - corridor_width):min(w, gx + corridor_width + 1)] = 1

    # Connectivity check
    structure = np.ones((3, 3), dtype=np.uint8)
    labeled, _ = ndi.label(occ == 1, structure=structure)
    s_lab = labeled[sy, sx]
    g_lab = labeled[gy, gx]

    if s_lab == 0:
        occ[max(0, sy - corridor_width):min(h, sy + corridor_width + 1),
            max(0, sx - corridor_width):min(w, sx + corridor_width + 1)] = 1
        labeled, _ = ndi.label(occ == 1, structure=structure)
        s_lab = labeled[sy, sx]

    if g_lab == 0:
        occ[max(0, gy - corridor_width):min(h, gy + corridor_width + 1),
            max(0, gx - corridor_width):min(w, gx + corridor_width + 1)] = 1
        labeled, _ = ndi.label(occ == 1, structure=structure)
        g_lab = labeled[gy, gx]

    def carve_corridor(y0, x0, y1, x1):
        if y0 == y1:
            y = y0
            x_min, x_max = sorted([x0, x1])
            occ[max(0, y - corridor_width):min(h, y + corridor_width + 1),
                max(0, x_min - corridor_width):min(w, x_max + corridor_width + 1)] = 1
        elif x0 == x1:
            x = x0
            y_min, y_max = sorted([y0, y1])
            occ[max(0, y_min - corridor_width):min(h, y_max + corridor_width + 1),
                max(0, x - corridor_width):min(w, x + corridor_width + 1)] = 1
        else:
            carve_corridor(y0, x0, y0, x1)
            carve_corridor(y0, x1, y1, x1)

    if s_lab != g_lab:
        carve_corridor(sy, sx, sy, gx)
        carve_corridor(sy, gx, gy, gx)
        labeled, _ = ndi.label(occ == 1, structure=structure)
        s_lab = labeled[sy, sx]
        g_lab = labeled[gy, gx]
        if s_lab != g_lab:
            carve_corridor(sy, sx, gy, sx)
            carve_corridor(gy, sx, gy, gx)

    return occ, start, goal

# --------------------------- Core Computations -----------------------

def compute_distance_transform(occ: np.ndarray) -> np.ndarray:
    return ndi.distance_transform_edt(occ)

def compute_skeleton(occ: np.ndarray, method: str = "medial_axis") -> Tuple[np.ndarray, np.ndarray]:
    if method == "medial_axis":
        skel, dist = medial_axis(occ > 0, return_distance=True)
        return skel.astype(bool), (dist * skel)
    elif method == "skeletonize":
        skel = skeletonize(occ > 0)
        dist = compute_distance_transform(occ)
        return skel.astype(bool), (dist * skel)
    else:
        raise ValueError("Unknown skeleton method")

def neighbors(y: int, x: int, h: int, w: int, conn: int = 8):
    steps4 = [(-1, 0), (0, -1), (0, 1), (1, 0)]
    steps8 = steps4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    steps = steps8 if conn == 8 else steps4
    for dy, dx in steps:
        ny, nx = y + dy, x + dx
        if 0 <= ny < h and 0 <= nx < w:
            yield ny, nx

def build_skeleton_graph(skel: np.ndarray, dist_on_skel: np.ndarray, cfg: PlannerConfig) -> Tuple[nx.Graph, dict]:
    h, w = skel.shape
    G = nx.Graph()
    coords = np.argwhere(skel)
    for (y, x) in coords:
        G.add_node((y, x), clearance=float(dist_on_skel[y, x]))

    eps = 1e-3
    for (y, x) in coords:
        for (ny_, nx_) in neighbors(y, x, h, w, cfg.connectivity):
            if not skel[ny_, nx_]:
                continue
            p = (y, x); q = (ny_, nx_)
            if G.has_edge(p, q):
                continue
            length = math.hypot(ny_ - y, nx_ - x)
            cl = (G.nodes[p]['clearance'] + G.nodes[q]['clearance']) / 2.0
            cost = (cfg.length_weight * length) / (cfg.clearance_weight * cl + eps)
            G.add_edge(p, q, length=length, clearance=cl, cost=cost)
    return G, {"eps": eps}

def closest_skeleton_node(G: nx.Graph, pt: Tuple[int, int]) -> Tuple[int, int]:
    py, px = pt
    best_node = None
    best_d2 = 1e18
    for (y, x) in G.nodes:
        d2 = (y - py) ** 2 + (x - px) ** 2
        if d2 < best_d2:
            best_d2 = d2
            best_node = (y, x)
    return best_node

def shortest_path_on_skeleton(G: nx.Graph, start_px: Tuple[int, int], goal_px: Tuple[int, int]) -> List[Tuple[int, int]]:
    return nx.shortest_path(G, source=start_px, target=goal_px, weight='cost', method='dijkstra')

# --------------------------- Path Post-processing --------------------

def smooth_path_reflect(path: List[Tuple[int, int]], k: int = 7) -> List[Tuple[float, float]]:
    """
    Moving average with reflection padding to avoid edge jumps that can draw stray long lines.
    """
    if len(path) <= 2 or k < 3 or k % 2 == 0:
        return [(float(y), float(x)) for (y, x) in path]
    ys = np.array([p[0] for p in path], dtype=float)
    xs = np.array([p[1] for p in path], dtype=float)
    pad = k // 2
    ys_pad = np.pad(ys, pad_width=pad, mode='reflect')
    xs_pad = np.pad(xs, pad_width=pad, mode='reflect')
    kernel = np.ones(k, dtype=float) / k
    ys_s = np.convolve(ys_pad, kernel, mode='valid')
    xs_s = np.convolve(xs_pad, kernel, mode='valid')
    return list(zip(ys_s, xs_s))

def simplify_path(path: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    """
    Remove consecutive duplicates, immediate backtracks, and colinear middle points.
    """
    if not path:
        return path
    out: List[Tuple[float, float]] = []
    for p in path:
        if not out or (p[0] != out[-1][0] or p[1] != out[-1][1]):
            out.append(p)
    i = 2
    while i < len(out):
        if out[i][0] == out[i - 2][0] and out[i][1] == out[i - 2][1]:
            del out[i - 1]
            del out[i - 1]
            i = max(2, i - 1)
        else:
            i += 1
    def colinear(a, b, c) -> bool:
        return (b[1] - a[1]) * (c[0] - a[0]) == (b[0] - a[0]) * (c[1] - a[1])
    j = 1
    while j < len(out) - 1:
        if colinear(out[j - 1], out[j], out[j + 1]):
            del out[j]
        else:
            j += 1
    return out

def break_long_jumps_for_plot(path: List[Tuple[float, float]], max_jump: float = 2.1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Insert NaNs between points where the Euclidean step exceeds max_jump.
    This prevents Matplotlib from drawing a long straight line across the figure.
    For 8-connectivity, typical step <= sqrt(2) ~ 1.41, so 2.1 is safe.
    """
    if len(path) == 0:
        return np.array([]), np.array([])
    px, py = [], []
    for i, (y, x) in enumerate(path):
        if i > 0:
            y0, x0 = path[i - 1]
            if math.hypot(y - y0, x - x0) > max_jump:
                px.append(np.nan)
                py.append(np.nan)
        py.append(y)
        px.append(x)
    return np.array(px), np.array(py)

# --------------------------- Visualization --------------------------

def plot_distance_field(dist: np.ndarray, occ: np.ndarray, out_path: str):
    plt.figure(figsize=(7, 7))
    plt.imshow(dist, cmap='viridis')
    plt.title("Distance Transform (Wave Propagation)")
    plt.colorbar(label="Clearance (px)")
    yy, xx = np.where(occ == 0)
    plt.scatter(xx, yy, s=1, c='k', marker='s', label='Obstacles')
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_skeleton(occ: np.ndarray, skel: np.ndarray, out_path: str):
    plt.figure(figsize=(7, 7))
    plt.imshow(occ == 1, cmap='gray')
    yy, xx = np.where(skel)
    plt.scatter(xx, yy, s=1, c='red', marker='s', label='Skeleton')
    plt.title("Medial-Axis Skeleton over Free Space")
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def plot_final_path(dist: np.ndarray, skel: np.ndarray, path_xy: List[Tuple[int, int]],
                    start: Tuple[int, int], goal: Tuple[int, int], out_path: str):
    plt.figure(figsize=(7, 7))
    plt.imshow(dist, cmap='viridis')
    yy, xx = np.where(skel)
    plt.scatter(xx, yy, s=1, c='white', marker='s', alpha=0.6, label='Skeleton')

    # 1) Smooth with reflection padding (avoids edge jumps)
    path_f = smooth_path_reflect(path_xy, k=7)

    # 2) Simplify to avoid overlaps
    path_f = simplify_path(path_f)

    # 3) Break long jumps so Matplotlib won't draw straight lines across the figure
    px, py = break_long_jumps_for_plot(path_f, max_jump=2.1)

    # 4) Draw; NaNs will split segments; no extra line possible
    plt.plot(px, py, c='red', linewidth=2.0, solid_capstyle='round', label='Final Path', zorder=5)

    plt.scatter([start[1]], [start[0]], c='lime', s=40, marker='o', label='Start', zorder=6)
    plt.scatter([goal[1]], [goal[0]], c='cyan', s=40, marker='x', label='Goal', zorder=6)
    plt.title("Random 1000x1000 Path on Medial-Axis Skeleton")
    plt.legend(loc='lower right', fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# --------------------------- Main --------------------------

def main():
    cfg = PlannerConfig(clearance_weight=2.0, length_weight=1.0, connectivity=8, smooth=True)

    out_dir = os.path.abspath(OUT_DIR)
    os.makedirs(out_dir, exist_ok=True)
    print(f"Saving outputs to: {out_dir}")

    # 1) Map
    occ, start, goal = generate_random_map(
        h=GRID_H, w=GRID_W,
        num_rects=NUM_RECTS,
        rect_h_range=RECT_H_RANGE, rect_w_range=RECT_W_RANGE,
        wall_thick=WALL_THICK, corridor_width=CORRIDOR_WIDTH,
        seed=RNG_SEED,
    )

    # 2) Distance transform
    dist = compute_distance_transform(occ)
    plot_distance_field(dist, occ, out_path=os.path.join(out_dir, OUT_DISTANCE))

    # 3) Skeleton
    skel, dist_on_skel = compute_skeleton(occ, method="medial_axis")
    plot_skeleton(occ, skel, out_path=os.path.join(out_dir, OUT_SKELETON))

    # 4) Graph + path
    G, _ = build_skeleton_graph(skel, dist_on_skel, cfg)
    s_node = closest_skeleton_node(G, start)
    g_node = closest_skeleton_node(G, goal)

    try:
        path_nodes = shortest_path_on_skeleton(G, s_node, g_node)
    except nx.NetworkXNoPath:
        print("No skeleton path between start and goal. Try different seed or parameters.")
        print(f"Saved: {os.path.join(out_dir, OUT_DISTANCE)}")
        print(f"Saved: {os.path.join(out_dir, OUT_SKELETON)}")
        return

    path = path_nodes  # list of (y, x)

    # 5) Final plot with no extra/overlap red line
    plot_final_path(dist, skel, path, start, goal, out_path=os.path.join(out_dir, OUT_FINAL))

    # Console summary
    total_len = 0.0
    for i in range(1, len(path)):
        y0, x0 = path[i-1]
        y1, x1 = path[i]
        total_len += math.hypot(y1 - y0, x1 - x0)
    print(f"Nodes: {G.number_of_nodes()}, Edges: {G.number_of_edges()}")
    print(f"Path length (px): {total_len:.2f}, nodes: {len(path)}")
    print("Saved:")
    print(f"  {os.path.join(out_dir, OUT_DISTANCE)}")
    print(f"  {os.path.join(out_dir, OUT_SKELETON)}")
    print(f"  {os.path.join(out_dir, OUT_FINAL)}")

if __name__ == "__main__":
    main()