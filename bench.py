"""Simulated focusing bench: DMD (amplitude only) -> scattering medium T -> camera.
E_out = |T E_in|^2 plus noise. The tool run_focusing() runs the macropixel flipping search."""
import csv, json, time
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DMD_N = 64          # DMD is DMD_N x DMD_N micromirrors
CAM_N = 32          # camera is CAM_N x CAM_N pixels
ALLOWED_MACRO = [1, 2, 4, 8, 16, 32]
MAX_ITERATIONS = 50000
OUT_DIR = Path("bench_results")

_T = None
def _medium(seed: int = 12345):
    """One fixed scattering medium (complex Gaussian transmission matrix), same for every run."""
    global _T
    if _T is None:
        rng = np.random.default_rng(seed)
        n_in, n_out = DMD_N * DMD_N, CAM_N * CAM_N
        T = (rng.standard_normal((n_out, n_in)) + 1j * rng.standard_normal((n_out, n_in))) / np.sqrt(2)
        _T = (T * np.sqrt(2.0 / n_in)).astype(np.complex64)   # a random half-on pattern gives mean intensity ~1
    return _T

def _camera(field, rng, shot=0.05, read=0.02):
    """Camera image of intensity |field|^2 with multiplicative and additive noise."""
    I = np.abs(field) ** 2
    I = I * (1 + shot * rng.standard_normal(I.shape)) + read * rng.standard_normal(I.shape)
    return np.maximum(I, 0.0)

def _regions(focus_size, bg_width, n_bg, center):
    """Boolean masks: the focus square and n_bg angular sectors of the square ring around it."""
    yy, xx = np.mgrid[0:CAM_N, 0:CAM_N]
    cy, cx = center
    half = focus_size / 2.0
    in_focus = (np.abs(yy - cy) < half) & (np.abs(xx - cx) < half)
    outer = (np.abs(yy - cy) < half + bg_width) & (np.abs(xx - cx) < half + bg_width)
    ring = outer & ~in_focus
    ang = (np.arctan2(yy - cy, xx - cx) + np.pi) / (2 * np.pi)          # 0..1
    sector = np.minimum((ang * n_bg).astype(int), n_bg - 1)
    return in_focus, [ring & (sector == k) for k in range(n_bg)]

def _enhancement(img, focus_mask, bg_masks):
    bg = np.mean([img[m].mean() for m in bg_masks if m.any()])
    return float(img[focus_mask].mean() / max(bg, 1e-9))

def run_focusing(macropixel_size: int = 8, focus_size: int = 3, bg_width: int = 3, n_bg: int = 4,
                 iterations: int = 3000, flips: int = 1, seed: int | None = None,
                 focus_row: int | None = None, focus_col: int | None = None, label: str = "") -> dict:
    """Run the macropixel flipping search on the simulated bench and return the enhancement and image paths."""
    if macropixel_size not in ALLOWED_MACRO:
        raise ValueError(f"macropixel_size must be one of {ALLOWED_MACRO}")
    if not 1 <= iterations <= MAX_ITERATIONS:
        raise ValueError(f"iterations must be between 1 and {MAX_ITERATIONS}")
    g = DMD_N // macropixel_size
    n_mac = g * g
    if not 1 <= flips <= n_mac:
        raise ValueError(f"flips must be between 1 and the number of macropixels ({n_mac})")
    if not 1 <= focus_size <= CAM_N // 2 or not 1 <= bg_width <= CAM_N // 4 or not 2 <= n_bg <= 16:
        raise ValueError("need 1<=focus_size<=16, 1<=bg_width<=8, 2<=n_bg<=16")
    center = (CAM_N // 2 if focus_row is None else focus_row, CAM_N // 2 if focus_col is None else focus_col)
    t0 = time.time()
    rng = np.random.default_rng(seed)
    T = _medium()
    # macropixel transmission: sum the DMD pixel columns that belong to each macropixel
    Tm = T.reshape(CAM_N * CAM_N, g, macropixel_size, g, macropixel_size).sum(axis=(2, 4)).reshape(CAM_N * CAM_N, n_mac)
    focus_mask, bg_masks = _regions(focus_size, bg_width, n_bg, center)

    x = (rng.random(n_mac) < 0.5).astype(np.float32)                  # random start pattern, amplitude 0 or 1
    field = Tm @ x
    img0 = _camera(field, rng).reshape(CAM_N, CAM_N)
    cur = _enhancement(img0, focus_mask, bg_masks)
    enh_initial = float(np.mean([_enhancement(_camera(field, rng).reshape(CAM_N, CAM_N), focus_mask, bg_masks) for _ in range(5)]))
    curve = [(0, cur)]
    for it in range(1, iterations + 1):
        idx = rng.choice(n_mac, size=flips, replace=False)
        delta = 1.0 - 2.0 * x[idx]                                    # +1 turns a macropixel on, -1 turns it off
        new_field = field + Tm[:, idx] @ delta
        new = _enhancement(_camera(new_field, rng).reshape(CAM_N, CAM_N), focus_mask, bg_masks)
        if new >= cur:                                                # keep the flip only if the measured enhancement did not drop
            x[idx] += delta
            field, cur = new_field, new
        if it % max(iterations // 100, 1) == 0:
            curve.append((it, cur))
    img1 = _camera(field, rng).reshape(CAM_N, CAM_N)
    enh_final = float(np.mean([_enhancement(_camera(field, rng).reshape(CAM_N, CAM_N), focus_mask, bg_masks) for _ in range(5)]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = OUT_DIR / "experiments.csv"
    run_id = 1 + (sum(1 for _ in open(csv_path)) - 1 if csv_path.exists() else 0)
    fig_path = OUT_DIR / f"run_{run_id:03d}.png"
    fig, ax = plt.subplots(1, 3, figsize=(12.5, 3.6), constrained_layout=True)
    vmax = max(img0.max(), img1.max())
    for a, im, title in ((ax[0], img0, f"before (random), enhancement {enh_initial:.1f}"),
                         (ax[1], img1, f"after, enhancement {enh_final:.1f}")):
        h = a.imshow(im, cmap="inferno", vmin=0, vmax=vmax); a.set_title(title, fontsize=9); a.axis("off")
        a.contour(focus_mask, levels=[0.5], colors="cyan", linewidths=0.8)
    fig.colorbar(h, ax=ax[:2], fraction=0.03, pad=0.02, label="intensity")
    ax[2].plot(*zip(*curve)); ax[2].set_xlabel("iteration"); ax[2].set_ylabel("measured enhancement"); ax[2].set_title("search progress", fontsize=9)
    fig.savefig(fig_path, dpi=130, bbox_inches="tight"); plt.close(fig)

    result = {"run_id": run_id, "label": label, "macropixel_size": macropixel_size, "n_macropixels": n_mac,
              "focus_size": focus_size, "bg_width": bg_width, "n_bg": n_bg, "iterations": iterations,
              "flips": flips, "seed": seed, "enhancement_initial": round(enh_initial, 3),
              "enhancement_final": round(enh_final, 3), "image": str(fig_path), "seconds": round(time.time() - t0, 2)}
    new_file = not csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(result)); 
        if new_file: w.writeheader()
        w.writerow(result)
    return result

if __name__ == "__main__":
    print(json.dumps(run_focusing(), indent=1))
