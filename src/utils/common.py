"""FlashBack: Shared utility functions.

Consolidates duplicated helpers used across detection, visualization, and video modules.
"""

import cv2
import numpy as np
from pathlib import Path


def imread_unicode(path):
    """Read image from Unicode path (cv2.imread fails on Korean paths)."""
    buf = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path, img):
    """Write image to Unicode path."""
    ext = Path(path).suffix
    result, buf = cv2.imencode(ext, img)
    if result:
        buf.tofile(str(path))
        return True
    return False


def fallback_origin_from_text(text):
    """Estimate origin coordinates from text description."""
    text = str(text).lower()
    x, y = 0.5, 0.5
    if "left" in text:
        x = 0.25
    elif "right" in text:
        x = 0.75
    if "top" in text or "upper" in text:
        y = 0.25
    elif "bottom" in text or "lower" in text or "base" in text:
        y = 0.75
    if "center" in text or "middle" in text:
        if "left" not in text and "right" not in text:
            x = 0.5
        if "top" not in text and "bottom" not in text:
            y = 0.5
    return x, y


def generate_fallback_arrows(origin_x, origin_y, spread_direction):
    """Generate spread arrows from text when model doesn't provide coordinates."""
    arrows = []
    sd = str(spread_direction).lower()

    if "up" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": origin_x, "to_y": max(0.0, origin_y - 0.25)})
    if "outward" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": min(1.0, origin_x + 0.2), "to_y": max(0.0, origin_y - 0.15)})
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": max(0.0, origin_x - 0.2), "to_y": max(0.0, origin_y - 0.15)})
    if "left" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": max(0.0, origin_x - 0.25), "to_y": origin_y})
    if "right" in sd:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": min(1.0, origin_x + 0.25), "to_y": origin_y})

    if not arrows:
        arrows.append({"from_x": origin_x, "from_y": origin_y,
                        "to_x": origin_x, "to_y": max(0.0, origin_y - 0.2)})

    return arrows


def generate_spread_offsets(spread_dir):
    """Generate relative spread arrow offsets from origin (pixel-space)."""
    offsets = []
    sd = str(spread_dir).lower()
    if "up" in sd:
        offsets.append((0, -80))
    if "outward" in sd:
        offsets.append((60, -50))
        offsets.append((-60, -50))
    if "left" in sd:
        offsets.append((-80, 0))
    if "right" in sd:
        offsets.append((80, 0))
    if not offsets:
        offsets.append((0, -60))
    return offsets


def compute_dynamic_spread_arrows(prev_gray, cur_gray, origin_x, origin_y, radius=80, n_arrows=3, arrow_length=80):
    """Compute spread arrows from dense optical flow around the origin point.

    Uses Farneback dense optical flow, then filters to only include
    physically plausible fire spread directions (upward/outward).
    Fire spreads upward via convection and outward via heat transfer —
    downward-only arrows are rejected.

    Args:
        prev_gray: Previous frame (grayscale).
        cur_gray: Current frame (grayscale).
        origin_x: Origin x in pixel coordinates.
        origin_y: Origin y in pixel coordinates.
        radius: Radius of the region around origin to analyze.
        n_arrows: Number of dominant direction arrows to return.
        arrow_length: Length of each arrow in pixels.

    Returns:
        List of (dx, dy) offset tuples representing spread directions.
    """
    h, w = cur_gray.shape[:2]
    cx, cy = int(origin_x), int(origin_y)

    # Compute dense optical flow (Farneback)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, cur_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
    )

    # Extract flow in a circular region around the origin
    y_min = max(0, cy - radius)
    y_max = min(h, cy + radius)
    x_min = max(0, cx - radius)
    x_max = min(w, cx + radius)

    if y_max <= y_min or x_max <= x_min:
        return [(0, -60)]

    region_flow = flow[y_min:y_max, x_min:x_max]

    # Create circular mask
    ry, rx = np.ogrid[y_min - cy:y_max - cy, x_min - cx:x_max - cx]
    mask = (rx * rx + ry * ry) <= radius * radius

    # Get flow vectors within the circle
    fx = region_flow[:, :, 0][mask]
    fy = region_flow[:, :, 1][mask]

    if len(fx) == 0:
        return [(0, -60)]

    # Filter: only keep upward or outward motion (fire physics)
    # In image coords, upward = negative y
    # Allow: upward (dy < 0), sideways (any dx with small dy), outward-up
    # Reject: purely downward motion (dy > 0 and |dy| > |dx|)
    upward_or_outward = (fy <= 0) | (np.abs(fx) > np.abs(fy))
    fx = fx[upward_or_outward]
    fy = fy[upward_or_outward]

    if len(fx) == 0:
        return [(0, -60)]

    # Compute flow magnitudes and filter out weak motion
    magnitudes = np.sqrt(fx ** 2 + fy ** 2)
    threshold = np.percentile(magnitudes, 70)  # top 30% of motion
    strong = magnitudes > max(threshold, 0.5)

    if np.sum(strong) < 5:
        return [(0, -60)]

    fx_strong = fx[strong]
    fy_strong = fy[strong]
    mag_strong = magnitudes[strong]

    # Bin flow directions into angular sectors to find dominant directions
    angles = np.arctan2(fy_strong, fx_strong)
    n_bins = max(n_arrows * 2, 8)
    # Only use upper half of angle range (-pi to 0 = upward, with some lateral)
    bin_edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    hist, _ = np.histogram(angles, bins=bin_edges, weights=mag_strong)

    # Find top n_arrows bins
    top_bins = np.argsort(hist)[-n_arrows:]

    arrows = []
    for b in top_bins:
        if hist[b] < 0.1:
            continue
        bin_center = (bin_edges[b] + bin_edges[b + 1]) / 2
        dx = int(arrow_length * np.cos(bin_center))
        dy = int(arrow_length * np.sin(bin_center))
        # Final safety: reject arrows pointing straight down
        if dy > abs(dx):
            continue
        arrows.append((dx, dy))

    if not arrows:
        return [(0, -60)]

    return arrows
