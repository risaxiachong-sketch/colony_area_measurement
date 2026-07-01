# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class Plate:
    cx: int
    cy: int
    radius: int


@dataclass(frozen=True)
class Measurement:
    image: Path
    plate: Plate
    pixel_mm: float
    plate_area_mm2: float
    colony_px: int
    colony_mm2: float
    colony_percent_plate: float
    mycelium_px: int
    mycelium_mm2: float
    dark_core_px: int
    dark_core_mm2: float
    overlay: Path | None


def read_image(path: Path) -> np.ndarray:
    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, encoded = cv2.imencode(path.suffix or ".png", image)
    if not ok:
        raise ValueError(f"Could not encode image: {path}")
    encoded.tofile(str(path))


def circle_mask(shape: tuple[int, int], cx: int, cy: int, radius: int) -> np.ndarray:
    height, width = shape
    yy, xx = np.ogrid[:height, :width]
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= radius**2


def odd_kernel_size(value: float, minimum: int = 3) -> int:
    size = max(minimum, int(round(value)))
    return size if size % 2 else size + 1


def detect_plate(
    image: np.ndarray,
    min_radius_frac: float,
    max_radius_frac: float,
    hough_param2: float,
) -> Plate:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    scale = min(1.0, 1000.0 / max(height, width))
    small = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    small = cv2.medianBlur(small, 5)

    min_side = min(small.shape[:2])
    min_radius = int(min_side * min_radius_frac)
    max_radius = int(min_side * max_radius_frac)
    if min_radius <= 0 or max_radius <= min_radius:
        raise ValueError("Invalid plate radius range.")

    circles = None
    for param2 in (hough_param2, 35, 28, 22, 18):
        circles = cv2.HoughCircles(
            small,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=small.shape[0] // 3,
            param1=80,
            param2=param2,
            minRadius=min_radius,
            maxRadius=max_radius,
        )
        if circles is not None:
            break

    if circles is None:
        raise ValueError("Could not detect a petri dish circle.")

    edges = cv2.Canny(small, 45, 120)
    candidates: list[tuple[float, int, int, int]] = []
    for x_small, y_small, r_small in np.round(circles[0]).astype(int):
        if y_small < small.shape[0] * 0.35:
            continue

        theta = np.linspace(0, 2 * np.pi, 240, endpoint=False)
        xs = np.clip(np.round(x_small + r_small * np.cos(theta)).astype(int), 0, edges.shape[1] - 1)
        ys = np.clip(np.round(y_small + r_small * np.sin(theta)).astype(int), 0, edges.shape[0] - 1)
        edge_support = float(np.mean(edges[ys, xs] > 0))
        bottom_bonus = 0.15 if y_small > small.shape[0] * 0.5 else 0.0
        radius_bonus = 0.1 * (r_small / max_radius)
        score = edge_support + bottom_bonus + radius_bonus
        candidates.append((score, x_small, y_small, r_small))

    if not candidates:
        raise ValueError("Detected circles, but none looked like the lower petri dish.")

    _, x_small, y_small, r_small = max(candidates, key=lambda item: item[0])
    return Plate(
        cx=int(round(x_small / scale)),
        cy=int(round(y_small / scale)),
        radius=int(round(r_small / scale)),
    )


def fill_holes(mask: np.ndarray) -> np.ndarray:
    foreground = (mask > 0).astype(np.uint8) * 255
    flood = foreground.copy()
    h, w = flood.shape
    border = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flood, border, (0, 0), 255)
    holes = cv2.bitwise_not(flood)
    return cv2.bitwise_or(foreground, holes)


def segment_colony(
    image: np.ndarray,
    plate: Plate,
    inner_plate_frac: float,
    background_inner_frac: float,
    background_outer_frac: float,
    color_delta: float,
    yellow_delta: float,
    dark_delta: float,
    min_component_frac: float,
    seed_radius_frac: float,
    keep_center_frac: float,
    analysis_mode: str = "mixed",
    white_delta: float = 10.0,
    texture_delta: float = 6.0,
    brown_delta: float = 7.0,
    footprint_close_frac: float = 0.08,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    yy, xx = np.ogrid[:height, :width]
    distance = np.sqrt((xx - plate.cx) ** 2 + (yy - plate.cy) ** 2)
    plate_inner = distance <= plate.radius * inner_plate_frac
    background = (
        plate_inner
        & (distance >= plate.radius * background_inner_frac)
        & (distance <= plate.radius * background_outer_frac)
    )
    if np.count_nonzero(background) < 500:
        background = plate_inner

    bg_lab = np.median(lab[background], axis=0)
    bg_hsv = np.median(hsv[background], axis=0)

    lab_float = lab.astype(np.float32)
    delta = np.sqrt(
        ((lab_float[:, :, 0] - bg_lab[0]) * 0.55) ** 2
        + ((lab_float[:, :, 1] - bg_lab[1]) * 1.2) ** 2
        + ((lab_float[:, :, 2] - bg_lab[2]) * 1.2) ** 2
    )

    lightness = lab[:, :, 0].astype(np.int16)
    lab_b = lab[:, :, 2].astype(np.int16)
    saturation = hsv[:, :, 1].astype(np.int16)
    value = hsv[:, :, 2].astype(np.int16)

    dark = (value < bg_hsv[2] - dark_delta) | (lightness < bg_lab[0] - 30)

    local_blur_size = odd_kernel_size(plate.radius * 0.12, minimum=15)
    local_blur = cv2.GaussianBlur(gray, (local_blur_size, local_blur_size), 0)
    local_bright = gray.astype(np.int16) - local_blur.astype(np.int16)

    tophat_size = odd_kernel_size(plate.radius * 0.035, minimum=7)
    tophat_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (tophat_size, tophat_size))
    white_tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, tophat_kernel).astype(np.int16)

    bright_mycelium = (
        ((lightness > bg_lab[0] + white_delta) | (value > bg_hsv[2] + white_delta))
        & ((local_bright > texture_delta) | (white_tophat > texture_delta))
    )

    warm_mycelium = (
        (lab_b > bg_lab[2] + yellow_delta)
        & (saturation > bg_hsv[1] + 4)
        & (value > bg_hsv[2] - 70)
    )
    brown_pigment = (
        (lab_b > bg_lab[2] + brown_delta)
        & (saturation > bg_hsv[1] + 2)
        & (value > bg_hsv[2] - 100)
    )

    mode = analysis_mode.lower()
    if mode == "light_mycelium":
        candidate_bool = plate_inner & (bright_mycelium | dark)
        mycelium_signal = bright_mycelium
        kernel_size = odd_kernel_size(plate.radius * 0.008)
        close_size = odd_kernel_size(plate.radius * 0.018)
        close_iterations = 1
    elif mode in {"pigment", "pigment_included"}:
        candidate_bool = plate_inner & (
            (delta > color_delta)
            | bright_mycelium
            | warm_mycelium
            | brown_pigment
            | dark
        )
        mycelium_signal = bright_mycelium | warm_mycelium | brown_pigment | (delta > color_delta)
        kernel_size = odd_kernel_size(plate.radius * 0.015)
        close_size = odd_kernel_size(plate.radius * 0.035)
        close_iterations = 2
    elif mode in {"footprint", "outer_footprint"}:
        candidate_bool = plate_inner & (
            (delta > color_delta)
            | bright_mycelium
            | warm_mycelium
            | brown_pigment
            | dark
        )
        mycelium_signal = candidate_bool
        kernel_size = odd_kernel_size(plate.radius * 0.012)
        close_size = odd_kernel_size(plate.radius * 0.040)
        close_iterations = 2
    else:
        candidate_bool = plate_inner & ((delta > color_delta) | dark | warm_mycelium)
        mycelium_signal = (
            ((lab_b > bg_lab[2] + yellow_delta) | (saturation > bg_hsv[1] + 4))
            & (value > bg_hsv[2] - 75)
        )
        kernel_size = odd_kernel_size(plate.radius * 0.025)
        close_size = kernel_size
        close_iterations = 2

    candidate = candidate_bool.astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_OPEN, kernel)
    candidate = cv2.morphologyEx(candidate, cv2.MORPH_CLOSE, close_kernel, iterations=close_iterations)

    seed = circle_mask((height, width), plate.cx, plate.cy, int(plate.radius * seed_radius_frac))
    min_area = max(20, int(math.pi * plate.radius**2 * min_component_frac))

    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(candidate, connectivity=8)
    selected = np.zeros((height, width), dtype=np.uint8)
    for label_id in range(1, n_labels):
        area = stats[label_id, cv2.CC_STAT_AREA]
        if area < min_area:
            continue

        component = labels == label_id
        overlap = np.count_nonzero(component & seed)
        cx_component, cy_component = centroids[label_id]
        center_distance = math.hypot(cx_component - plate.cx, cy_component - plate.cy)
        if overlap > 0 or center_distance <= plate.radius * keep_center_frac:
            selected[component] = 255

    selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, close_kernel)

    if mode in {"footprint", "outer_footprint"}:
        footprint_size = odd_kernel_size(plate.radius * footprint_close_frac, minimum=9)
        footprint_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (footprint_size, footprint_size))
        selected = cv2.morphologyEx(selected, cv2.MORPH_CLOSE, footprint_kernel, iterations=2)
        selected = fill_holes(selected)

    selected_bool = selected > 0
    central_core_zone = distance <= plate.radius * max(seed_radius_frac, 0.30)
    dark_core = selected_bool & dark & central_core_zone
    mycelium = selected_bool & ~dark_core & mycelium_signal

    if mode in {"footprint", "outer_footprint"}:
        colony = selected
        mycelium = selected_bool & ~dark_core
        return colony, mycelium.astype(np.uint8) * 255, dark_core.astype(np.uint8) * 255, plate_inner.astype(np.uint8) * 255

    colony_seed = ((mycelium | dark_core).astype(np.uint8)) * 255
    colony_seed = cv2.morphologyEx(colony_seed, cv2.MORPH_CLOSE, close_kernel)
    if mode == "light_mycelium":
        colony = colony_seed
    else:
        colony = fill_holes(colony_seed)

    return colony, mycelium.astype(np.uint8) * 255, dark_core.astype(np.uint8) * 255, plate_inner.astype(np.uint8) * 255


def make_overlay(
    image: np.ndarray,
    plate: Plate,
    colony: np.ndarray,
    mycelium: np.ndarray,
    dark_core: np.ndarray,
    measurement: Measurement,
) -> np.ndarray:
    overlay = image.copy()

    color_layer = np.zeros_like(image)
    color_layer[colony > 0] = (0, 155, 255)
    color_layer[mycelium > 0] = (70, 210, 70)
    color_layer[dark_core > 0] = (220, 70, 220)
    overlay = cv2.addWeighted(overlay, 1.0, color_layer, 0.45, 0)

    cv2.circle(overlay, (plate.cx, plate.cy), plate.radius, (255, 255, 0), 4)
    cv2.circle(overlay, (plate.cx, plate.cy), 4, (255, 255, 0), -1)

    text = (
        f"colony {measurement.colony_mm2:.1f} mm2, "
        f"mycelium {measurement.mycelium_mm2:.1f} mm2"
    )
    cv2.putText(overlay, text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
    cv2.putText(overlay, text, (30, 55), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
    return overlay


def measure_image(path: Path, args: argparse.Namespace, output_dir: Path | None) -> Measurement:
    image = read_image(path)
    if args.plate:
        plate = Plate(cx=args.plate[0], cy=args.plate[1], radius=args.plate[2])
    else:
        plate = detect_plate(
            image=image,
            min_radius_frac=args.min_radius_frac,
            max_radius_frac=args.max_radius_frac,
            hough_param2=args.hough_param2,
        )

    colony, mycelium, dark_core, _ = segment_colony(
        image=image,
        plate=plate,
        inner_plate_frac=args.inner_plate_frac,
        background_inner_frac=args.background_inner_frac,
        background_outer_frac=args.background_outer_frac,
        color_delta=args.color_delta,
        yellow_delta=args.yellow_delta,
        dark_delta=args.dark_delta,
        min_component_frac=args.min_component_frac,
        seed_radius_frac=args.seed_radius_frac,
        keep_center_frac=args.keep_center_frac,
        analysis_mode=getattr(args, "analysis_mode", "mixed"),
        white_delta=getattr(args, "white_delta", 10.0),
        texture_delta=getattr(args, "texture_delta", 6.0),
        brown_delta=getattr(args, "brown_delta", 7.0),
        footprint_close_frac=getattr(args, "footprint_close_frac", 0.08),
    )

    pixel_mm = args.plate_diameter_mm / (2 * plate.radius)
    mm2_per_pixel = pixel_mm**2
    plate_area_mm2 = math.pi * (args.plate_diameter_mm / 2) ** 2
    colony_px = int(np.count_nonzero(colony))
    mycelium_px = int(np.count_nonzero(mycelium))
    dark_core_px = int(np.count_nonzero(dark_core))

    measurement = Measurement(
        image=path,
        plate=plate,
        pixel_mm=pixel_mm,
        plate_area_mm2=plate_area_mm2,
        colony_px=colony_px,
        colony_mm2=colony_px * mm2_per_pixel,
        colony_percent_plate=100.0 * colony_px * mm2_per_pixel / plate_area_mm2,
        mycelium_px=mycelium_px,
        mycelium_mm2=mycelium_px * mm2_per_pixel,
        dark_core_px=dark_core_px,
        dark_core_mm2=dark_core_px * mm2_per_pixel,
        overlay=None,
    )

    overlay_path = None
    if output_dir is not None and not args.no_overlay:
        overlay_path = output_dir / "overlays" / f"{path.stem}_overlay.png"
        overlay = make_overlay(image, plate, colony, mycelium, dark_core, measurement)
        write_image(overlay_path, overlay)
        measurement = Measurement(**{**measurement.__dict__, "overlay": overlay_path})

    return measurement


def collect_images(inputs: list[Path], pattern: str) -> list[Path]:
    images: list[Path] = []
    for item in inputs:
        if item.is_dir():
            images.extend(sorted(p for p in item.glob(pattern) if p.suffix.lower() in IMAGE_EXTENSIONS))
        elif item.is_file():
            images.append(item)
        else:
            raise FileNotFoundError(item)
    return images


def save_csv(path: Path, measurements: list[Measurement]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image",
        "plate_center_x",
        "plate_center_y",
        "plate_radius_px",
        "plate_diameter_px",
        "pixel_mm",
        "plate_area_mm2",
        "colony_px",
        "colony_mm2",
        "colony_percent_plate",
        "mycelium_px",
        "mycelium_mm2",
        "dark_core_px",
        "dark_core_mm2",
        "overlay",
    ]
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for item in measurements:
            writer.writerow(
                {
                    "image": str(item.image),
                    "plate_center_x": item.plate.cx,
                    "plate_center_y": item.plate.cy,
                    "plate_radius_px": item.plate.radius,
                    "plate_diameter_px": item.plate.radius * 2,
                    "pixel_mm": f"{item.pixel_mm:.8f}",
                    "plate_area_mm2": f"{item.plate_area_mm2:.3f}",
                    "colony_px": item.colony_px,
                    "colony_mm2": f"{item.colony_mm2:.3f}",
                    "colony_percent_plate": f"{item.colony_percent_plate:.3f}",
                    "mycelium_px": item.mycelium_px,
                    "mycelium_mm2": f"{item.mycelium_mm2:.3f}",
                    "dark_core_px": item.dark_core_px,
                    "dark_core_mm2": f"{item.dark_core_mm2:.3f}",
                    "overlay": "" if item.overlay is None else str(item.overlay),
                }
            )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure projected fungal colony area on petri dish images.",
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Image files or directories.")
    parser.add_argument("--pattern", default="*", help="Glob pattern when an input is a directory.")
    parser.add_argument("--output", type=Path, default=Path("colony_area_results"), help="Output directory.")
    parser.add_argument("--plate-diameter-mm", type=float, default=90.0, help="Physical petri dish diameter.")
    parser.add_argument("--plate", nargs=3, type=int, metavar=("CX", "CY", "R"), help="Manual plate circle.")
    parser.add_argument("--no-overlay", action="store_true", help="Do not save overlay images.")

    parser.add_argument("--min-radius-frac", type=float, default=0.18, help="Min plate radius as image min-side fraction.")
    parser.add_argument("--max-radius-frac", type=float, default=0.42, help="Max plate radius as image min-side fraction.")
    parser.add_argument("--hough-param2", type=float, default=35.0, help="Hough circle sensitivity.")
    parser.add_argument("--inner-plate-frac", type=float, default=0.82, help="Fraction of plate radius used for analysis.")
    parser.add_argument("--background-inner-frac", type=float, default=0.45, help="Inner background annulus fraction.")
    parser.add_argument("--background-outer-frac", type=float, default=0.75, help="Outer background annulus fraction.")
    parser.add_argument("--color-delta", type=float, default=24.0, help="Lab color distance threshold.")
    parser.add_argument("--yellow-delta", type=float, default=5.0, help="Lab b-channel threshold above agar background.")
    parser.add_argument("--dark-delta", type=float, default=35.0, help="HSV value threshold below agar background.")
    parser.add_argument("--min-component-frac", type=float, default=0.0004, help="Remove components smaller than this plate area fraction.")
    parser.add_argument("--seed-radius-frac", type=float, default=0.22, help="Center seed radius fraction for colony selection.")
    parser.add_argument("--keep-center-frac", type=float, default=0.38, help="Keep components near plate center.")
    parser.add_argument(
        "--analysis-mode",
        choices=["mixed", "light_mycelium", "pigment_included", "footprint"],
        default="mixed",
        help="Segmentation mode for different colony appearances.",
    )
    parser.add_argument("--white-delta", type=float, default=10.0, help="Brightness threshold for white mycelium.")
    parser.add_argument("--texture-delta", type=float, default=6.0, help="Local contrast threshold for filamentous mycelium.")
    parser.add_argument("--brown-delta", type=float, default=7.0, help="Lab b-channel threshold for brown/yellow pigment.")
    parser.add_argument("--footprint-close-frac", type=float, default=0.08, help="Close gaps when measuring colony footprint.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    images = collect_images(args.inputs, args.pattern)
    if not images:
        raise SystemExit("No images found.")

    output_dir = args.output.resolve()
    measurements = []
    for image_path in images:
        try:
            measurement = measure_image(image_path, args, output_dir)
        except Exception as exc:
            print(f"[ERROR] {image_path}: {exc}")
            continue

        measurements.append(measurement)
        print(
            f"{image_path.name}: colony={measurement.colony_mm2:.1f} mm^2 "
            f"({measurement.colony_percent_plate:.1f}% of plate), "
            f"mycelium={measurement.mycelium_mm2:.1f} mm^2, "
            f"dark_core={measurement.dark_core_mm2:.1f} mm^2"
        )

    if measurements:
        csv_path = output_dir / "colony_area_measurements.csv"
        save_csv(csv_path, measurements)
        print(f"Saved CSV: {csv_path}")
        if not args.no_overlay:
            print(f"Saved overlays: {output_dir / 'overlays'}")


if __name__ == "__main__":
    main()
