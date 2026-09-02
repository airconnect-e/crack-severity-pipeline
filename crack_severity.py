"""Crack-severity pipeline: YOLO segmentation -> geometry -> millimetres."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class CrackMeasurement:
    image: str
    max_width_px: float
    max_width_mm: float
    severity: str
    mm_per_pixel: float
    crack_area_px: int
    max_point_x: int | None
    max_point_y: int | None
    detections_used: int


def severity_from_mm(width_mm: float) -> str:
    if width_mm <= 3.0:
        return "เล็กน้อย"
    if width_mm <= 6.0:
        return "ปานกลาง"
    return "สูง"


def resolve_scale(args: argparse.Namespace, image_shape: tuple[int, ...]) -> float:
    """Return millimetres per pixel; never silently invent a physical scale."""
    if args.mm_per_pixel is not None:
        return args.mm_per_pixel
    if args.reference_mm is not None and args.reference_pixels is not None:
        return args.reference_mm / args.reference_pixels
    if args.image_width_mm is not None:
        return args.image_width_mm / image_shape[1]
    raise ValueError(
        "ต้องระบุสเกล: --mm-per-pixel หรือ --reference-mm พร้อม "
        "--reference-pixels หรือ --image-width-mm"
    )


def select_class_ids(names: dict[int, str], requested: str) -> set[int]:
    if requested.lower() == "all":
        return set(names)
    wanted = {x.strip().lower() for x in requested.split(",") if x.strip()}
    selected = {i for i, name in names.items() if str(name).lower() in wanted}
    if not selected:
        available = ", ".join(f"{i}:{name}" for i, name in names.items())
        raise ValueError(f"ไม่พบ class '{requested}' ในโมเดล (มี: {available})")
    return selected


def build_crack_mask(result: Any, shape: tuple[int, int], class_ids: set[int], threshold: float) -> tuple[np.ndarray, int]:
    union = np.zeros(shape, dtype=np.uint8)
    if result.masks is None or result.boxes is None:
        return union, 0
    masks = result.masks.data.detach().cpu().numpy()
    classes = result.boxes.cls.detach().cpu().numpy().astype(int)
    used = 0
    for mask, class_id in zip(masks, classes):
        if class_id not in class_ids:
            continue
        # Some Ultralytics versions expose probabilities, others binary masks.
        binary = (mask >= threshold).astype(np.uint8)
        if binary.shape != shape:
            binary = cv2.resize(binary, (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST)
        union |= binary
        used += 1
    return union, used


def measure_mask(mask: np.ndarray) -> tuple[float, np.ndarray, tuple[int, int] | None]:
    if not np.any(mask):
        return 0.0, np.zeros_like(mask), None
    # Morphological skeletonization using OpenCV only. Each iteration peels one
    # layer and retains the centre residue; suitable for arbitrary crack shapes.
    working = mask.astype(np.uint8).copy()
    skeleton = np.zeros_like(working)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while np.any(working):
        eroded = cv2.erode(working, element)
        opened = cv2.dilate(eroded, element)
        skeleton |= working & cv2.bitwise_not(opened)
        working = eroded
    distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    radii = np.where(skeleton > 0, distance, 0.0)
    flat_index = int(np.argmax(radii))
    y, x = np.unravel_index(flat_index, radii.shape)
    return float(2.0 * radii[y, x]), skeleton, (int(x), int(y))


def draw_overlay(image: np.ndarray, mask: np.ndarray, skeleton: np.ndarray, point: tuple[int, int] | None,
                 measurement: CrackMeasurement) -> np.ndarray:
    out = image.copy()
    tint = np.zeros_like(out)
    tint[mask > 0] = (0, 0, 255)
    out = cv2.addWeighted(out, 1.0, tint, 0.38, 0)
    out[skeleton > 0] = (0, 255, 255)
    if point is not None:
        cv2.circle(out, point, 7, (255, 0, 255), 2)
    # OpenCV's built-in font is ASCII-only; keep Thai severity in JSON/CSV/console.
    severity_en = {"เล็กน้อย": "Low", "ปานกลาง": "Moderate", "สูง": "High"}[measurement.severity]
    label = f"Max: {measurement.max_width_mm:.3f} mm | {severity_en}"
    cv2.rectangle(out, (8, 8), (min(out.shape[1] - 8, 520), 42), (0, 0, 0), -1)
    cv2.putText(out, label, (15, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.66, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def process_image(model: YOLO, image_path: Path, output_dir: Path, args: argparse.Namespace) -> CrackMeasurement:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"เปิดภาพไม่ได้: {image_path}")
    scale = resolve_scale(args, image.shape)
    prediction = model.predict(source=str(image_path), imgsz=args.imgsz, conf=args.conf, device=args.device,
                               retina_masks=True, verbose=False)[0]
    names = {int(k): str(v) for k, v in prediction.names.items()}
    class_ids = select_class_ids(names, args.crack_class)
    mask, used = build_crack_mask(prediction, image.shape[:2], class_ids, args.mask_threshold)

    if args.erosion_iterations:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (args.erosion_kernel, args.erosion_kernel))
        mask = cv2.erode(mask, kernel, iterations=args.erosion_iterations)

    width_px, skeleton, point = measure_mask(mask)
    width_mm = width_px * scale
    measurement = CrackMeasurement(
        image=str(image_path), max_width_px=round(width_px, 4), max_width_mm=round(width_mm, 4),
        severity=severity_from_mm(width_mm), mm_per_pixel=scale, crack_area_px=int(mask.sum()),
        max_point_x=None if point is None else point[0], max_point_y=None if point is None else point[1],
        detections_used=used,
    )
    stem = image_path.stem
    cv2.imwrite(str(output_dir / f"{stem}_mask.png"), mask * 255)
    cv2.imwrite(str(output_dir / f"{stem}_skeleton.png"), skeleton * 255)
    cv2.imwrite(str(output_dir / f"{stem}_result.jpg"), draw_overlay(image, mask, skeleton, point, measurement))
    with (output_dir / f"{stem}_result.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(measurement), f, ensure_ascii=False, indent=2)
    return measurement


def parse_args() -> argparse.Namespace:
    project_dir = Path(__file__).resolve().parent
    default_model = project_dir.parent / "gdrive_download" / "weights" / "ACE_1.pt"
    p = argparse.ArgumentParser(description="วัดความกว้างและจำแนกความรุนแรงของรอยแตก")
    p.add_argument("--source", required=True, help="ไฟล์ภาพหรือโฟลเดอร์ภาพ")
    p.add_argument("--model", default=str(default_model), help="ไฟล์โมเดล ACE_1.pt")
    p.add_argument("--output", default=str(project_dir / "outputs"))
    p.add_argument("--crack-class", default="crack", help="ชื่อ class รอยแตก หรือ all")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO detection confidence")
    p.add_argument("--mask-threshold", type=float, default=0.75, help="threshold ของ mask (0-1)")
    p.add_argument("--erosion-kernel", type=int, default=3, help="ขนาด kernel เลขคี่")
    p.add_argument("--erosion-iterations", type=int, default=1)
    p.add_argument("--imgsz", type=int, default=1280)
    p.add_argument("--device", default="cpu", help="cpu, mps หรือ CUDA เช่น 0")
    scale = p.add_argument_group("scale calibration (เลือกหนึ่งวิธี)")
    scale.add_argument("--mm-per-pixel", type=float)
    scale.add_argument("--reference-mm", type=float)
    scale.add_argument("--reference-pixels", type=float)
    scale.add_argument("--image-width-mm", type=float)
    args = p.parse_args()
    if not 0 <= args.mask_threshold <= 1:
        p.error("--mask-threshold ต้องอยู่ระหว่าง 0 และ 1")
    if args.erosion_kernel < 1 or args.erosion_kernel % 2 == 0:
        p.error("--erosion-kernel ต้องเป็นเลขคี่บวก")
    positive = [args.mm_per_pixel, args.reference_mm, args.reference_pixels, args.image_width_mm]
    if any(v is not None and v <= 0 for v in positive):
        p.error("ค่าคาลิเบรตต้องมากกว่า 0")
    return args


def main() -> None:
    args = parse_args()
    source = Path(args.source)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    if source.is_dir():
        images = sorted(p for p in source.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    else:
        images = [source]
    if not images:
        raise SystemExit("ไม่พบไฟล์ภาพ")
    model = YOLO(args.model)
    rows = [process_image(model, path, output, args) for path in images]
    with (output / "summary.csv").open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(rows[0]).keys())
        writer.writeheader()
        writer.writerows(asdict(row) for row in rows)
    for row in rows:
        print(f"{Path(row.image).name}: {row.max_width_mm:.3f} mm ({row.severity})")
    print(f"บันทึกผลลัพธ์ที่: {output.resolve()}")


if __name__ == "__main__":
    main()
    











