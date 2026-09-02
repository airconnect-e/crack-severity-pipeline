"""Synthetic ground-truth test for measure_mask(): draw shapes of known width,
measure them with the real pipeline function, and report the error."""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from crack_severity import measure_mask

SHAPE = (300, 300)


@dataclass
class TestCase:
    name: str
    mask: np.ndarray
    expected_px: float
    tolerance_px: float
    


def straight_line(width: int, vertical: bool = False) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    h, w = SHAPE
    half = width // 2
    if vertical:
        c = w // 2
        mask[:, c - half: c - half + width] = 1
    else:
        c = h // 2
        mask[c - half: c - half + width, :] = 1
    return mask


def diagonal_line(width: float, length: float, angle_deg: float) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    cy, cx = SHAPE[0] // 2, SHAPE[1] // 2
    theta = math.radians(angle_deg)
    dx, dy = math.cos(theta), math.sin(theta)
    nx, ny = -dy, dx
    half_l, half_w = length / 2, width / 2
    corners = [
        (cx + sl * half_l * dx + sw * half_w * nx, cy + sl * half_l * dy + sw * half_w * ny)
        for sl, sw in ((-1, -1), (-1, 1), (1, 1), (1, -1))
    ]
    cv2.fillPoly(mask, [np.array(corners, dtype=np.int32)], 1)
    return mask


def ring(outer_r: int, width: int) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    cy, cx = SHAPE[0] // 2, SHAPE[1] // 2
    cv2.circle(mask, (cx, cy), outer_r, 1, thickness=-1)
    inner = np.zeros(SHAPE, dtype=np.uint8)
    cv2.circle(inner, (cx, cy), outer_r - width, 1, thickness=-1)
    mask[inner > 0] = 0
    return mask


def arc(radius: int, width: int, start_angle: float, end_angle: float) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    cy, cx = SHAPE[0] // 2, SHAPE[1] // 2
    cv2.ellipse(mask, (cx, cy), (radius, radius), 0, start_angle, end_angle, 1, thickness=width)
    return mask


def tapered_line(width_start: float, width_end: float) -> np.ndarray:
    mask = np.zeros(SHAPE, dtype=np.uint8)
    h, w = SHAPE
    cy = h // 2
    for x in range(w):
        half = (width_start + (width_end - width_start) * (x / (w - 1))) / 2
        mask[int(round(cy - half)):int(round(cy + half)), x] = 1
    return mask


def build_cases() -> list[TestCase]:
    return [
        *[TestCase(f"เส้นตรงแนวนอน กว้าง {wpx}px", straight_line(wpx), float(wpx), 1.5)
          for wpx in (3, 5, 9, 15, 21, 31)],
        TestCase("เส้นตรงแนวตั้ง กว้าง 15px", straight_line(15, vertical=True), 15.0, 1.5),
        TestCase("เส้นเฉียง 45 องศา กว้าง 15px", diagonal_line(15, 220, 45), 15.0, 2.5),
        TestCase("เส้นเฉียง 30 องศา กว้าง 9px", diagonal_line(9, 220, 30), 9.0, 2.5),
        TestCase("วงแหวนโค้ง รัศมี 80px กว้าง 11px", ring(80, 11), 11.0, 2.5),
        TestCase("ส่วนโค้ง (arc) กว้าง 13px", arc(100, 13, -60, 60), 13.0, 3.0),
        TestCase("เส้นกว้างไม่เท่ากัน 5 ถึง 25px (ต้องจับค่าสูงสุด)", tapered_line(5, 25), 25.0, 2.5),
    ]


def run(save_debug: Path | None) -> bool:
    all_passed = True
    header = f"{'เคสทดสอบ':45s}{'ค่าจริง(px)':>12s}{'วัดได้(px)':>12s}{'error':>8s}{'error%':>8s}  ผล"
    print(header)
    print("-" * len(header))
    for case in build_cases():
        width_px, skeleton, point = measure_mask(case.mask)
        error = width_px - case.expected_px
        error_pct = (error / case.expected_px * 100) if case.expected_px else 0.0
        passed = abs(error) <= case.tolerance_px
        all_passed &= passed
        status = "ผ่าน" if passed else "ไม่ผ่าน"
        print(f"{case.name:45s}{case.expected_px:12.2f}{width_px:12.2f}{error:8.2f}{error_pct:7.1f}%  {status}")
        if save_debug is not None:
            save_debug.mkdir(parents=True, exist_ok=True)
            overlay = cv2.cvtColor(case.mask * 255, cv2.COLOR_GRAY2BGR)
            overlay[skeleton > 0] = (0, 255, 255)
            if point is not None:
                cv2.circle(overlay, point, 5, (255, 0, 255), 2)
            safe_name = "".join(c if c.isalnum() else "_" for c in case.name)
            cv2.imwrite(str(save_debug / f"{safe_name}.png"), overlay)
    print("-" * len(header))
    print("สรุป: ผ่านทั้งหมด" if all_passed else "สรุป: มีเคสไม่ผ่าน — ตรวจสอบ measure_mask()")
    return all_passed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="ทดสอบความแม่นยำของ measure_mask() ด้วย mask ที่รู้ความกว้างจริง")
    p.add_argument("--save-debug", type=Path, default=None, help="โฟลเดอร์บันทึกภาพ mask/skeleton เพื่อตรวจด้วยตา")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    passed = run(args.save_debug)
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
