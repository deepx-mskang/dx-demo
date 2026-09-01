#!/usr/bin/env python3
"""데모용 시리얼 라벨 시트를 생성한다.

data/seed_devices.json 의 시리얼을 읽어 A4 한 장에 라벨 8개를 배치한다.
(기기 목록은 서버가 소유한다. 시드가 그 원본이다.)
인쇄해서 잘라 쓰거나, 휴대폰 화면에 띄워 카메라에 비춰도 된다.

사용법:
    python3 apps/serial-qr/tools/make_labels.py
    -> apps/serial-qr/assets/serial_labels.png
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

APP_DIR = Path(__file__).resolve().parent.parent
SEED_JSON = APP_DIR / "data" / "seed_devices.json"
OUT_PATH = APP_DIR / "assets" / "serial_labels.png"

# A4 300dpi
PAGE_W, PAGE_H = 2480, 3508
COLS, ROWS = 2, 4
MARGIN = 120
GUTTER = 60

MONO_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
SANS = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
SANS_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def load_serials() -> list[str]:
    """시드 JSON 에서 시리얼을 순서대로 뽑는다."""
    if not SEED_JSON.exists():
        sys.exit(f"not found: {SEED_JSON}")
    try:
        records = json.loads(SEED_JSON.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        sys.exit(f"{SEED_JSON} 파싱 실패: {e}")
    serials = [r["serial"] for r in records if isinstance(r, dict) and r.get("serial")]
    if not serials:
        sys.exit(f"{SEED_JSON} 에서 시리얼을 찾지 못했습니다.")
    return serials


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


def draw_label(canvas: Image.Image, box: tuple[int, int, int, int], serial: str) -> None:
    x0, y0, x1, _ = box
    # 라벨은 셀 높이를 다 쓰지 않는다. 내용이 상단 400px 안에 들어가므로
    # 그만큼만 그려서 잘라내기 쉬운 크기로 만든다.
    y1 = y0 + 400
    d = ImageDraw.Draw(canvas)

    # 라벨 테두리 (자르는 선)
    d.rounded_rectangle([x0, y0, x1, y1], radius=24, outline=(150, 150, 150), width=3)

    w = x1 - x0

    f_brand = font(SANS_BOLD, 58)
    f_serial = font(MONO_BOLD, 72)
    f_caption = font(SANS, 34)

    # 상단 브랜드 바
    d.rectangle([x0 + 3, y0 + 3, x1 - 3, y0 + 100], fill=(10, 14, 20))
    d.text((x0 + 40, y0 + 20), "DEEPX", font=f_brand, fill=(0, 212, 224))

    # 시리얼 — OCR 이 읽어야 하는 본문. 여백을 넉넉히 두고 크게 그린다.
    #
    # "S/N:" 접두사는 장식이 아니다. 실시간 자동 인식이 "이건 진짜 시리얼
    # 라벨이다" 를 판정하는 키워드 근거이므로 같은 줄에 붙여 둔다.
    line = f"S/N: {serial}"
    bbox = d.textbbox((0, 0), line, font=f_serial)
    d.text(
        (x0 + (w - (bbox[2] - bbox[0])) / 2, y0 + 190),
        line,
        font=f_serial,
        fill=(0, 0, 0),
    )

    caption = "DX-M1 NPU MODULE"
    bbox = d.textbbox((0, 0), caption, font=f_caption)
    d.text(
        (x0 + (w - (bbox[2] - bbox[0])) / 2, y0 + 320),
        caption,
        font=f_caption,
        fill=(110, 110, 110),
    )


def main() -> None:
    serials = load_serials()
    slots = COLS * ROWS
    if len(serials) > slots:
        print(f"경고: 시리얼 {len(serials)}개 중 앞 {slots}개만 배치합니다.")
        serials = serials[:slots]

    canvas = Image.new("RGB", (PAGE_W, PAGE_H), "white")

    cell_w = (PAGE_W - 2 * MARGIN - (COLS - 1) * GUTTER) // COLS
    cell_h = (PAGE_H - 2 * MARGIN - (ROWS - 1) * GUTTER) // ROWS

    for i, serial in enumerate(serials):
        col, row = i % COLS, i // COLS
        x0 = MARGIN + col * (cell_w + GUTTER)
        y0 = MARGIN + row * (cell_h + GUTTER)
        draw_label(canvas, (x0, y0, x0 + cell_w, y0 + cell_h), serial)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(OUT_PATH, dpi=(300, 300))
    print(f"wrote {OUT_PATH}  ({len(serials)} labels)")


if __name__ == "__main__":
    main()
