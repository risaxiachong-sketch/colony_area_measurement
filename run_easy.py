# -*- coding: utf-8 -*-

import os
from pathlib import Path
from types import SimpleNamespace

from measure_colony_area import collect_images, measure_image, save_csv


# ============================================================
# ここだけ変えれば使えます
# ============================================================

# 測りたい写真が入っているフォルダ
PHOTO_FOLDER = Path(
    r"C:\Users\risa4\OneDrive\ドキュメント\Office のカスタム テンプレート\デスクトップ\植物病理\スコポレチン培地試験　2回目　12日目\iCloud写真"


# 結果を保存するフォルダ
OUTPUT_FOLDER = Path(r"C:\Users\risa4\colony_area_measurement\results")

# シャーレの直径。90 mmシャーレならこのままでOK
PLATE_DIAMETER_MM = 90.0


# ============================================================
# 認識結果を調整したいときに変える値
# ============================================================

# 大きくすると、菌糸として拾う範囲が狭くなります。
# 小さくすると、薄い菌糸も拾いやすくなります。
COLOR_DELTA = 24.0

# 大きくすると、黄色っぽい培地のムラを拾いにくくなります。
# 小さくすると、淡いベージュの菌糸を拾いやすくなります。
YELLOW_DELTA = 5.0

# シャーレの縁を拾ってしまうときは 0.78 くらいに下げます。
INNER_PLATE_FRAC = 0.82


# ============================================================
# ここから下は基本的に変えなくてOK
# ============================================================


def make_settings() -> SimpleNamespace:
    """measure_colony_area.py に渡す設定をまとめる。"""
    return SimpleNamespace(
        plate=None,
        plate_diameter_mm=PLATE_DIAMETER_MM,
        no_overlay=False,
        min_radius_frac=0.18,
        max_radius_frac=0.42,
        hough_param2=35.0,
        inner_plate_frac=INNER_PLATE_FRAC,
        background_inner_frac=0.45,
        background_outer_frac=0.75,
        color_delta=COLOR_DELTA,
        yellow_delta=YELLOW_DELTA,
        dark_delta=35.0,
        min_component_frac=0.0004,
        seed_radius_frac=0.22,
        keep_center_frac=0.38,
    )


def main() -> None:
    """写真フォルダを読み、菌叢面積を測り、CSVと確認画像を保存する。"""
    settings = make_settings()
    output_folder = OUTPUT_FOLDER.resolve()

    print("1. 写真を探しています")
    image_paths = collect_images([PHOTO_FOLDER], pattern="*")
    print(f"   {len(image_paths)} 枚見つかりました")

    if not image_paths:
        print("写真が見つかりませんでした。PHOTO_FOLDERを確認してください。")
        return

    print("2. 菌叢面積を測っています")
    results = []
    for image_path in image_paths:
        result = measure_image(image_path, settings, output_folder)
        results.append(result)

        print(
            f"   {image_path.name}: "
            f"菌糸 {result.mycelium_mm2:.1f} mm2, "
            f"中心の黒い部分 {result.dark_core_mm2:.1f} mm2, "
            f"合計 {result.colony_mm2:.1f} mm2"
        )

    print("3. CSVを保存しています")
    csv_path = output_folder / "colony_area_measurements.csv"
    save_csv(csv_path, results)

    print("完了しました")
    print(f"CSV: {csv_path}")
    print(f"確認画像: {output_folder / 'overlays'}")
    open_results_folder(output_folder / "overlays")


def open_results_folder(path: Path) -> None:
    """Windows で結果の確認画像フォルダを開く。"""
    path.mkdir(parents=True, exist_ok=True)
    try:
        os.startfile(str(path))
    except OSError:
        print(f"結果フォルダを開けませんでした: {path}")


if __name__ == "__main__":
    main()
