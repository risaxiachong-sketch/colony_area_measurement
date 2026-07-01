# -*- coding: utf-8 -*-

from __future__ import annotations

import io
import tempfile
import zipfile
from pathlib import Path
from types import SimpleNamespace

import cv2
import pandas as pd
import streamlit as st

from measure_colony_area import IMAGE_EXTENSIONS, Measurement, measure_image, read_image, save_csv


APP_TITLE = "シャーレ菌叢 解析"
UPLOAD_EXTENSIONS = sorted(ext.lstrip(".") for ext in IMAGE_EXTENSIONS) + ["zip"]

ANALYSIS_PRESETS = {
    "標準（黄色・黒色中心）": {
        "description": "これまでのスコポレチン培地のように、黄色〜ベージュの菌糸と黒い中心部を拾います。",
        "analysis_mode": "mixed",
        "color_delta": 24.0,
        "yellow_delta": 5.0,
        "dark_delta": 35.0,
        "white_delta": 10.0,
        "texture_delta": 6.0,
        "brown_delta": 7.0,
        "inner_plate_frac": 0.82,
        "background_inner_frac": 0.45,
        "background_outer_frac": 0.75,
        "min_component_frac": 0.0004,
        "seed_radius_frac": 0.22,
        "keep_center_frac": 0.38,
        "footprint_close_frac": 0.08,
    },
    "白色菌糸を拾う": {
        "description": "白い枝状の菌糸を拾いやすくします。色よりも明るさと細かい線状のコントラストを重視します。",
        "analysis_mode": "light_mycelium",
        "color_delta": 16.0,
        "yellow_delta": 3.0,
        "dark_delta": 35.0,
        "white_delta": 6.0,
        "texture_delta": 4.0,
        "brown_delta": 7.0,
        "inner_plate_frac": 0.90,
        "background_inner_frac": 0.76,
        "background_outer_frac": 0.92,
        "min_component_frac": 0.00005,
        "seed_radius_frac": 0.24,
        "keep_center_frac": 0.60,
        "footprint_close_frac": 0.06,
    },
    "茶色色素も含める": {
        "description": "菌糸だけでなく、茶色〜黄色に変色した培地部分も菌叢として含めます。",
        "analysis_mode": "pigment_included",
        "color_delta": 18.0,
        "yellow_delta": 2.0,
        "dark_delta": 35.0,
        "white_delta": 7.0,
        "texture_delta": 4.0,
        "brown_delta": 5.0,
        "inner_plate_frac": 0.90,
        "background_inner_frac": 0.76,
        "background_outer_frac": 0.92,
        "min_component_frac": 0.00008,
        "seed_radius_frac": 0.24,
        "keep_center_frac": 0.65,
        "footprint_close_frac": 0.08,
    },
    "外縁面積を測る": {
        "description": "枝のすき間をある程度埋めて、菌叢が広がった外側の範囲を測ります。",
        "analysis_mode": "footprint",
        "color_delta": 18.0,
        "yellow_delta": 2.0,
        "dark_delta": 35.0,
        "white_delta": 7.0,
        "texture_delta": 4.0,
        "brown_delta": 5.0,
        "inner_plate_frac": 0.90,
        "background_inner_frac": 0.76,
        "background_outer_frac": 0.92,
        "min_component_frac": 0.00008,
        "seed_radius_frac": 0.24,
        "keep_center_frac": 0.70,
        "footprint_close_frac": 0.10,
    },
}


def make_args(
    plate_diameter_mm: float,
    analysis_mode: str,
    color_delta: float,
    yellow_delta: float,
    dark_delta: float,
    white_delta: float,
    texture_delta: float,
    brown_delta: float,
    inner_plate_frac: float,
    background_inner_frac: float,
    background_outer_frac: float,
    min_component_frac: float,
    seed_radius_frac: float,
    keep_center_frac: float,
    footprint_close_frac: float,
) -> SimpleNamespace:
    """measure_colony_area.py に渡す解析設定を作ります。"""

    background_outer_frac = max(background_outer_frac, background_inner_frac + 0.02)

    return SimpleNamespace(
        plate=None,
        plate_diameter_mm=plate_diameter_mm,
        no_overlay=False,
        min_radius_frac=0.18,
        max_radius_frac=0.42,
        hough_param2=35.0,
        inner_plate_frac=inner_plate_frac,
        background_inner_frac=background_inner_frac,
        background_outer_frac=background_outer_frac,
        color_delta=color_delta,
        yellow_delta=yellow_delta,
        dark_delta=dark_delta,
        min_component_frac=min_component_frac,
        seed_radius_frac=seed_radius_frac,
        keep_center_frac=keep_center_frac,
        analysis_mode=analysis_mode,
        white_delta=white_delta,
        texture_delta=texture_delta,
        brown_delta=brown_delta,
        footprint_close_frac=footprint_close_frac,
    )


def measurement_to_row(measurement: Measurement) -> dict[str, object]:
    """解析結果を表に出しやすい形へ変換します。"""

    return {
        "image": measurement.image.name,
        "plate_center_x": measurement.plate.cx,
        "plate_center_y": measurement.plate.cy,
        "plate_radius_px": measurement.plate.radius,
        "plate_diameter_px": measurement.plate.radius * 2,
        "pixel_mm": measurement.pixel_mm,
        "plate_area_mm2": measurement.plate_area_mm2,
        "colony_px": measurement.colony_px,
        "colony_mm2": measurement.colony_mm2,
        "colony_percent_plate": measurement.colony_percent_plate,
        "mycelium_px": measurement.mycelium_px,
        "mycelium_mm2": measurement.mycelium_mm2,
        "dark_core_px": measurement.dark_core_px,
        "dark_core_mm2": measurement.dark_core_mm2,
        "overlay": "" if measurement.overlay is None else measurement.overlay.name,
    }


def clean_folder_text(folder_text: str) -> Path:
    """エクスプローラーからコピーしたパスの余分な引用符を取り除きます。"""

    return Path(folder_text.strip().strip('"').strip("'"))


def collect_folder_images(folder: Path, recursive: bool) -> list[Path]:
    """PC内フォルダから解析対象の画像だけを集めます。"""

    if not folder.exists():
        raise FileNotFoundError(f"画像フォルダが見つかりません: {folder}")
    if not folder.is_dir():
        raise NotADirectoryError(f"フォルダではありません: {folder}")

    iterator = folder.rglob("*") if recursive else folder.iterdir()
    images = sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not images:
        raise FileNotFoundError(f"解析できる画像が見つかりません: {folder}")
    return images


def make_safe_filename(name: str, fallback: str) -> str:
    """アップロード名からフォルダ記号を取り除き、保存しやすい名前にします。"""

    filename = Path(name.replace("\\", "/")).name
    return filename or fallback


def save_uploaded_image(uploaded_file, upload_dir: Path, index: int) -> Path | None:
    """アップロードされた画像を一時フォルダに保存します。"""

    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix not in IMAGE_EXTENSIONS:
        return None

    filename = make_safe_filename(uploaded_file.name, f"image_{index}{suffix}")
    path = upload_dir / f"{index:03d}_{filename}"
    path.write_bytes(uploaded_file.getbuffer())
    return path


def save_zip_images(uploaded_zip, upload_dir: Path, start_index: int) -> tuple[list[Path], int]:
    """ZIPファイル内の画像だけを取り出して一時フォルダに保存します。"""

    image_paths: list[Path] = []
    next_index = start_index

    try:
        archive = zipfile.ZipFile(io.BytesIO(uploaded_zip.getbuffer()))
    except zipfile.BadZipFile as exc:
        raise ValueError(f"ZIPファイルを開けません: {uploaded_zip.name}") from exc

    with archive:
        for member in archive.infolist():
            if member.is_dir():
                continue

            suffix = Path(member.filename).suffix.lower()
            if suffix not in IMAGE_EXTENSIONS:
                continue

            filename = make_safe_filename(member.filename, f"image_{next_index}{suffix}")
            path = upload_dir / f"{next_index:03d}_{filename}"
            path.write_bytes(archive.read(member))
            image_paths.append(path)
            next_index += 1

    return image_paths, next_index


def save_uploads(uploaded_files: list, upload_dir: Path) -> list[Path]:
    """画像ファイルとZIPファイルを受け取り、解析用の画像パス一覧を作ります。"""

    upload_dir.mkdir(parents=True, exist_ok=True)
    image_paths: list[Path] = []
    next_index = 1

    for uploaded_file in uploaded_files:
        suffix = Path(uploaded_file.name).suffix.lower()

        if suffix == ".zip":
            zip_paths, next_index = save_zip_images(uploaded_file, upload_dir, next_index)
            image_paths.extend(zip_paths)
            continue

        image_path = save_uploaded_image(uploaded_file, upload_dir, next_index)
        if image_path is not None:
            image_paths.append(image_path)
            next_index += 1

    return image_paths


def build_zip(output_dir: Path) -> bytes:
    """CSVとオーバーレイ画像をまとめたZIPを作ります。"""

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in output_dir.rglob("*"):
            if path.is_file():
                archive.write(path, path.relative_to(output_dir))
    buffer.seek(0)
    return buffer.getvalue()


def show_overlay(path: Path) -> None:
    """保存したオーバーレイ画像を画面に表示します。"""

    try:
        image = read_image(path)
    except ValueError:
        st.warning(f"オーバーレイ画像を表示できません: {path.name}")
        return

    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(rgb, caption=path.name, width="stretch")


def run_analysis_paths(
    image_paths: list[Path],
    args: SimpleNamespace,
    output_dir: Path,
) -> tuple[pd.DataFrame, Path, list[dict[str, str]]]:
    """画像を1枚ずつ解析し、CSVとオーバーレイ画像を保存します。"""

    measurements: list[Measurement] = []
    errors: list[dict[str, str]] = []

    progress = st.progress(0)
    status = st.empty()

    for index, image_path in enumerate(image_paths, start=1):
        status.write(f"解析中: {image_path.name}")
        try:
            measurement = measure_image(image_path, args, output_dir)
            measurements.append(measurement)
        except Exception as exc:
            errors.append({"image": image_path.name, "error": str(exc)})
        progress.progress(index / len(image_paths))

    status.write("解析が完了しました")

    if not measurements:
        raise ValueError("すべての画像で解析に失敗しました。設定値や画像を確認してください。")

    csv_path = output_dir / "colony_area_measurements.csv"
    save_csv(csv_path, measurements)

    if errors:
        pd.DataFrame(errors).to_csv(output_dir / "errors.csv", index=False, encoding="utf-8-sig")

    rows = [measurement_to_row(item) for item in measurements]
    return pd.DataFrame(rows), output_dir, errors


def run_uploaded_analysis(
    uploaded_files: list,
    args: SimpleNamespace,
) -> tuple[pd.DataFrame, Path, list[dict[str, str]]]:
    """アップロードされた画像、またはZIP内の画像を一括解析します。"""

    run_dir = Path(tempfile.mkdtemp(prefix="colony_area_app_"))
    upload_dir = run_dir / "uploads"
    output_dir = run_dir / "results"

    image_paths = save_uploads(uploaded_files, upload_dir)
    if not image_paths:
        extensions = ", ".join(sorted(IMAGE_EXTENSIONS))
        raise ValueError(f"解析できる画像がありません。対応形式: {extensions}, .zip")

    return run_analysis_paths(image_paths, args, output_dir)


def run_folder_analysis(
    folder_text: str,
    recursive: bool,
    args: SimpleNamespace,
) -> tuple[pd.DataFrame, Path, list[dict[str, str]]]:
    """このアプリを動かしているPC内のフォルダを一括解析します。"""

    folder = clean_folder_text(folder_text)
    image_paths = collect_folder_images(folder, recursive)

    run_dir = Path(tempfile.mkdtemp(prefix="colony_area_app_"))
    output_dir = run_dir / "results"
    return run_analysis_paths(image_paths, args, output_dir)


def show_result_area(result_df: pd.DataFrame, output_dir: Path, errors: list[dict[str, str]]) -> None:
    """解析後の表、ダウンロード、オーバーレイを表示します。"""

    st.subheader("結果")
    st.dataframe(result_df, width="stretch")

    csv_path = output_dir / "colony_area_measurements.csv"
    st.download_button(
        "CSVをダウンロード",
        data=csv_path.read_bytes(),
        file_name="colony_area_measurements.csv",
        mime="text/csv",
    )

    st.download_button(
        "CSVとオーバーレイ画像をZIPでダウンロード",
        data=build_zip(output_dir),
        file_name="colony_area_results.zip",
        mime="application/zip",
    )

    if errors:
        with st.expander("解析できなかった画像"):
            st.dataframe(pd.DataFrame(errors), width="stretch")

    overlay_paths = sorted((output_dir / "overlays").glob("*_overlay.png"))
    if overlay_paths:
        st.subheader("オーバーレイ画像")
        max_count = st.slider("表示する枚数", 1, len(overlay_paths), min(3, len(overlay_paths)))
        columns = st.columns(min(3, max_count))
        for index, overlay_path in enumerate(overlay_paths[:max_count]):
            with columns[index % len(columns)]:
                show_overlay(overlay_path)


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)

    st.caption("画像をまとめて解析し、菌叢面積をCSVとオーバーレイ画像として保存します。")

    with st.sidebar:
        st.header("解析設定")
        preset_name = st.selectbox("菌の種類・解析目的", list(ANALYSIS_PRESETS.keys()))
        preset = ANALYSIS_PRESETS[preset_name]
        st.caption(str(preset["description"]))

        plate_diameter_mm = st.number_input("シャーレ直径 mm", min_value=1.0, value=90.0, step=1.0)

        with st.expander("詳細設定", expanded=False):
            st.caption("オーバーレイ画像を見ながら、拾いすぎ・拾わなさすぎを調整します。")
            color_delta = st.slider("背景との差の強さ", 5.0, 60.0, float(preset["color_delta"]), 1.0)
            yellow_delta = st.slider("黄色〜ベージュ部分の拾いやすさ", 0.0, 20.0, float(preset["yellow_delta"]), 1.0)
            dark_delta = st.slider("黒い中心部の拾いやすさ", 5.0, 80.0, float(preset["dark_delta"]), 1.0)
            white_delta = st.slider("白い菌糸の拾いやすさ", 0.0, 30.0, float(preset["white_delta"]), 1.0)
            texture_delta = st.slider("細い枝状菌糸の拾いやすさ", 0.0, 25.0, float(preset["texture_delta"]), 1.0)
            brown_delta = st.slider("茶色い色素の拾いやすさ", 0.0, 25.0, float(preset["brown_delta"]), 1.0)
            inner_plate_frac = st.slider("シャーレ内側だけを解析する割合", 0.60, 0.95, float(preset["inner_plate_frac"]), 0.01)
            footprint_close_frac = st.slider(
                "外縁面積で枝のすき間を埋める強さ",
                0.02,
                0.18,
                float(preset["footprint_close_frac"]),
                0.01,
            )

    input_mode = st.radio(
        "入力方法",
        ["画像またはZIPをアップロード", "PC内フォルダのパスを指定"],
        horizontal=True,
    )

    uploaded_files = None
    folder_text = ""
    recursive = False

    if input_mode == "画像またはZIPをアップロード":
        st.info(
            "画像を複数選ぶか、画像フォルダをZIPにしてアップロードできます。"
            "みんなで使うWebアプリにする場合は、この方法が向いています。"
        )
        uploaded_files = st.file_uploader(
            "画像ファイル、または画像フォルダを圧縮したZIP",
            type=UPLOAD_EXTENSIONS,
            accept_multiple_files=True,
        )
        ready_to_run = bool(uploaded_files)
    else:
        st.info(
            "このアプリを動かしているPC内のフォルダだけ指定できます。"
            "Web公開した場合、利用者のPCフォルダは直接読めないため、アップロード方式を使ってください。"
        )
        folder_text = st.text_input(
            "画像フォルダのパス",
            placeholder=r"C:\Users\risa4\OneDrive\...\iCloud写真",
        )
        recursive = st.checkbox("サブフォルダ内の画像も含める", value=False)
        ready_to_run = bool(folder_text.strip())

    args = make_args(
        plate_diameter_mm=plate_diameter_mm,
        analysis_mode=str(preset["analysis_mode"]),
        color_delta=color_delta,
        yellow_delta=yellow_delta,
        dark_delta=dark_delta,
        white_delta=white_delta,
        texture_delta=texture_delta,
        brown_delta=brown_delta,
        inner_plate_frac=inner_plate_frac,
        background_inner_frac=float(preset["background_inner_frac"]),
        background_outer_frac=float(preset["background_outer_frac"]),
        min_component_frac=float(preset["min_component_frac"]),
        seed_radius_frac=float(preset["seed_radius_frac"]),
        keep_center_frac=float(preset["keep_center_frac"]),
        footprint_close_frac=footprint_close_frac,
    )

    if st.button("解析開始", type="primary", disabled=not ready_to_run):
        try:
            if input_mode == "画像またはZIPをアップロード":
                result_df, output_dir, errors = run_uploaded_analysis(uploaded_files or [], args)
            else:
                result_df, output_dir, errors = run_folder_analysis(folder_text, recursive, args)
        except Exception as exc:
            st.error(str(exc))
            return

        st.session_state["result_df"] = result_df
        st.session_state["output_dir"] = str(output_dir)
        st.session_state["errors"] = errors

    if "result_df" not in st.session_state:
        st.info("画像ファイル、ZIP、または画像フォルダのパスを指定して、解析開始を押してください。")
        return

    result_df = st.session_state["result_df"]
    output_dir = Path(st.session_state["output_dir"])
    errors = st.session_state.get("errors", [])
    show_result_area(result_df, output_dir, errors)


if __name__ == "__main__":
    main()
