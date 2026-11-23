# -*- coding: utf-8 -*-
"""VS Codeの実行ボタンで起動する、確認付き菌糸面積測定ツール。"""
from __future__ import annotations

import csv
import json
import math
import re
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk


# ===== 基本設定（通常は画面から変更できます） =====
PLATE_DIAMETER_MM = 90.0  # ND90-15の公称直径。高さ15mmは面積換算に使いません。
EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}
SCRIPT_DIR = Path(__file__).resolve().parent
RESULT_FOLDER = '解析結果'
ANALYSIS_MODES = {'白色・明るい菌叢': 'white', '黒色・褐色も含める': 'mixed'}


def collect_photos(folder: Path):
    """作成済みの解析画像を再入力しないように除外します。"""
    excluded = {RESULT_FOLDER, 'analysis_results', 'overlays', 'masks'}
    return sorted(p for p in folder.rglob('*')
                  if p.is_file() and p.suffix.lower() in EXTENSIONS
                  and not excluded.intersection(p.relative_to(folder).parts[:-1]))


def folder_day(folder: Path):
    # 「6日目」等の明記がある場合だけ撮影時点を仮入力します。
    for part in reversed(folder.parts):
        match = re.search(r'(\d+)\s*日目', part)
        if match:
            return match.group(1)
    return ''


def read_image(path: Path) -> np.ndarray:
    # EXIFの向きと日本語パスに対応します。
    with Image.open(path) as source:
        rgb = np.asarray(ImageOps.exif_transpose(source).convert('RGB'))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, data = cv2.imencode('.png', image)
    if not ok:
        raise ValueError('画像を保存できません')
    data.tofile(str(path))


def circle_from_points(points) -> tuple[float, float, float]:
    points = np.asarray(points, dtype=float)
    a = 2 * (points[1:] - points[0])
    b = (points[1:] ** 2).sum(axis=1) - (points[0] ** 2).sum()
    if abs(np.linalg.det(a)) < 1e-5:
        raise ValueError('3点が一直線です。円周の離れた3点を選んでください。')
    center = np.linalg.solve(a, b)
    return float(center[0]), float(center[1]), float(np.linalg.norm(center - points[0]))


def circle_roi(shape, circle, inner_fraction=1.0):
    yy, xx = np.ogrid[:shape[0], :shape[1]]
    x, y, radius = circle
    return (xx - x) ** 2 + (yy - y) ** 2 <= (radius * inner_fraction) ** 2


def detect_circle(image) -> tuple[float, float, float]:
    """黒背景上のシャーレ外周を候補にします。確定前に目視確認が必要です。"""
    h, w = image.shape[:2]
    scale = min(1.0, 800 / max(h, w))
    gray = cv2.cvtColor(cv2.resize(image, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        (x, y), r = cv2.minEnclosingCircle(contour)
        if not 0.15 * min(gray.shape) < r < 0.53 * min(gray.shape):
            continue
        filled_fraction = cv2.contourArea(contour) / (math.pi * r * r)
        if filled_fraction > 0.72:
            candidates.append((r * r * filled_fraction, x, y, r))
    if not candidates:
        raise ValueError('シャーレを検出できません。外周3点を指定してください。')
    _, x, y, r = max(candidates)
    return x / scale, y / scale, r / scale


def fit_background(light, circle, annulus):
    """周辺培地の明るさから、緩やかな照明ムラを推定します。"""
    yy, xx = np.indices(light.shape, dtype=np.float32)
    x = (xx - circle[0]) / circle[2]
    y = (yy - circle[1]) / circle[2]
    # 二次曲面は中心へ外挿した際に偽の明るい領域を作りやすいため、平面で補正します。
    terms = np.stack([np.ones_like(x), x, y], axis=-1)
    samples = terms[annulus][::max(1, int(annulus.sum()) // 4000)]
    values = light[annulus][::max(1, int(annulus.sum()) // 4000)]
    keep = np.ones(len(values), dtype=bool)
    for _ in range(4):
        coefficients = np.linalg.lstsq(samples[keep], values[keep], rcond=None)[0]
        residual = values - samples @ coefficients
        median = np.median(residual[keep])
        deviation = max(2.0, 1.4826 * np.median(np.abs(residual[keep] - median)))
        new_keep = np.abs(residual - median) < 2.5 * deviation
        if new_keep.sum() < 30:
            break
        keep = new_keep
    return np.clip(terms @ coefficients, 0, 255)


def background_surfaces(lab, circle, roi, annulus, mode, background_points):
    """色のある菌叢と反射を避け、培地のLab背景を推定します。"""
    samples = annulus.copy()
    if background_points:
        samples[:] = False
        for point in background_points:
            if not (0 <= point[0] < roi.shape[1] and 0 <= point[1] < roi.shape[0]) or not roi[min(roi.shape[0]-1, round(point[1])), min(roi.shape[1]-1, round(point[0]))]:
                raise ValueError('背景はオレンジ円の内側の、菌叢のない培地を指定してください。')
            samples |= circle_roi(roi.shape, (*point, max(3, circle[2] * 0.025))) & roi
    elif mode == 'mixed':
        chroma = np.linalg.norm(lab[:, :, 1:] - 128, axis=2)
        light = lab[:, :, 0]
        lo, hi = np.percentile(light[annulus], [10, 90])
        clear_agar = annulus & (chroma <= np.percentile(chroma[annulus], 55)) & (light >= lo) & (light <= hi)
        if clear_agar.sum() >= 30:
            samples = clear_agar
    if samples.sum() < 30:
        raise ValueError('背景の画素が足りません。菌叢のない培地を指定し直してください。')
    if background_points and len(background_points) < 3:
        surfaces = np.broadcast_to(np.median(lab[samples], axis=0), lab.shape).copy()
    else:
        if background_points:
            normalized = (np.asarray(background_points, dtype=float) - np.asarray(circle[:2])) / circle[2]
            if abs(np.linalg.det(np.column_stack([np.ones(3), normalized]))) < 0.02:
                raise ValueError('背景の3点は一直線にせず、離れた培地部分を選んでください。')
        surfaces = np.stack([fit_background(lab[:, :, channel], circle, samples) for channel in range(3)], axis=-1)
    return surfaces, float(np.median(lab[samples, 0]))


def segment_mycelium(image, circle, threshold=16.0, inner_fraction=0.90, colony_center=None,
                     analysis_mode='white', background_points=None):
    """背景との差を使い、菌叢付近につながる成分を抽出します。"""
    if analysis_mode not in ANALYSIS_MODES.values():
        raise ValueError('解析モードが不正です')
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB).astype(np.float32)
    light = lab[:, :, 0]
    roi = circle_roi(light.shape, circle, inner_fraction)
    annulus = roi & ~circle_roi(light.shape, circle, max(0.05, inner_fraction - 0.12))
    if annulus.sum() < 30:
        raise ValueError('シャーレ円または解析範囲が小さすぎます')
    surface, background = background_surfaces(lab, circle, roi, annulus, analysis_mode, background_points)
    delta = lab - surface
    if analysis_mode == 'white':
        contrast = delta[:, :, 0]
    else:
        # OpenCV Labの0〜255表現での色差。暗い変化も対象にし、
        # 明暗だけの影響は半分にして色の違いを重視します（標準のDelta Eではありません）。
        contrast = np.sqrt((delta[:, :, 0] * 0.5)**2 + delta[:, :, 1]**2 + delta[:, :, 2]**2)
    center = colony_center if colony_center is not None else circle[:2]
    seed_region = circle_roi(light.shape, (*center, circle[2] * 0.25)) & roi
    strong = roi & (contrast >= threshold)
    # 境界は低めのしきい値で拾い、中心付近の明確な菌叢とつながる成分を採用します。
    weak_factor = 0.5 if analysis_mode == 'white' else 0.75
    weak = roi & (contrast >= max(2.0, threshold * weak_factor))
    count, labels, stats, _ = cv2.connectedComponentsWithStats(weak.astype(np.uint8), 8)
    keep = np.zeros(count, dtype=bool)
    seeds = np.unique(labels[strong & seed_region])
    seeds = seeds[seeds > 0]
    keep[seeds] = stats[seeds, cv2.CC_STAT_AREA] >= 3
    return keep[labels] & roi, background


def area_mm2(mask, circle, diameter_mm):
    if circle[2] <= 0 or diameter_mm <= 0:
        raise ValueError('シャーレ直径は正の値にしてください')
    return int(np.count_nonzero(mask)) * (diameter_mm / (2 * circle[2])) ** 2


def include_disk_mask(mycelium, disk_circle, plate_circle, inner_fraction):
    """菌糸と指定ディスクの和集合。重なる画素は一度だけ数えます。"""
    disk = np.zeros(mycelium.shape, dtype=bool)
    if disk_circle is not None:
        disk = circle_roi(mycelium.shape, disk_circle)
        if not np.any(disk) or np.any(disk & ~circle_roi(mycelium.shape, plate_circle, inner_fraction)):
            raise ValueError('ディスク全体が解析範囲に入るよう、円または解析半径を調整してください。')
        x, y, r = disk_circle
        h, w = mycelium.shape
        if x-r < 0 or y-r < 0 or x+r > w-1 or y+r > h-1:
            raise ValueError('ディスクが画像からはみ出しています。輪郭を指定し直してください。')
    return mycelium | disk, disk


def locate_disk_center(image, plate_circle, mycelium, expected_radius, colony_center=None):
    """ディスク径を固定し、中央付近の暗い成分から位置の候補を出します。"""
    light = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    origin = colony_center if colony_center is not None else plate_circle[:2]
    search = circle_roi(light.shape, (*origin, plate_circle[2] * 0.35))
    baseline = float(np.median(light[search]))
    dark = search & (light.astype(float) < baseline - 12)
    count, labels, stats, centroids = cv2.connectedComponentsWithStats(dark.astype(np.uint8), 8)
    expected_area = math.pi * expected_radius**2
    candidates = []
    for label in range(1, count):
        area = stats[label, cv2.CC_STAT_AREA]
        if max(3, expected_area * 0.08) <= area <= expected_area * 4:
            x, y = centroids[label]
            distance = math.hypot(x - origin[0], y - origin[1]) / plate_circle[2]
            size_error = abs(math.log(area / expected_area))
            candidates.append((distance + size_error * 0.08, x, y))
    if candidates:
        _, x, y = min(candidates)
        return (float(x), float(y)), 'auto_dark_center'
    ys, xs = np.where(mycelium & search)
    if len(xs):
        return (float(np.mean(xs)), float(np.mean(ys))), 'auto_colony_center'
    return tuple(origin), 'plate_center_fallback'


def overlay_image(image, mask, circle, inner_fraction):
    overlay = image.copy()
    overlay[mask] = (image[mask] * 0.55 + np.array([40, 220, 40]) * 0.45).astype(np.uint8)
    x, y, r = (int(round(v)) for v in circle)
    cv2.circle(overlay, (x, y), r, (255, 255, 0), 1)
    cv2.circle(overlay, (x, y), round(r * inner_fraction), (0, 180, 255), 1)
    return overlay


def save_rows(path: Path, rows):
    # 保存途中で終了しても、前のCSVを壊しにくいように置換します。
    temporary = path.with_suffix('.tmp')
    fields = ['image_name', 'photo_folder', 'relative_path', 'treatment', 'replicate', 'specimen_id', 'timepoint',
              'time_unit', 'review_status', 'area_mm2', 'area_px', 'plate_diameter_mm',
              'plate_center_x', 'plate_center_y', 'plate_radius_px', 'mm_per_pixel',
              'width_px', 'height_px', 'brightness_threshold', 'inner_fraction',
              'background_L', 'manual_edited', 'circle_method', 'mask_file', 'overlay_file',
              'segmentation_method', 'colony_center_x', 'colony_center_y']
    fields += ['disk_included', 'disk_center_x', 'disk_center_y', 'disk_radius_px',
               'disk_mm2', 'mycelium_excluding_disk_mm2', 'mycelium_mask_file',
               'disk_diameter_mm', 'disk_position_method']
    fields += ['analysis_mode', 'background_method', 'background_points_json']
    with temporary.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


class ColonyApp:
    def __init__(self, root):
        self.root = root
        root.title('菌糸の投影面積測定')
        root.geometry('1180x820')
        self.paths, self.records = [], {}
        self.index = 0
        self.image = None
        self.mask = None
        self.points = []
        self.tool = 'view'
        self.edited = False
        self.circle_method = 'auto'
        self.scale, self.origin = 1.0, (0, 0)
        self.last_point = None
        self.colony_center = None
        self.background_points = []
        self.mode = tk.StringVar(value='黒色・褐色も含める')
        self.disk_circle = None
        self.disk_template_mm = None
        self.disk_method = 'unassigned'
        self.reuse_disk = tk.BooleanVar(value=True)
        self.disk_status = tk.StringVar(value='ディスク径: 最初の1枚で指定')
        self.include_disk = tk.BooleanVar(value=True)
        self.diameter = tk.DoubleVar(value=PLATE_DIAMETER_MM)
        self.threshold = tk.DoubleVar(value=18)
        self.inner = tk.DoubleVar(value=0.85)
        self.brush = tk.IntVar(value=8)
        self.checked = tk.BooleanVar(value=False)
        self.status = tk.StringVar(value='画像フォルダを選んでください')
        top = ttk.Frame(root, padding=8)
        top.pack(fill='x')
        ttk.Button(top, text='画像フォルダを開く', command=self.open_folder).pack(side='left')
        ttk.Label(top, textvariable=self.status).pack(side='left', padx=12)
        main = ttk.Frame(root)
        main.pack(fill='both', expand=True)
        panel = ttk.Frame(main, width=285)
        panel.pack(side='left', fill='y')
        scroll = tk.Canvas(panel, width=270, highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient='vertical', command=scroll.yview)
        scroll.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side='right', fill='y')
        scroll.pack(side='left', fill='y', expand=True)
        controls = ttk.Frame(scroll, padding=10)
        scroll.create_window((0, 0), window=controls, anchor='nw')
        controls.bind('<Configure>', lambda event: scroll.configure(scrollregion=scroll.bbox('all')))
        self.canvas = tk.Canvas(main, bg='#262626', highlightthickness=0)
        self.canvas.pack(side='left', fill='both', expand=True)
        self.canvas.bind('<Configure>', lambda event: self.draw())
        self.canvas.bind('<Button-1>', self.click)
        self.canvas.bind('<B1-Motion>', self.drag)
        self.canvas.bind('<ButtonRelease-1>', lambda event: setattr(self, 'last_point', None))
        ttk.Label(controls, text='シャーレ直径 (mm)').pack(anchor='w')
        ttk.Entry(controls, textvariable=self.diameter, width=16).pack(anchor='w')
        ttk.Button(controls, text='外周の3点を指定', command=self.start_circle).pack(fill='x', pady=6)
        ttk.Button(controls, text='菌叢の中心を指定', command=lambda: self.set_tool('seed')).pack(fill='x', pady=6)
        ttk.Label(controls, text='解析モード').pack(anchor='w')
        ttk.Combobox(controls, textvariable=self.mode, values=list(ANALYSIS_MODES), state='readonly', width=26).pack(fill='x')
        ttk.Button(controls, text='背景の培地を指定（1〜3点）',
                   command=lambda: self.set_tool('background')).pack(fill='x', pady=4)
        ttk.Button(controls, text='背景を自動指定に戻す', command=self.clear_background).pack(fill='x')
        ttk.Checkbutton(controls, text='中央の菌叢ディスクも面積に含める',
                        variable=self.include_disk, command=self.draw).pack(anchor='w')
        ttk.Button(controls, text='ディスクの輪郭を3点で指定',
                   command=lambda: self.set_tool('disk')).pack(fill='x', pady=6)
        ttk.Checkbutton(controls, text='次の写真も同じディスク直径を使う',
                        variable=self.reuse_disk).pack(anchor='w')
        ttk.Label(controls, textvariable=self.disk_status, wraplength=245).pack(anchor='w')
        ttk.Button(controls, text='ディスクの中心だけ修正（1点）',
                   command=lambda: self.set_tool('disk_center')).pack(fill='x', pady=4)
        ttk.Label(controls, text='抽出しきい値（大きいほど厳しい）').pack(anchor='w', pady=(12, 0))
        tk.Scale(controls, variable=self.threshold, from_=1, to=80, resolution=1,
                 orient='horizontal', length=235).pack()
        ttk.Label(controls, text='解析半径の割合').pack(anchor='w')
        tk.Scale(controls, variable=self.inner, from_=0.65, to=0.98, resolution=0.01,
                 orient='horizontal', length=235).pack()
        ttk.Button(controls, text='この設定で抽出し直す', command=self.extract).pack(fill='x', pady=4)
        ttk.Label(controls, text='修正ブラシの半径 (元画像px)').pack(anchor='w', pady=(10, 0))
        tk.Scale(controls, variable=self.brush, from_=1, to=30, orient='horizontal', length=235).pack()
        for label, tool in [('菌糸を塗り足す', 'add'), ('拾いすぎを消す', 'erase'), ('修正操作を終了', 'view')]:
            ttk.Button(controls, text=label, command=lambda t=tool: self.set_tool(t)).pack(fill='x', pady=2)
        self.metadata = {}
        for key, label in [('treatment', '処理区'), ('replicate', '反復番号'),
                           ('specimen_id', 'シャーレID'), ('timepoint', '撮影時点'), ('time_unit', '時点の単位（例: day）')]:
            ttk.Label(controls, text=label).pack(anchor='w', pady=(5, 0))
            var = tk.StringVar()
            self.metadata[key] = var
            ttk.Entry(controls, textvariable=var).pack(fill='x')
        bottom = ttk.Frame(root, padding=8)
        bottom.pack(fill='x')
        ttk.Button(bottom, text='前へ', command=self.previous).pack(side='left')
        ttk.Button(bottom, text='保留して次へ', command=self.skip).pack(side='left', padx=8)
        ttk.Button(bottom, text='OK・保存して次へ', command=self.approve).pack(side='left')
        self.area_text = tk.StringVar(value='')
        ttk.Label(bottom, textvariable=self.area_text).pack(side='right')
        self.save_location = tk.StringVar(value='')
        ttk.Label(root, textvariable=self.save_location, wraplength=1120, padding=(8, 2)).pack(fill='x')

    def open_folder(self):
        folder = filedialog.askdirectory(title='解析する画像フォルダ（サブフォルダも含む）')
        if not folder:
            return
        if self.paths and not messagebox.askyesno('フォルダ変更', '別の解析を開始しますか？ 保存済みCSVは残ります。'):
            return
        self.input_root = Path(folder)
        self.paths = collect_photos(self.input_root)
        if not self.paths:
            messagebox.showerror('画像なし', '対応画像が見つかりません')
            return
        self.output = self.input_root / RESULT_FOLDER / datetime.now().strftime('reviewed_%Y%m%d_%H%M%S_%f')
        try:
            self.output.mkdir(parents=True)
            (self.output / 'session.json').write_text(json.dumps({
                'input_folder': str(self.input_root), 'target': 'mycelium_and_optional_inoculation_disk_projected_area',
                'plate_model': 'AS ONE ND90-15', 'note': 'Only approved rows contain areas.',
                'asset_paths_relative_to': str(self.input_root)
            }, indent=2), encoding='utf-8')
        except OSError as exc:
            self.paths = []
            messagebox.showerror('保存先エラー', f'選んだフォルダに結果を保存できません。\n{exc}')
            return
        self.records = {}
        self.index = 0
        self.load()

    def load(self):
        path = self.paths[self.index]
        self.tool, self.points, self.edited = 'view', [], False
        self.checked.set(False)
        self.colony_center = None
        self.background_points = []
        self.disk_circle = None
        self.disk_method = 'unassigned'
        self.mask = None
        rel = path.relative_to(self.input_root)
        try:
            self.image = read_image(path)
        except Exception as exc:
            self.image = self.mask = None
            messagebox.showerror('読み込みエラー', str(exc))
            self.status.set(f'{self.index + 1}/{len(self.paths)}  {rel} : 読み込み失敗、保留可能')
            self.draw()
            return
        row = self.records.get(str(rel))
        if row and row['review_status'] == 'approved':
            self.circle = tuple(float(row[k]) for k in ('plate_center_x', 'plate_center_y', 'plate_radius_px'))
            self.mask = read_image(self.input_root / row.get('mycelium_mask_file', row['mask_file']))[:, :, 0] > 0
            self.include_disk.set(row.get('disk_included', False))
            if row.get('disk_radius_px'):
                self.disk_circle = tuple(float(row[k]) for k in ('disk_center_x', 'disk_center_y', 'disk_radius_px'))
                self.disk_method = row.get('disk_position_method', 'manual_three_points')
            self.threshold.set(row['brightness_threshold'])
            self.inner.set(row['inner_fraction'])
            self.diameter.set(row['plate_diameter_mm'])
            self.background = row['background_L']
            self.mask_threshold = row['brightness_threshold']
            self.mask_inner = row['inner_fraction']
            self.mask_mode = row.get('analysis_mode', 'white')
            self.mode.set(next(label for label, value in ANALYSIS_MODES.items() if value == self.mask_mode))
            self.background_points = json.loads(row.get('background_points_json', '[]'))
            self.mask_background_points = tuple(tuple(p) for p in self.background_points)
            self.edited = row['manual_edited']
            self.circle_method = row['circle_method']
            if row.get('colony_center_x') is not None:
                self.colony_center = (row['colony_center_x'], row['colony_center_y'])
            self.checked.set(True)
            for key, var in self.metadata.items():
                var.set(row[key])
        else:
            self.circle_method = 'auto'
            try:
                self.circle = detect_circle(self.image)
            except ValueError:
                h, w = self.image.shape[:2]
                self.circle = (w / 2, h / 2, min(h, w) * 0.44)
                self.circle_method = 'fallback'
                messagebox.showwarning('円を指定', '自動検出できませんでした。外周の3点を指定してください。')
            for var in self.metadata.values():
                var.set('')
            day = folder_day(path.parent)
            self.metadata['treatment'].set(rel.parent.as_posix() if rel.parent != Path('.') and not day else '')
            self.metadata['specimen_id'].set(path.stem)
            self.metadata['timepoint'].set(day)
            self.metadata['time_unit'].set('day' if day else '')
            self.extract()
        self.status.set(f'{self.index + 1}/{len(self.paths)}  {rel}')
        self.save_location.set(f'この写真の保存先: {self.folder_output(rel)}')
        self.draw()

    def extract(self):
        if self.image is None:
            return
        if self.edited and not messagebox.askyesno('再抽出', '手直しをリセットして抽出し直しますか？'):
            return
        try:
            self.mask, self.background = segment_mycelium(
                self.image, self.circle, self.threshold.get(), self.inner.get(), self.colony_center,
                ANALYSIS_MODES[self.mode.get()], self.background_points)
            self.edited = False
            self.mask_threshold = self.threshold.get()
            self.mask_inner = self.inner.get()
            self.mask_mode = ANALYSIS_MODES[self.mode.get()]
            self.mask_background_points = tuple(tuple(p) for p in self.background_points)
            if self.include_disk.get() and self.reuse_disk.get() and self.disk_template_mm is not None:
                if self.disk_circle is None or self.disk_method.startswith('auto_') or self.disk_method == 'plate_center_fallback':
                    radius = self.disk_template_mm / self.diameter.get() * self.circle[2]
                    center, self.disk_method = locate_disk_center(
                        self.image, self.circle, self.mask, radius, self.colony_center)
                    self.disk_circle = (*center, radius)
            self.draw()
        except Exception as exc:
            messagebox.showerror('抽出エラー', str(exc))

    def set_tool(self, tool):
        if self.image is None:
            return
        if tool == 'disk_center' and self.disk_circle is None:
            messagebox.showinfo('ディスク径の指定', '最初の1枚は「ディスクの輪郭を3点で指定」を使ってください。')
            return
        self.tool, self.points = tool, []
        names = {'view': 'プレビュー', 'add': '菌糸を塗り足す', 'erase': '拾いすぎを消す', 'circle': '外周の離れた3点をクリック',
                 'seed': '左の元画像で菌叢の中心をクリック',
                 'disk': '左の元画像でディスク輪郭上の離れた3点をクリック',
                 'disk_center': '左の元画像でディスクの中心をクリック',
                 'background': '左の元画像で菌叢・反射のない培地を1〜3点クリック。終了は「修正操作を終了」'}
        self.status.set(f'{self.index + 1}/{len(self.paths)}  {names[tool]}')

    def start_circle(self):
        if self.image is not None:
            self.set_tool('circle')

    def clear_background(self):
        if self.image is None:
            return
        if self.edited and not messagebox.askyesno('再抽出', '手直しをリセットして背景を変更しますか？'):
            return
        self.background_points = []
        self.edited = False
        self.extract()

    def image_point(self, event):
        if self.image is None:
            return None
        x = (event.x - self.origin[0]) / self.scale
        y = (event.y - self.origin[1]) / self.scale
        h, w = self.image.shape[:2]
        return (min(w-1, round(x)), min(h-1, round(y))) if 0 <= x < w and 0 <= y < h else None

    def click(self, event):
        point = self.image_point(event)
        if point is None:
            return
        if self.tool == 'background':
            if not circle_roi(self.image.shape[:2], self.circle, self.inner.get())[point[1], point[0]]:
                messagebox.showwarning('背景の位置', 'オレンジ円の内側の培地を選んでください。')
                return
            if len(self.background_points) >= 3:
                messagebox.showinfo('背景の指定', '背景は3点までです。変更する場合は「背景を自動指定に戻す」を押してください。')
                return
            if self.edited and not messagebox.askyesno('再抽出', '手直しをリセットして背景を変更しますか？'):
                return
            self.background_points.append(point)
            self.edited = False
            self.extract()
        elif self.tool == 'seed':
            if self.edited and not messagebox.askyesno('再抽出', '手直しをリセットして中心を変更しますか？'):
                return
            self.colony_center = point
            self.edited = False
            self.extract()
            self.tool = 'view'
        elif self.tool == 'disk_center':
            candidate = (*point, self.disk_circle[2])
            try:
                include_disk_mask(self.mask, candidate, self.circle, self.mask_inner)
                self.disk_circle = candidate
                self.disk_method = 'manual_center'
                self.tool = 'view'
                self.draw()
            except ValueError as exc:
                messagebox.showerror('ディスクの位置', str(exc))
        elif self.tool == 'disk':
            self.points.append(point)
            if len(self.points) == 3:
                try:
                    disk = circle_from_points(self.points)
                    if not 1 <= disk[2] < self.circle[2] * 0.5:
                        raise ValueError('ディスクの円が不自然です。ディスク輪郭上の離れた3点を選んでください。')
                    include_disk_mask(self.mask, disk, self.circle, self.mask_inner)
                    self.disk_circle = disk
                    self.disk_method = 'manual_three_points'
                    self.include_disk.set(True)
                    self.tool = 'view'
                except ValueError as exc:
                    messagebox.showerror('ディスクの指定', str(exc))
                self.points = []
            self.draw()
        elif self.tool == 'circle':
            self.points.append(point)
            if len(self.points) == 3:
                try:
                    circle = circle_from_points(self.points)
                    h, w = self.image.shape[:2]
                    if not (0 < circle[0] < w and 0 < circle[1] < h and 10 < circle[2] < min(h, w) * 0.60):
                        raise ValueError('円が不自然です。外周の離れた3点を選び直してください。')
                    self.circle = circle
                    self.circle_method = 'three_points'
                    self.checked.set(True)
                    self.edited = False
                    self.extract()
                    self.tool = 'view'
                except ValueError as exc:
                    messagebox.showerror('円の指定', str(exc))
                self.points = []
            self.draw()
        else:
            self.paint(point)

    def drag(self, event):
        point = self.image_point(event)
        if point is not None:
            self.paint(point)

    def paint(self, point):
        if self.tool not in {'add', 'erase'} or self.mask is None:
            return
        image = self.mask.astype(np.uint8)
        value = 1 if self.tool == 'add' else 0
        radius = self.brush.get()
        cv2.line(image, self.last_point or point, point, value, radius * 2)
        cv2.circle(image, point, radius, value, -1)
        self.mask = (image > 0) & circle_roi(image.shape, self.circle, self.mask_inner)
        self.last_point, self.edited = point, True
        self.draw()

    def draw(self):
        self.canvas.delete('all')
        if self.image is None or self.mask is None:
            return
        h, w = self.image.shape[:2]
        cw, ch = max(1, self.canvas.winfo_width()), max(1, self.canvas.winfo_height())
        self.scale = min((cw - 12) / (2 * w), (ch - 32) / h)
        if self.scale <= 0:
            return
        size = (max(1, round(w * self.scale)), max(1, round(h * self.scale)))
        self.origin = (4, 28)
        error = ''
        try:
            measured_mask, _ = self.measured_masks()
        except ValueError as exc:
            measured_mask = self.mask
            error = str(exc)
        overlay = overlay_image(self.image, measured_mask, self.circle, self.mask_inner)
        self.photos = []
        for i, (image, title) in enumerate([(self.image, '元画像'), (overlay, '緑：面積に数える画素')]):
            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            photo = ImageTk.PhotoImage(Image.fromarray(rgb).resize(size))
            self.photos.append(photo)
            x = 4 + i * (size[0] + 4)
            self.canvas.create_text(x + 3, 12, text=title, fill='white', anchor='w')
            self.canvas.create_image(x, 28, image=photo, anchor='nw')
        for x, y in self.points:
            px, py = x * self.scale + 4, y * self.scale + 28
            self.canvas.create_oval(px-3, py-3, px+3, py+3, fill='red')
        for x, y in self.background_points:
            px, py = x*self.scale + 4, y*self.scale + 28
            r = max(3, self.circle[2] * 0.025)*self.scale
            self.canvas.create_oval(px-r, py-r, px+r, py+r, outline='#30bfff', width=2)
        if self.colony_center is not None:
            px = self.colony_center[0] * self.scale + 4
            py = self.colony_center[1] * self.scale + 28
            self.canvas.create_line(px-5, py, px+5, py, fill='red')
            self.canvas.create_line(px, py-5, px, py+5, fill='red')
        if self.disk_circle is not None and self.include_disk.get():
            dx, dy, dr = self.disk_circle
            for offset in (0, size[0] + 4):
                x, y = dx*self.scale + 4 + offset, dy*self.scale + 28
                r = dr*self.scale
                self.canvas.create_oval(x-r, y-r, x+r, y+r, outline='#ff80ce')
        try:
            value = area_mm2(measured_mask, self.circle, self.diameter.get())
            methods = {'auto_dark_center': '暗い中心から推定', 'auto_colony_center': '菌糸の位置から推定',
                       'plate_center_fallback': '仮位置・確認してください', 'manual_center': '中心を手動修正',
                       'manual_three_points': '輪郭を手動指定', 'unassigned': '未指定'}
            self.disk_status.set('ディスク径: 最初の1枚で指定' if self.disk_circle is None else
                                 f'ディスク径 {self.disk_circle[2]/self.circle[2]*self.diameter.get():.2f} mm\n'
                                 f'位置: {methods.get(self.disk_method, self.disk_method)}')
            label = '総面積（菌糸＋ディスク）' if self.include_disk.get() else '菌糸面積'
            self.area_text.set(error or ('ディスクの輪郭を指定してください' if self.include_disk.get() and self.disk_circle is None
                                        else f'{label} {value:.2f} mm²'))
        except (tk.TclError, ValueError):
            self.area_text.set('直径を確認してください')

    def folder_output(self, relative_path):
        return self.input_root / Path(relative_path).parent / RESULT_FOLDER / self.output.name

    def measured_masks(self):
        return include_disk_mask(self.mask, self.disk_circle if self.include_disk.get() else None,
                                 self.circle, self.mask_inner)

    def flush(self):
        rows = list(self.records.values())
        save_rows(self.output / 'measurements_all.csv', rows)
        folders = {}
        for row in rows:
            destination = self.folder_output(row['relative_path'])
            folders.setdefault(destination, []).append(row)
        for destination, local_rows in folders.items():
            destination.mkdir(parents=True, exist_ok=True)
            save_rows(destination / 'measurements.csv', local_rows)

    def approve(self):
        if self.image is None or self.mask is None:
            return
        if self.circle_method == 'fallback':
            messagebox.showwarning('円の指定', 'シャーレの自動検出に失敗しています。「外周の3点を指定」で修正してください。')
            return
        if (self.threshold.get() != self.mask_threshold or self.inner.get() != self.mask_inner
                or ANALYSIS_MODES[self.mode.get()] != self.mask_mode
                or tuple(tuple(p) for p in self.background_points) != self.mask_background_points):
            messagebox.showwarning('再抽出', '設定が変わっています。「この設定で抽出し直す」を押してください。')
            return
        if self.include_disk.get() and self.disk_circle is None:
            messagebox.showwarning('ディスクの指定', '「ディスクの輪郭を3点で指定」で、中央のディスクを指定してください。')
            return
        try:
            diameter = self.diameter.get()
            measured_mask, disk_mask = self.measured_masks()
            area = area_mm2(measured_mask, self.circle, diameter)
            if area == 0 and not messagebox.askyesno('ゼロ面積', '抽出面積が0です。この結果を確定しますか？'):
                return
            rel = self.paths[self.index].relative_to(self.input_root)
            destination = self.folder_output(rel)
            # 元の拡張子も残し、同名のJPEGとPNGがある場合の衝突を防ぎます。
            mask_path = (destination / 'masks' / (rel.name + '_mask.png')).relative_to(self.input_root)
            mycelium_path = (destination / 'masks' / (rel.name + '_mycelium_mask.png')).relative_to(self.input_root)
            overlay_path = (destination / 'overlays' / (rel.name + '_overlay.png')).relative_to(self.input_root)
            write_image(self.input_root / mask_path, measured_mask.astype(np.uint8) * 255)
            write_image(self.input_root / mycelium_path, self.mask.astype(np.uint8) * 255)
            write_image(self.input_root / overlay_path, overlay_image(self.image, measured_mask, self.circle, self.mask_inner))
            row = {key: var.get().strip() for key, var in self.metadata.items()}
            row.update(image_name=rel.name, photo_folder=self.paths[self.index].parent.name,
                       relative_path=rel.as_posix(), review_status='approved', area_mm2=round(area, 6),
                       area_px=int(measured_mask.sum()), plate_diameter_mm=diameter,
                       plate_center_x=self.circle[0], plate_center_y=self.circle[1], plate_radius_px=self.circle[2],
                       mm_per_pixel=diameter/(2*self.circle[2]), width_px=self.image.shape[1], height_px=self.image.shape[0],
                       brightness_threshold=self.mask_threshold, inner_fraction=self.mask_inner,
                       background_L=self.background, manual_edited=self.edited, circle_method=self.circle_method,
                       mask_file=mask_path.as_posix(), overlay_file=overlay_path.as_posix(),
                       segmentation_method='lab_background_connected_v3',
                       colony_center_x=(self.colony_center or self.circle[:2])[0],
                       colony_center_y=(self.colony_center or self.circle[:2])[1],
                       disk_included=self.include_disk.get(),
                       disk_center_x=self.disk_circle[0] if self.disk_circle else '',
                       disk_center_y=self.disk_circle[1] if self.disk_circle else '',
                       disk_radius_px=self.disk_circle[2] if self.disk_circle else '',
                       disk_mm2=round(area_mm2(disk_mask, self.circle, diameter), 6),
                       mycelium_excluding_disk_mm2=round(area_mm2(self.mask & ~disk_mask, self.circle, diameter), 6),
                       mycelium_mask_file=mycelium_path.as_posix(),
                       disk_diameter_mm=self.disk_circle[2]/self.circle[2]*diameter if self.disk_circle else '',
                       disk_position_method=self.disk_method,
                       analysis_mode=self.mask_mode,
                       background_method='manual_patches' if self.background_points else 'auto_annulus',
                       background_points_json=json.dumps(self.background_points))
            self.records[str(rel)] = row
            self.flush()
            self.checked.set(True)
            if self.include_disk.get() and self.disk_circle is not None and self.reuse_disk.get():
                self.disk_template_mm = self.disk_circle[2]/self.circle[2]*diameter

            self.next()
        except Exception as exc:
            messagebox.showerror('保存エラー', str(exc))

    def skip(self):
        if not self.paths:
            return
        rel = self.paths[self.index].relative_to(self.input_root)
        if str(rel) not in self.records:
            row = {key: var.get().strip() for key, var in self.metadata.items()}
            row.update(image_name=rel.name, photo_folder=self.paths[self.index].parent.name,
                       relative_path=rel.as_posix(), review_status='pending')
            self.records[str(rel)] = row
            self.flush()
        self.next()

    def next(self):
        if self.index + 1 < len(self.paths):
            self.index += 1
            self.load()
        else:
            approved = sum(row['review_status'] == 'approved' for row in self.records.values())
            messagebox.showinfo('確認状況', f'確認済み {approved}/{len(self.paths)} 枚\n保存先: {self.output}\n保留画像は「前へ」で戻れます。')

    def previous(self):
        if self.paths and self.index > 0:
            self.index -= 1
            self.load()


if __name__ == '__main__':
    root = tk.Tk()
    ColonyApp(root)
    root.mainloop()
