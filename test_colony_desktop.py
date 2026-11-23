"""計算と保存の検証: python -m unittest test_colony_desktop.py"""
import csv
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import tkinter as tk

import cv2
import numpy as np

from measure_colony_desktop import ColonyApp, area_mm2, circle_from_points, collect_photos, detect_circle, folder_day, include_disk_mask, save_rows, segment_mycelium, write_image


class MeasurementTests(unittest.TestCase):
    def test_circle_calibration(self):
        circle = circle_from_points([(300, 30), (570, 300), (300, 570)])
        np.testing.assert_allclose(circle, (300, 300, 270))
        with self.assertRaises(ValueError):
            circle_from_points([(0, 0), (10, 10), (20, 20)])

    def test_area_scaling(self):
        mask = np.ones((20, 20), bool)
        self.assertAlmostEqual(area_mm2(mask, (300, 300, 270), 90), 400/36)
        self.assertAlmostEqual(area_mm2(mask, (300, 300, 135), 90), 4*400/36)

    def test_disk_fills_dark_center_without_counting_overlap_twice(self):
        mask = np.zeros((600, 600), np.uint8)
        cv2.circle(mask, (300, 300), 40, 1, -1)
        cv2.circle(mask, (300, 300), 15, 0, -1)
        measured, disk = include_disk_mask(mask > 0, (300, 300, 20), (300, 300, 270), 0.85)
        self.assertTrue(measured[300, 300])
        self.assertFalse(measured[100, 100])
        self.assertEqual(int(measured.sum()), int(mask.sum()) + int((disk & ~(mask > 0)).sum()))
        without_disk = (mask > 0) & ~disk
        self.assertAlmostEqual(area_mm2(measured, (300, 300, 270), 90),
                               area_mm2(without_disk, (300, 300, 270), 90) + area_mm2(disk, (300, 300, 270), 90))
        excluded, _ = include_disk_mask(mask > 0, None, (300, 300, 270), 0.85)
        np.testing.assert_array_equal(excluded, mask > 0)
        with self.assertRaises(ValueError):
            include_disk_mask(mask > 0, (520, 300, 30), (300, 300, 270), 0.85)

    def test_segmentation_keeps_gaps_and_excludes_rim(self):
        image = np.full((600, 600, 3), 100, np.uint8)
        cv2.circle(image, (300, 300), 50, (220, 220, 220), -1)
        cv2.circle(image, (300, 300), 15, (100, 100, 100), -1)
        cv2.circle(image, (300, 300), 265, (255, 255, 255), 8)
        mask, _ = segment_mycelium(image, (300, 300, 270), 16, 0.90)
        self.assertTrue(mask[300, 330])
        self.assertFalse(mask[300, 300])
        self.assertFalse(mask[300, 565])
        stricter, _ = segment_mycelium(image, (300, 300, 270), 80, 0.90)
        self.assertLessEqual(stricter.sum(), mask.sum())

    def test_circle_on_full_and_distant_photos(self):
        for shape, center, radius in [((600, 600), (300, 300), 270), ((900, 600), (300, 650), 180)]:
            image = np.full((*shape, 3), 20, np.uint8)
            cv2.circle(image, center, radius, (140, 140, 140), -1)
            x, y, r = detect_circle(image)
            self.assertLess(math.hypot(x-center[0], y-center[1]), 3)
            self.assertAlmostEqual(r, radius, delta=3)

    def test_pale_mycelium_under_uneven_light_without_remote_glare(self):
        yy, xx = np.indices((600, 600))
        light = (100 + xx * 0.08 + yy * 0.04).astype(np.uint8)
        lab = np.full((600, 600, 3), 128, np.uint8)
        lab[:, :, 0] = light
        # 明確な菌糸、つながる薄い先端、離れた強い反射を配置します。
        lab[285:315, 285:315, 0] += 35
        lab[295:305, 315:350, 0] += 9
        lab[400:425, 430:455, 0] += 50
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        mask, _ = segment_mycelium(image, (300, 300, 270), 12, 0.90)
        self.assertTrue(mask[300, 300])
        self.assertTrue(mask[300, 340])
        self.assertFalse(mask[412, 442])
        self.assertFalse(mask[200, 300])

    def test_offset_colony_with_explicit_center(self):
        image = np.full((600, 600, 3), 100, np.uint8)
        cv2.circle(image, (150, 300), 20, (220, 220, 220), -1)
        automatic, _ = segment_mycelium(image, (300, 300, 270), 12, 0.90)
        selected, _ = segment_mycelium(image, (300, 300, 270), 12, 0.90, (150, 300))
        self.assertFalse(automatic[300, 150])
        self.assertTrue(selected[300, 150])

    def test_uniform_agar_has_no_detected_colony(self):
        image = np.full((600, 600, 3), 100, np.uint8)
        mask, _ = segment_mycelium(image, (300, 300, 270), 12, 0.90)
        self.assertEqual(int(mask.sum()), 0)

    def test_mixed_mode_detects_dark_and_brown_colony_not_remote_glare(self):
        lab = np.full((600, 600, 3), 128, np.uint8)
        lab[:, :, 0] = 155
        cv2.circle(lab, (300, 300), 50, (85, 128, 128), -1)
        cv2.circle(lab, (345, 300), 35, (145, 137, 155), -1)
        cv2.circle(lab, (470, 380), 20, (220, 128, 128), -1)
        image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        white, _ = segment_mycelium(image, (300, 300, 270), 18, 0.85, analysis_mode='white')
        mixed, _ = segment_mycelium(image, (300, 300, 270), 18, 0.85, analysis_mode='mixed')
        self.assertFalse(white[300, 300])
        self.assertTrue(mixed[300, 300])
        self.assertTrue(mixed[300, 365])
        self.assertFalse(mixed[380, 470])
        self.assertFalse(mixed[150, 300])

    def test_explicit_background_patch_with_large_dark_colony(self):
        image = np.full((600, 600, 3), 150, np.uint8)
        cv2.circle(image, (300, 300), 175, (75, 75, 75), -1)
        mask, _ = segment_mycelium(image, (300, 300, 270), 18, 0.90,
                                   analysis_mode='mixed', background_points=[(100, 300)])
        self.assertTrue(mask[300, 300])
        self.assertTrue(mask[300, 460])
        self.assertFalse(mask[300, 100])
        with self.assertRaises(ValueError):
            segment_mycelium(image, (300, 300, 270), 18, 0.90, analysis_mode='mixed', background_points=[(0, 0)])

    def test_csv_retains_replicates_and_pending_blank(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'measurements.csv'
            save_rows(path, [{'relative_path': 'C1/0.jpg', 'treatment': 'Control', 'replicate': '1',
                              'area_mm2': 123.45, 'review_status': 'approved'},
                             {'relative_path': 'C2/0.jpg', 'review_status': 'pending'}])
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows[0]['replicate'], '1')
            self.assertEqual(rows[1]['area_mm2'], '')

    def test_gui_save_and_reload_reviewed_mask(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                image = np.full((600, 600, 3), 20, np.uint8)
                cv2.circle(image, (300, 300), 270, (110, 110, 110), -1)
                cv2.circle(image, (300, 300), 50, (230, 230, 230), -1)
                cv2.circle(image, (300, 300), 15, (60, 60, 60), -1)
                write_image(base / 'sample.png', image)
                app = ColonyApp(root)
                app.input_root, app.output = base, base / 'output'
                app.output.mkdir()
                app.paths = [base / 'sample.png']
                app.mode.set('白色・明るい菌叢')
                with patch('measure_colony_desktop.messagebox.showinfo'), patch('measure_colony_desktop.messagebox.showerror') as error:
                    app.load()
                    app.checked.set(True)
                    app.set_tool('disk')
                    with patch.object(app, 'image_point', side_effect=[(300, 280), (320, 300), (300, 320)]):
                        for _ in range(3):
                            app.click(None)
                    self.assertIsNotNone(app.disk_circle)
                    self.assertFalse(app.mask[300, 300])
                    self.assertTrue(app.measured_masks()[0][300, 300])
                    app.metadata['treatment'].set('Control')
                    app.approve()
                    error.assert_not_called()
                    app.load()
                    self.assertTrue(app.checked.get())
                    self.assertTrue(app.include_disk.get())
                    self.assertTrue(app.measured_masks()[0][300, 300])
                    self.assertFalse(app.mask[300, 300])
                    self.assertEqual(app.mask_inner, app.inner.get())
                    self.assertEqual(app.metadata['treatment'].get(), 'Control')
                    self.assertTrue((app.output / 'measurements_all.csv').exists())
        finally:
            root.destroy()

    def test_day_folders_preserve_names_and_exclude_outputs(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                image = np.full((600, 600, 3), 20, np.uint8)
                cv2.circle(image, (300, 300), 270, (110, 110, 110), -1)
                cv2.circle(image, (300, 300), 50, (230, 230, 230), -1)
                for day in ('6日目', '9日目'):
                    write_image(base / day / 'sample.png', image)
                app = ColonyApp(root)
                app.input_root = base
                app.output = base / '解析結果' / 'reviewed_test'
                app.output.mkdir(parents=True)
                app.paths = collect_photos(base)
                with patch('measure_colony_desktop.messagebox.showinfo'), patch('measure_colony_desktop.messagebox.showerror') as error:
                    app.load()
                    self.assertEqual(app.metadata['timepoint'].get(), '6')
                    app.checked.set(True)
                    app.disk_circle = (300, 300, 20)
                    app.approve()
                    self.assertEqual(app.metadata['timepoint'].get(), '9')
                    app.checked.set(True)
                    app.disk_circle = (300, 300, 20)
                    app.approve()
                    error.assert_not_called()
                for day in ('6日目', '9日目'):
                    destination = base / day / '解析結果' / 'reviewed_test'
                    self.assertTrue((destination / 'overlays' / 'sample.png_overlay.png').exists())
                    with (destination / 'measurements.csv').open(encoding='utf-8-sig') as handle:
                        rows = list(csv.DictReader(handle))
                    self.assertEqual(len(rows), 1)
                    self.assertEqual(rows[0]['image_name'], 'sample.png')
                    self.assertEqual(rows[0]['photo_folder'], day)
                self.assertEqual(len(collect_photos(base)), 2)
                with (app.output / 'measurements_all.csv').open(encoding='utf-8-sig') as handle:
                    self.assertEqual(len(list(csv.DictReader(handle))), 2)
                self.assertEqual(folder_day(Path('experiment') / '6日目' / 'Control'), '6')
        finally:
            root.destroy()

    def test_approve_next_reuses_physical_disk_size_and_finds_new_center(self):
        root = tk.Tk()
        root.withdraw()
        try:
            with tempfile.TemporaryDirectory() as folder:
                base = Path(folder)
                for name, plate_radius, center, disk_radius in [
                    ('1.png', 270, (300, 300), 16), ('2.png', 210, (330, 290), 12)]:
                    image = np.full((600, 600, 3), 20, np.uint8)
                    cv2.circle(image, (300, 300), plate_radius, (110, 110, 110), -1)
                    cv2.circle(image, center, 45, (210, 210, 210), -1)
                    cv2.circle(image, center, disk_radius, (45, 45, 45), -1)
                    write_image(base / name, image)
                app = ColonyApp(root)
                app.input_root, app.output = base, base / 'output'
                app.output.mkdir()
                app.paths = [base / '1.png', base / '2.png']
                with patch('measure_colony_desktop.messagebox.showinfo'), \
                     patch('measure_colony_desktop.messagebox.showerror') as error, \
                     patch('measure_colony_desktop.messagebox.showwarning') as warning:
                    app.load()
                    app.disk_circle = (300, 300, 20)
                    app.disk_method = 'manual_three_points'
                    app.approve()
                    self.assertEqual(app.index, 1)
                    self.assertEqual(app.disk_method, 'auto_dark_center')
                    self.assertAlmostEqual(app.disk_circle[0], 330, delta=1)
                    self.assertAlmostEqual(app.disk_circle[1], 290, delta=1)
                    self.assertAlmostEqual(app.disk_circle[2] / app.circle[2] * app.diameter.get(), app.disk_template_mm)
                    first = app.records['1.png']
                    self.assertAlmostEqual(app.disk_template_mm, 20 / first['plate_radius_px'] * 90)
                    self.assertLess(app.disk_circle[2], 20)
                    self.assertFalse(app.checked.get())
                    app.approve()
                    self.assertEqual(app.records['2.png']['review_status'], 'approved')
                    warning.assert_not_called()
                    error.assert_not_called()
        finally:
            root.destroy()


if __name__ == '__main__':
    unittest.main()
