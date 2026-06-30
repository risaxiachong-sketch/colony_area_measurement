# シャーレ菌叢面積解析 Colab版 README

このノートブックは、シャーレ写真から菌叢の広がり面積を一括測定するための実行用ノートです。

測定しているのは、写真を上から見たときの投影面積です。菌糸の高さや凹凸を含む3D表面積ではありません。

## 使うファイル

- `colab_run_colony_area.ipynb`
  - 実行用ノートブックです。
  - 上から順番にセルを実行すると、画像を一括解析してCSVと確認画像を保存します。

## 入力データ

Google Driveに、解析したい画像を入れたフォルダを作ります。

例:

```text
MyDrive/colony_images/day11/iCloud写真
```

対応している画像形式:

```text
.jpg
.jpeg
.png
.tif
.tiff
.bmp
```

## 出力データ

解析後、指定した出力フォルダに以下が保存されます。

```text
colony_area_results.csv
overlays/
masks/
```

主な出力:

- `colony_area_results.csv`
  - 各画像の測定結果をまとめたCSVです。
- `overlays/`
  - 元画像に解析結果を色で重ねた確認用画像です。
- `masks/`
  - 菌糸部分と黒い中心部の白黒マスク画像です。

## CSVの主な列

- `filename`
  - 画像ファイル名
- `mycelium_mm2`
  - 白からベージュの菌糸部分の面積
- `dark_core_mm2`
  - 中央の黒っぽい部分の面積
- `total_colony_mm2`
  - 菌糸部分と黒い中心部を合わせた面積
- `plate_radius_px`
  - 画像上で検出されたシャーレ半径
- `pixel_mm`
  - 1ピクセルが何mmに相当するか
- `overlay_path`
  - 確認用オーバーレイ画像の保存先

## 使い方

1. Google Driveに画像フォルダを用意します。
2. `colab_run_colony_area.ipynb` をGoogle Colabで開きます。
3. 最初のセルから順番に実行します。
4. 設定セルで `PHOTO_FOLDER` と `OUTPUT_FOLDER` を自分のDrive内のパスに変更します。
5. 一括解析セルを実行します。
6. `overlays` の画像を見て、緑色と紫色の認識結果が妥当か確認します。

## 調整する値

菌糸を拾いすぎる場合:

```python
color_delta = 28.0
yellow_delta = 7.0
```

菌糸を拾い足りない場合:

```python
color_delta = 18.0
yellow_delta = 3.0
```

シャーレの縁を拾う場合:

```python
inner_plate_frac = 0.78
```

## 色の意味

オーバーレイ画像では、次の色を使っています。

- 緑: 菌糸部分
- 紫: 黒い中心部
- 黄色の円: 検出したシャーレ

## 注意

Colabでは、Windowsのローカルパスは使えません。

使えない例:

```text
C:\Users\risa4\OneDrive\...
```

ColabではGoogle Driveをマウントして、次のようなパスを使います。

```text
/content/drive/MyDrive/...
```

