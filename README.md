# 菌叢面積計測スクリプトの使い方

このスクリプトは、シャーレ写真から菌叢の面積を自動計測します。

大まかな流れは次の通りです。

1. 画像を読み込む
2. シャーレの円を見つける
3. シャーレ直径から「1ピクセルが何 mm か」を計算する
4. シャーレ内の培地色を背景として推定する
5. 背景と色や明るさが違う部分を菌叢候補として抽出する
6. 小さなノイズを消す
7. 中央付近の菌叢だけを残す
8. 面積を `mm2` に変換して CSV に保存する
9. どこを菌叢として認識したかを確認用画像に色で重ねる

## 実行方法

初回だけ、OpenCVが入っていない場合は次を実行してください。

```powershell
python -m pip install opencv-python numpy
```

## まず使うファイル

初心者向けには、まず `run_easy.py` を使うのがおすすめです。

このファイルは上のほうに「ここだけ変えれば使えます」という場所があります。
写真フォルダ、出力フォルダ、シャーレ直径をそこに書いて、VS Codeの実行ボタンで動かせます。
実行が終わると `results/overlays` フォルダが自動で開くので、結果画像をすぐに確認できます。

```powershell
python .\colony_area_measurement\run_easy.py
```

PowerShellで以下のように実行します。

```powershell
cd C:\Users\risa4
python .\colony_area_measurement\measure_colony_area.py "画像フォルダのパス" --plate-diameter-mm 90 --output "出力フォルダのパス"
```

今回の写真フォルダなら例はこれです。

```powershell
python .\colony_area_measurement\measure_colony_area.py "C:\Users\risa4\OneDrive\ドキュメント\Office のカスタム テンプレート\デスクトップ\植物病理\スコポレチン培地試験　2回目　9日目\iCloud写真" --plate-diameter-mm 90 --output "C:\Users\risa4\colony_area_measurement\results"
```

画像を1枚だけ測る場合は、フォルダではなく画像ファイルを指定します。

```powershell
python .\colony_area_measurement\measure_colony_area.py "C:\path\to\IMG_0001.JPEG" --plate-diameter-mm 90 --output "C:\Users\risa4\colony_area_measurement\results"
```

## 出力されるもの

`colony_area_measurements.csv`

面積の数値が入ったCSVです。Excelで開けます。

`overlays` フォルダ

認識結果を重ねた確認用画像です。

色の意味は次の通りです。

- 水色の円: 自動検出したシャーレ
- 緑: 白からベージュの菌糸部分
- 紫: 中央の黒っぽい接種片や暗い部分
- オレンジ: 総菌叢領域として埋められた部分

## CSVの主な列

`mycelium_mm2`

白からベージュの菌糸として認識された面積です。今回の写真では、この値を主に見るのがおすすめです。

`dark_core_mm2`

中央の黒っぽい接種片や暗い部分の面積です。

`colony_mm2`

菌糸と中央部を合わせた総面積です。隙間を埋める処理も入るので、菌糸だけを見たい場合は `mycelium_mm2` を使ってください。

`plate_radius_px`

画像上で検出されたシャーレ半径です。シャーレ検出がずれていないかの確認に使います。

## よく調整する値

菌糸を拾いすぎるとき:

```powershell
--color-delta 28 --yellow-delta 7
```

菌糸を拾い足りないとき:

```powershell
--color-delta 18 --yellow-delta 3
```

シャーレの縁や周辺を拾うとき:

```powershell
--inner-plate-frac 0.78
```

中央から離れたノイズを拾うとき:

```powershell
--keep-center-frac 0.30
```

シャーレ検出が失敗するとき:

確認用画像を見て、シャーレの中心座標と半径を手で指定できます。

```powershell
--plate 776 1586 370
```

数字は `中心X 中心Y 半径px` の順です。

## 新しい写真を追加するとき

同じ撮り方の写真なら、同じフォルダに画像を追加して、フォルダ指定で実行すればまとめて測れます。

違う実験日の写真なら、新しいフォルダを指定して実行してください。

写真の撮り方が大きく変わると、しきい値の調整が必要になることがあります。まずは `overlays` の画像を見て、緑色が菌糸に合っているか確認してください。
