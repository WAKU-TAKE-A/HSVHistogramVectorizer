# HSVHistogramVectorizer

HSV 色空間の 3D ヒストグラムを使って、画像 crop から特徴ベクトルを作り、類似度を比較するための小さな Python ツールです。

Current version: `v0.9.0`

前提として、入力画像はすでに対象物が切り出され、必要なら OBB などで水平矩形化された `BGR` 画像であることを想定しています。

## できること

- `HSVHistogramVectorizer` で画像から HSV ヒストグラムベクトルを抽出
- ベクトル同士の類似度を比較
- `compare_samples.py` で `sample/ref.png` と `sample*.png` をまとめて比較

## 必要環境

- Python 3.11 以上推奨
- `opencv-python`
- `numpy`

インストール例:

```bash
pip install numpy opencv-python
```

## ファイル構成

```text
hsv_histogram_vectorizer.py  # ベクトル化と類似度比較の本体
compare_samples.py         # サンプル画像比較用の CLI
sample/                    # 比較用サンプル画像
```

## 使い方

### 1. サンプル比較を実行する

```bash
python compare_samples.py
```

デフォルトでは `sample/ref.png` を基準画像として、`sample/sample*.png` を `intersection` 方式で比較します。

全比較方式を並べて確認したい場合:

```bash
python compare_samples.py --all-methods
```

比較方式を指定する場合:

```bash
python compare_samples.py --method cosine
```

中心領域の使い方や bin 数を変える場合:

```bash
python compare_samples.py --h-bins 16 --s-bins 8 --v-bins 8 --center-scale 0.85 --center-crop-mode rect
```

## ライブラリとして使う

```python
import cv2

from hsv_histogram_vectorizer import (
    HSVHistogramVectorizer,
    HSVHistogramVectorizerConfig,
)

image_a = cv2.imread("a.png")
image_b = cv2.imread("b.png")

config = HSVHistogramVectorizerConfig(
    h_bins=16,
    s_bins=8,
    v_bins=8,
    center_scale=0.85,
    center_crop_mode="rect",
    normalize="l1",
    compare_method="intersection",
)

vectorizer = HSVHistogramVectorizer(config)

vec_a = vectorizer.extract(image_a)
vec_b = vectorizer.extract(image_b)
score = vectorizer.compare(vec_a, vec_b)

print(vec_a.shape)
print(score)
```

## 主な設定

`HSVHistogramVectorizerConfig` で以下を調整できます。

- `h_bins`, `s_bins`, `v_bins`: HSV 各チャンネルの bin 数
- `center_scale`: 中央領域を何割使うか
- `center_crop_mode`: `rect` / `ellipse` / `none`
- `normalize`: `l1` / `l2` / `none`
- `compare_method`: `intersection` / `cosine` / `correlation` / `bhattacharyya` / `chisqr`

## 類似度の見方

- `intersection`, `cosine`, `correlation` は高いほど類似
- `bhattacharyya`, `chisqr` は距離を内部で類似度に変換しており、高いほど類似
- 返り値は基本的に `0.0` から `1.0` に近い範囲で扱えるように揃えられています

## 入力画像の前提

- OpenCV 形式の `numpy.ndarray`
- shape は `(height, width, 3)`
- dtype は `np.uint8`
- 色順は `RGB` ではなく `BGR`
