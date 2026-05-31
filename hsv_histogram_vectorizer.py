from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import cv2
import numpy as np

__version__ = "0.9.0"


NormalizeMode = Literal["l1", "l2", "none"]
CenterCropMode = Literal["rect", "ellipse", "none"]
HSVCompareMethod = Literal["intersection", "cosine", "correlation", "bhattacharyya", "chisqr"]


@dataclass(frozen=True)
class HSVHistogramVectorizerConfig:
    """
    HSVヒストグラム特徴量の設定。

    前提:
        入力画像は、すでにOBBなどで水平矩形化されたcrop画像。
        OpenCV形式のBGR画像を入力する。

    h_bins, s_bins, v_bins:
        HSV各チャンネルのbin数。
        OpenCVのHSV範囲は H: 0-179, S: 0-255, V: 0-255。

    center_scale:
        crop画像の中央何割を使うか。
        1.0なら全体。
        0.85なら中央85%のみ。

    center_crop_mode:
        "rect"    : 中央矩形を使用
        "ellipse" : 中央楕円を使用
        "none"    : 中央cropなし。全体を使用

    normalize:
        "l1"   : 合計値が1になるよう正規化。intersection向き。
        "l2"   : L2ノルムが1になるよう正規化。cosine向き。
        "none" : 正規化しない。

    compare_method:
        "intersection"  : ヒストグラム交差。推奨初期値。高いほど類似。
        "cosine"        : コサイン類似度。高いほど類似。
        "correlation"   : OpenCVの相関。高いほど類似。
        "bhattacharyya" : OpenCVの距離を 1 / (1 + distance) に変換。高いほど類似。
        "chisqr"        : OpenCVの距離を 1 / (1 + distance) に変換。高いほど類似。

    output_dtype:
        出力ベクトルのdtype。
    """

    h_bins: int = 16
    s_bins: int = 8
    v_bins: int = 8
    center_scale: float = 0.85
    center_crop_mode: CenterCropMode = "rect"
    normalize: NormalizeMode = "l1"
    compare_method: HSVCompareMethod = "intersection"
    output_dtype: type = np.float32


class HSVHistogramVectorizer:
    """
    水平矩形化済みcrop画像からHSVヒストグラムベクトルを作成し、
    HSVヒストグラム同士の類似度を計算するクラス。

    使い方:
        vectorizer = HSVHistogramVectorizer()

        vec1 = vectorizer.extract(crop_bgr_1)
        vec2 = vectorizer.extract(crop_bgr_2)

        score = vectorizer.compare(vec1, vec2)

    入力:
        crop_bgr:
            OpenCV形式のBGR画像。
            shape = (height, width, 3)
            dtype = np.uint8

    出力:
        numpy.ndarray
            shape = (h_bins * s_bins * v_bins,)
            dtype = np.float32 by default

    compare() の戻り値:
        float
            原則として 1.0 に近いほど類似、0.0 に近いほど非類似。
            correlation は理論上マイナスになることがあるため、0.0〜1.0へ簡易変換する。
    """

    def __init__(self, config: HSVHistogramVectorizerConfig | None = None) -> None:
        self.config = config or HSVHistogramVectorizerConfig()
        self._validate_config()

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray:
        """
        水平化済みcrop画像からHSVヒストグラムベクトルを返す。
        """
        self._validate_image(crop_bgr)

        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        mask = self._create_mask(crop_bgr.shape[:2])

        hist = cv2.calcHist(
            images=[hsv],
            channels=[0, 1, 2],
            mask=mask,
            histSize=[
                self.config.h_bins,
                self.config.s_bins,
                self.config.v_bins,
            ],
            ranges=[
                0, 180,
                0, 256,
                0, 256,
            ],
        )

        vector = hist.flatten().astype(self.config.output_dtype, copy=False)
        vector = self._normalize(vector)

        return vector.astype(self.config.output_dtype, copy=False)

    def compare(self, a: np.ndarray, b: np.ndarray) -> float:
        """
        HSVヒストグラムベクトル同士の類似度を返す。

        Returns:
            float:
                原則として 1.0 に近いほど類似。
        """
        self._validate_vector_pair(a, b)

        method = self.config.compare_method

        if method == "intersection":
            return self._compare_intersection(a, b)

        if method == "cosine":
            return self._compare_cosine(a, b)

        if method == "correlation":
            return self._compare_correlation(a, b)

        if method == "bhattacharyya":
            return self._compare_bhattacharyya(a, b)

        if method == "chisqr":
            return self._compare_chisqr(a, b)

        raise ValueError(f"Unsupported compare_method: {method}")

    def expected_dim(self) -> int:
        """
        出力ベクトル次元を返す。
        """
        return self.config.h_bins * self.config.s_bins * self.config.v_bins

    def _create_mask(self, image_hw: tuple[int, int]) -> np.ndarray | None:
        """
        中央領域だけを使うためのmaskを作る。

        OpenCVのcalcHistでは、maskはuint8で、
        0以外の領域がヒストグラム対象になる。
        """
        height, width = image_hw

        if self.config.center_crop_mode == "none":
            return None

        scale = self.config.center_scale

        if scale >= 1.0:
            return None

        mask = np.zeros((height, width), dtype=np.uint8)

        crop_w = max(1, int(round(width * scale)))
        crop_h = max(1, int(round(height * scale)))

        x1 = max(0, (width - crop_w) // 2)
        y1 = max(0, (height - crop_h) // 2)
        x2 = min(width, x1 + crop_w)
        y2 = min(height, y1 + crop_h)

        if self.config.center_crop_mode == "rect":
            mask[y1:y2, x1:x2] = 255
            return mask

        if self.config.center_crop_mode == "ellipse":
            center = (width // 2, height // 2)
            axes = (max(1, crop_w // 2), max(1, crop_h // 2))
            cv2.ellipse(
                img=mask,
                center=center,
                axes=axes,
                angle=0,
                startAngle=0,
                endAngle=360,
                color=255,
                thickness=-1,
            )
            return mask

        raise ValueError(f"Unsupported center_crop_mode: {self.config.center_crop_mode}")

    def _normalize(self, vector: np.ndarray) -> np.ndarray:
        mode = self.config.normalize

        if mode == "none":
            return vector

        if mode == "l1":
            total = float(vector.sum())
            if total <= 0.0:
                return vector
            return vector / total

        if mode == "l2":
            norm = float(np.linalg.norm(vector))
            if norm <= 0.0:
                return vector
            return vector / norm

        raise ValueError(f"Unsupported normalize mode: {mode}")

    @staticmethod
    def _compare_intersection(a: np.ndarray, b: np.ndarray) -> float:
        """
        L1正規化済みヒストグラム向け。

        L1正規化済みなら、
            完全一致: 1.0
            完全不一致に近い: 0.0

        未正規化でも動くが、スケール差の影響を受ける。
        """
        score = float(np.minimum(a, b).sum())
        return HSVHistogramVectorizer._clamp_score(score)

    @staticmethod
    def _compare_cosine(a: np.ndarray, b: np.ndarray) -> float:
        """
        コサイン類似度。

        L2正規化済みなら内積と同等。
        HSVヒストグラムは非負なので、通常は 0.0〜1.0 に収まる。
        """
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= 0.0:
            return 0.0

        score = float(np.dot(a, b) / denom)
        return HSVHistogramVectorizer._clamp_score(score)

    @staticmethod
    def _compare_correlation(a: np.ndarray, b: np.ndarray) -> float:
        """
        OpenCVのHISTCMP_CORREL。

        戻り値は通常、
            1.0 に近いほど類似
            0.0 付近は弱い相関
            -1.0 に近いほど逆相関

        Matcherで扱いやすいように、[-1, 1] を [0, 1] へ変換する。
        """
        raw = float(
            cv2.compareHist(
                a.astype(np.float32, copy=False),
                b.astype(np.float32, copy=False),
                cv2.HISTCMP_CORREL,
            )
        )
        score = (raw + 1.0) * 0.5
        return HSVHistogramVectorizer._clamp_score(score)

    @staticmethod
    def _compare_bhattacharyya(a: np.ndarray, b: np.ndarray) -> float:
        """
        OpenCVのHISTCMP_BHATTACHARYYA。

        compareHistの戻り値は距離なので、小さいほど類似。
        ここでは Matcher で扱いやすいように類似度へ変換する。
        """
        distance = float(
            cv2.compareHist(
                a.astype(np.float32, copy=False),
                b.astype(np.float32, copy=False),
                cv2.HISTCMP_BHATTACHARYYA,
            )
        )
        return HSVHistogramVectorizer._distance_to_similarity(distance)

    @staticmethod
    def _compare_chisqr(a: np.ndarray, b: np.ndarray) -> float:
        """
        OpenCVのHISTCMP_CHISQR。

        compareHistの戻り値は距離なので、小さいほど類似。
        ここでは Matcher で扱いやすいように類似度へ変換する。
        """
        distance = float(
            cv2.compareHist(
                a.astype(np.float32, copy=False),
                b.astype(np.float32, copy=False),
                cv2.HISTCMP_CHISQR,
            )
        )
        return HSVHistogramVectorizer._distance_to_similarity(distance)

    @staticmethod
    def _distance_to_similarity(distance: float) -> float:
        """
        距離を 0.0〜1.0 付近の類似度に変換する。

        distance = 0.0 なら similarity = 1.0。
        distance が大きいほど 0.0 に近づく。
        """
        if not np.isfinite(distance):
            return 0.0

        if distance < 0.0:
            distance = 0.0

        return float(1.0 / (1.0 + distance))

    @staticmethod
    def _clamp_score(score: float) -> float:
        if not np.isfinite(score):
            return 0.0

        if score < 0.0:
            return 0.0

        if score > 1.0:
            return 1.0

        return float(score)

    def _validate_config(self) -> None:
        if self.config.h_bins <= 0:
            raise ValueError("h_bins must be positive.")

        if self.config.s_bins <= 0:
            raise ValueError("s_bins must be positive.")

        if self.config.v_bins <= 0:
            raise ValueError("v_bins must be positive.")

        if not (0.0 < self.config.center_scale <= 1.0):
            raise ValueError("center_scale must be in the range (0.0, 1.0].")

        if self.config.center_crop_mode not in ("rect", "ellipse", "none"):
            raise ValueError(
                "center_crop_mode must be one of: 'rect', 'ellipse', 'none'."
            )

        if self.config.normalize not in ("l1", "l2", "none"):
            raise ValueError("normalize must be one of: 'l1', 'l2', 'none'.")

        if self.config.compare_method not in (
            "intersection",
            "cosine",
            "correlation",
            "bhattacharyya",
            "chisqr",
        ):
            raise ValueError(
                "compare_method must be one of: "
                "'intersection', 'cosine', 'correlation', 'bhattacharyya', 'chisqr'."
            )

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("crop_bgr must be a numpy.ndarray.")

        if image.ndim != 3:
            raise ValueError("crop_bgr must have shape (height, width, channels).")

        if image.shape[2] != 3:
            raise ValueError("crop_bgr must have 3 channels.")

        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError("crop_bgr must have positive width and height.")

        if image.dtype != np.uint8:
            raise ValueError("crop_bgr dtype must be np.uint8.")

    def _validate_vector_pair(self, a: np.ndarray, b: np.ndarray) -> None:
        if not isinstance(a, np.ndarray) or not isinstance(b, np.ndarray):
            raise TypeError("Both vectors must be numpy.ndarray.")

        if a.ndim != 1 or b.ndim != 1:
            raise ValueError("Both vectors must be 1-dimensional.")

        if a.shape != b.shape:
            raise ValueError(f"Vector shape mismatch: {a.shape} != {b.shape}")

        expected = self.expected_dim()
        if a.shape[0] != expected:
            raise ValueError(
                f"Vector dimension mismatch: got {a.shape[0]}, expected {expected}."
            )


if __name__ == "__main__":
    from compare_samples import main

    raise SystemExit(main())
