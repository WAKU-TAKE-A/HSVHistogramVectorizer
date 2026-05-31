from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from hsv_histogram_vectorizer import (
    HSVCompareMethod,
    HSVHistogramVectorizer,
    HSVHistogramVectorizerConfig,
)


DEFAULT_SAMPLE_DIR = Path(__file__).resolve().parent / "sample"
COMPARE_METHODS: tuple[HSVCompareMethod, ...] = (
    "intersection",
    "cosine",
    "correlation",
    "bhattacharyya",
    "chisqr",
)


def _read_image(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Image not found or unreadable: {path}")
    return image


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare sample images against sample/ref.png using HSV histograms."
    )
    parser.add_argument(
        "--sample-dir",
        type=Path,
        default=DEFAULT_SAMPLE_DIR,
        help="Directory containing ref.png and sample*.png.",
    )
    parser.add_argument(
        "--ref",
        type=Path,
        default=None,
        help="Reference image path. Defaults to <sample-dir>/ref.png.",
    )
    parser.add_argument(
        "--method",
        choices=COMPARE_METHODS,
        default="intersection",
        help="Histogram comparison method.",
    )
    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Show scores for all comparison methods side by side.",
    )
    parser.add_argument("--h-bins", type=int, default=16)
    parser.add_argument("--s-bins", type=int, default=8)
    parser.add_argument("--v-bins", type=int, default=8)
    parser.add_argument("--center-scale", type=float, default=0.85)
    parser.add_argument(
        "--center-crop-mode",
        choices=("rect", "ellipse", "none"),
        default="rect",
    )
    parser.add_argument(
        "--normalize",
        choices=("l1", "l2", "none"),
        default="l1",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    sample_dir = args.sample_dir
    ref_path = args.ref if args.ref is not None else sample_dir / "ref.png"
    sample_paths = sorted(sample_dir.glob("sample*.png"))

    if not sample_paths:
        raise FileNotFoundError(f"No sample*.png files found in: {sample_dir}")

    methods = COMPARE_METHODS if args.all_methods else (args.method,)
    vectorizers = {
        method: HSVHistogramVectorizer(
            _create_config(args=args, compare_method=method)
        )
        for method in methods
    }

    ref_image = _read_image(ref_path)
    ref_vectors = {
        method: vectorizer.extract(ref_image)
        for method, vectorizer in vectorizers.items()
    }

    print(f"reference: {ref_path}")
    print(f"vector dim: {next(iter(vectorizers.values())).expected_dim()}")
    print(f"compare methods: {', '.join(methods)}")
    print()

    if args.all_methods:
        _print_all_methods_table(
            sample_paths=sample_paths,
            ref_vectors=ref_vectors,
            vectorizers=vectorizers,
        )
    else:
        _print_single_method_table(
            sample_paths=sample_paths,
            ref_vector=ref_vectors[args.method],
            vectorizer=vectorizers[args.method],
        )

    return 0


def _create_config(
    args: argparse.Namespace,
    compare_method: HSVCompareMethod,
) -> HSVHistogramVectorizerConfig:
    return HSVHistogramVectorizerConfig(
        h_bins=args.h_bins,
        s_bins=args.s_bins,
        v_bins=args.v_bins,
        center_scale=args.center_scale,
        center_crop_mode=args.center_crop_mode,
        normalize=args.normalize,
        compare_method=compare_method,
    )


def _print_single_method_table(
    sample_paths: Sequence[Path],
    ref_vector: np.ndarray,
    vectorizer: HSVHistogramVectorizer,
) -> None:
    print(f"{'sample':<16} {'score':>10} {'size':>12}")
    print("-" * 40)

    for sample_path in sample_paths:
        sample_image = _read_image(sample_path)
        sample_vector = vectorizer.extract(sample_image)
        score = vectorizer.compare(ref_vector, sample_vector)
        height, width = sample_image.shape[:2]
        size = f"{width}x{height}"
        print(f"{sample_path.name:<16} {score:>10.6f} {size:>12}")


def _print_all_methods_table(
    sample_paths: Sequence[Path],
    ref_vectors: dict[HSVCompareMethod, np.ndarray],
    vectorizers: dict[HSVCompareMethod, HSVHistogramVectorizer],
) -> None:
    method_headers = {
        "intersection": "intersect",
        "cosine": "cosine",
        "correlation": "correl",
        "bhattacharyya": "bhatta",
        "chisqr": "chisqr",
    }

    header = f"{'sample':<16} {'size':>10}"
    for method in vectorizers:
        header += f" {method_headers[method]:>10}"

    print(header)
    print("-" * len(header))

    for sample_path in sample_paths:
        sample_image = _read_image(sample_path)
        height, width = sample_image.shape[:2]
        row = f"{sample_path.name:<16} {f'{width}x{height}':>10}"

        for method, vectorizer in vectorizers.items():
            sample_vector = vectorizer.extract(sample_image)
            score = vectorizer.compare(ref_vectors[method], sample_vector)
            row += f" {score:>10.6f}"

        print(row)


if __name__ == "__main__":
    raise SystemExit(main())
