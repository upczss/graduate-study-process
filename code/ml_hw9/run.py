from __future__ import annotations

import argparse
from pathlib import Path

from bert_explain import run_bert_explanations
from cnn_explain import run_cnn_explanations
from common import (
    DEFAULT_HW3_DATA,
    DEFAULT_HW3_MODEL,
    DEFAULT_HW7_MODEL,
    DEFAULT_OUTPUT,
    list_food_images,
    load_food_model,
    select_device,
    set_seed,
)


CNN_METHODS = ("lime", "saliency", "smoothgrad", "filter", "integrated-gradients")
BERT_METHODS = ("attention", "embedding", "similarity")


def parse_indices(text: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("indices must be comma-separated integers") from exc
    if not values or any(value < 0 for value in values):
        raise argparse.ArgumentTypeError("indices must contain non-negative integers")
    return values


def parse_layers(text: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in text.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("layers must be comma-separated integers") from exc
    if not values:
        raise argparse.ArgumentTypeError("at least one layer is required")
    return values


def add_shared(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=42)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="HW9 Explainable AI experiments")
    subparsers = parser.add_subparsers(dest="topic", required=True)

    cnn = subparsers.add_parser("cnn", help="Explain the HW3 Food-11 CNN")
    add_shared(cnn)
    cnn.add_argument("--method", choices=("all",) + CNN_METHODS, default="all")
    cnn.add_argument("--data-dir", type=Path, default=DEFAULT_HW3_DATA)
    cnn.add_argument("--split", choices=("training", "validation", "test"), default="validation")
    cnn.add_argument("--model-path", type=Path, default=DEFAULT_HW3_MODEL)
    cnn.add_argument("--image-indices", type=parse_indices, default=parse_indices("0,1,2,3,4,5,6,7,8,9"))
    cnn.add_argument("--layer", default="cnn_layer6.block.0")
    cnn.add_argument("--filter-index", type=int, default=0)
    cnn.add_argument("--smooth-samples", type=int, default=30)
    cnn.add_argument("--ig-steps", type=int, default=50)
    cnn.add_argument("--lime-samples", type=int, default=500)

    bert = subparsers.add_parser("bert", help="Explain the HW7 BERT QA model")
    add_shared(bert)
    bert.add_argument("--method", choices=("all",) + BERT_METHODS, default="all")
    bert.add_argument("--model-dir", type=Path, default=DEFAULT_HW7_MODEL)
    bert.add_argument("--data-path", type=Path, default=DEFAULT_HW3_DATA.parent / "hw7" / "hw7_dev.json")
    bert.add_argument("--example-index", type=int, default=0)
    bert.add_argument("--attention-layer", type=int, default=-1)
    bert.add_argument("--attention-head", type=int, default=0)
    bert.add_argument("--pca-layers", type=parse_layers, default=parse_layers("0,6,12"))
    bert.add_argument("--token-a")
    bert.add_argument("--token-b")
    bert.add_argument("--max-length", type=int, default=384)
    bert.add_argument("--plot-tokens", type=int, default=60)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    set_seed(args.seed)
    device = select_device(args.device)
    print(f"device: {device}")

    if args.topic == "cnn":
        model, image_size = load_food_model(args.model_path.resolve(), device)
        files = list_food_images((args.data_dir / args.split).resolve())
        if max(args.image_indices) >= len(files):
            raise IndexError(f"Image index exceeds the {len(files)} available files")
        selected = [files[index] for index in args.image_indices]
        methods = list(CNN_METHODS) if args.method == "all" else [args.method]
        run_cnn_explanations(
            model,
            selected,
            image_size,
            device,
            (args.output_dir / "cnn").resolve(),
            methods,
            args.layer,
            args.filter_index,
            args.smooth_samples,
            args.ig_steps,
            args.lime_samples,
        )
        print(f"CNN results saved to {(args.output_dir / 'cnn').resolve()}")
    else:
        methods = list(BERT_METHODS) if args.method == "all" else [args.method]
        run_bert_explanations(
            args.model_dir.resolve(),
            args.data_path.resolve(),
            args.example_index,
            device,
            (args.output_dir / "bert").resolve(),
            methods,
            args.attention_layer,
            args.attention_head,
            args.pca_layers,
            args.token_a,
            args.token_b,
            args.max_length,
            args.plot_tokens,
        )
        print(f"BERT results saved to {(args.output_dir / 'bert').resolve()}")


if __name__ == "__main__":
    main()
