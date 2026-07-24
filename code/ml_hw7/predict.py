import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

from data import (
    QATensorDataset,
    build_eval_features,
    postprocess_predictions,
    read_qa_data,
)


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CODE_DIR.parent / "data" / "hw7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate HW7 Kaggle result.csv.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--model-dir", default=str(CODE_DIR / "outputs" / "best_model"))
    parser.add_argument("--output", default=str(CODE_DIR / "result.csv"))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=40)
    parser.add_argument("--n-best-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-fp16", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(
            f"找不到训练后的模型：{model_dir}\n请先运行 train.py。"
        )

    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(model_dir).to(device)
    test_questions, test_paragraphs = read_qa_data(
        Path(args.data_dir) / "hw7_test.json"
    )
    print(f"Test questions: {len(test_questions)} | Device: {device}")
    print("Building test windows...")
    test_features = build_eval_features(
        test_questions,
        test_paragraphs,
        tokenizer,
        args.max_length,
        args.doc_stride,
    )
    loader = DataLoader(
        QATensorDataset(test_features, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    start_logits: list[np.ndarray] = []
    end_logits: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Predicting"):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model(**batch)
            start_logits.extend(output.start_logits.float().cpu().numpy())
            end_logits.extend(output.end_logits.float().cpu().numpy())

    predictions = postprocess_predictions(
        test_questions,
        test_paragraphs,
        test_features,
        start_logits,
        end_logits,
        n_best_size=args.n_best_size,
        max_answer_length=args.max_answer_length,
    )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Answer"])
        for question, answer in zip(test_questions, predictions):
            writer.writerow([question["id"], answer.replace(",", "")])

    empty_answers = sum(not answer for answer in predictions)
    print(f"Saved {len(predictions)} predictions to {output_path}")
    print(f"Empty answers: {empty_answers}")
    if len(predictions) != 4957:
        print("Warning: the original HW7 test set should contain 4,957 questions.")


if __name__ == "__main__":
    main()

