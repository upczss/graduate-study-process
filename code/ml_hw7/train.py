import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import (
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

from data import (
    QATensorDataset,
    build_eval_features,
    build_train_features,
    exact_match_accuracy,
    postprocess_predictions,
    read_qa_data,
)


CODE_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = CODE_DIR.parent / "data" / "hw7"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Chinese QA model for HW7.")
    parser.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    parser.add_argument("--output-dir", default=str(CODE_DIR / "outputs"))
    parser.add_argument("--model-name", default="bert-base-chinese")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--doc-stride", type=int, default=128)
    parser.add_argument("--max-answer-length", type=int, default=40)
    parser.add_argument("--n-best-size", type=int, default=20)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument(
        "--limit-train",
        type=int,
        default=None,
        help="Only use the first N train questions for a quick test.",
    )
    parser.add_argument(
        "--limit-dev",
        type=int,
        default=None,
        help="Only use the first N dev questions for a quick test.",
    )
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collect_logits(model, loader, device, use_fp16):
    all_start_logits: list[np.ndarray] = []
    all_end_logits: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for batch in tqdm(loader, desc="Validating", leave=False):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model(**batch)
            all_start_logits.extend(output.start_logits.float().cpu().numpy())
            all_end_logits.extend(output.end_logits.float().cpu().numpy())
    return all_start_logits, all_end_logits


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch-size must be positive.")
    if args.doc_stride >= args.max_length:
        raise ValueError("--doc-stride must be smaller than --max-length.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_fp16 = device.type == "cuda" and not args.no_fp16
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    best_model_dir = output_dir / "best_model"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_questions, train_paragraphs = read_qa_data(data_dir / "hw7_train.json")
    dev_questions, dev_paragraphs = read_qa_data(data_dir / "hw7_dev.json")
    if args.limit_train:
        train_questions = train_questions[: args.limit_train]
    if args.limit_dev:
        dev_questions = dev_questions[: args.limit_dev]

    print(f"Device: {device} | FP16: {use_fp16}")
    print(f"Model: {args.model_name}")
    print(f"Train questions: {len(train_questions)} | Dev questions: {len(dev_questions)}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)
    if not tokenizer.is_fast:
        raise ValueError("HW7 requires a fast tokenizer to map tokens back to text.")
    model = AutoModelForQuestionAnswering.from_pretrained(args.model_name).to(device)

    print("Building sliding-window features...")
    train_features = build_train_features(
        train_questions,
        train_paragraphs,
        tokenizer,
        args.max_length,
        args.doc_stride,
    )
    dev_features = build_eval_features(
        dev_questions,
        dev_paragraphs,
        tokenizer,
        args.max_length,
        args.doc_stride,
    )
    print(f"Train features: {len(train_features)} | Dev windows: {len(dev_features)}")

    train_loader = DataLoader(
        QATensorDataset(train_features, training=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    dev_loader = DataLoader(
        QATensorDataset(dev_features, training=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    no_decay = ("bias", "LayerNorm.weight")
    optimizer_groups = [
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if not any(item in name for item in no_decay)
            ],
            "weight_decay": args.weight_decay,
        },
        {
            "params": [
                parameter
                for name, parameter in model.named_parameters()
                if any(item in name for item in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]
    optimizer = torch.optim.AdamW(optimizer_groups, lr=args.learning_rate)
    updates_per_epoch = (
        len(train_loader) + args.gradient_accumulation - 1
    ) // args.gradient_accumulation
    total_updates = updates_per_epoch * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(total_updates * args.warmup_ratio),
        num_training_steps=total_updates,
    )
    scaler = torch.cuda.amp.GradScaler(enabled=use_fp16)

    best_accuracy = -1.0
    global_update = 0
    optimizer.zero_grad(set_to_none=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")

        for step, batch in enumerate(progress, start=1):
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.cuda.amp.autocast(enabled=use_fp16):
                output = model(**batch)
                loss = output.loss / args.gradient_accumulation

            scaler.scale(loss).backward()
            running_loss += loss.item() * args.gradient_accumulation
            should_update = (
                step % args.gradient_accumulation == 0
                or step == len(train_loader)
            )
            if should_update:
                scaler.unscale_(optimizer)
                clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_update += 1

            progress.set_postfix(
                loss=f"{running_loss / step:.4f}",
                lr=f"{scheduler.get_last_lr()[0]:.2e}",
            )

        start_logits, end_logits = collect_logits(
            model, dev_loader, device, use_fp16
        )
        predictions = postprocess_predictions(
            dev_questions,
            dev_paragraphs,
            dev_features,
            start_logits,
            end_logits,
            n_best_size=args.n_best_size,
            max_answer_length=args.max_answer_length,
        )
        accuracy = exact_match_accuracy(dev_questions, predictions)
        print(f"Epoch {epoch}: dev exact match = {accuracy:.5f}")

        epoch_dir = output_dir / f"epoch_{epoch}"
        model.save_pretrained(epoch_dir)
        tokenizer.save_pretrained(epoch_dir)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            model.save_pretrained(best_model_dir)
            tokenizer.save_pretrained(best_model_dir)
            (best_model_dir / "training_args.json").write_text(
                json.dumps(vars(args), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            (best_model_dir / "dev_metrics.json").write_text(
                json.dumps(
                    {"exact_match": accuracy, "epoch": epoch},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            print(f"Saved new best model to {best_model_dir}")

    print(f"Training complete. Best dev exact match: {best_accuracy:.5f}")


if __name__ == "__main__":
    main()

