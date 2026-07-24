import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


def read_qa_data(path: str | Path) -> tuple[list[dict[str, Any]], list[str]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"找不到数据文件：{path}")
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data["questions"], data["paragraphs"]


class QATensorDataset(Dataset):
    """Only returns tensors that can be passed directly to the QA model."""

    def __init__(self, features: list[dict[str, Any]], training: bool) -> None:
        self.features = features
        self.training = training

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        feature = self.features[index]
        item = {
            "input_ids": torch.tensor(feature["input_ids"], dtype=torch.long),
            "attention_mask": torch.tensor(feature["attention_mask"], dtype=torch.long),
        }
        if "token_type_ids" in feature:
            item["token_type_ids"] = torch.tensor(
                feature["token_type_ids"], dtype=torch.long
            )
        if self.training:
            item["start_positions"] = torch.tensor(
                feature["start_positions"], dtype=torch.long
            )
            item["end_positions"] = torch.tensor(
                feature["end_positions"], dtype=torch.long
            )
        return item


def build_train_features(
    questions: list[dict[str, Any]],
    paragraphs: list[str],
    tokenizer,
    max_length: int,
    doc_stride: int,
) -> list[dict[str, Any]]:
    """
    Tokenize training examples with overlapping windows.

    Only windows containing the gold answer are retained because every HW7
    question is answerable.
    """
    features: list[dict[str, Any]] = []

    for question in questions:
        context = paragraphs[question["paragraph_id"]]
        answer_start = question["answer_start"]
        answer_end_exclusive = question["answer_end"] + 1

        encoded = tokenizer(
            question["question_text"],
            context,
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        for window_index in range(len(encoded["input_ids"])):
            sequence_ids = encoded.sequence_ids(window_index)
            offsets = encoded["offset_mapping"][window_index]
            context_indices = [
                index for index, sequence_id in enumerate(sequence_ids)
                if sequence_id == 1
            ]
            if not context_indices:
                continue

            context_start = context_indices[0]
            context_end = context_indices[-1]
            window_char_start = offsets[context_start][0]
            window_char_end = offsets[context_end][1]
            if (
                answer_start < window_char_start
                or answer_end_exclusive > window_char_end
            ):
                continue

            start_position = context_start
            while (
                start_position <= context_end
                and offsets[start_position][0] <= answer_start
            ):
                start_position += 1
            start_position -= 1

            end_position = context_end
            while (
                end_position >= context_start
                and offsets[end_position][1] >= answer_end_exclusive
            ):
                end_position -= 1
            end_position += 1

            feature = {
                key: encoded[key][window_index]
                for key in ("input_ids", "attention_mask", "token_type_ids")
                if key in encoded
            }
            feature["start_positions"] = start_position
            feature["end_positions"] = end_position
            features.append(feature)

    return features


def build_eval_features(
    questions: list[dict[str, Any]],
    paragraphs: list[str],
    tokenizer,
    max_length: int,
    doc_stride: int,
) -> list[dict[str, Any]]:
    """Tokenize dev/test examples and retain metadata for answer extraction."""
    features: list[dict[str, Any]] = []

    for example_index, question in enumerate(questions):
        context = paragraphs[question["paragraph_id"]]
        encoded = tokenizer(
            question["question_text"],
            context,
            truncation="only_second",
            max_length=max_length,
            stride=doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length",
        )

        for window_index in range(len(encoded["input_ids"])):
            sequence_ids = encoded.sequence_ids(window_index)
            offsets = encoded["offset_mapping"][window_index]
            # Question and special-token offsets must never become answers.
            context_offsets = [
                tuple(offset) if sequence_ids[index] == 1 else None
                for index, offset in enumerate(offsets)
            ]
            feature = {
                key: encoded[key][window_index]
                for key in ("input_ids", "attention_mask", "token_type_ids")
                if key in encoded
            }
            feature["example_index"] = example_index
            feature["offset_mapping"] = context_offsets
            features.append(feature)

    return features


def postprocess_predictions(
    questions: list[dict[str, Any]],
    paragraphs: list[str],
    features: list[dict[str, Any]],
    start_logits: list,
    end_logits: list,
    n_best_size: int = 20,
    max_answer_length: int = 40,
) -> list[str]:
    """Choose the highest-scoring valid answer span across all windows."""
    features_by_example: dict[int, list[int]] = defaultdict(list)
    for feature_index, feature in enumerate(features):
        features_by_example[feature["example_index"]].append(feature_index)

    predictions: list[str] = []
    for example_index, question in enumerate(questions):
        context = paragraphs[question["paragraph_id"]]
        best_answer = ""
        best_score = float("-inf")

        for feature_index in features_by_example[example_index]:
            offsets = features[feature_index]["offset_mapping"]
            starts = start_logits[feature_index]
            ends = end_logits[feature_index]
            start_candidates = starts.argsort()[-n_best_size:][::-1]
            end_candidates = ends.argsort()[-n_best_size:][::-1]

            for start_index in start_candidates:
                for end_index in end_candidates:
                    if (
                        start_index >= len(offsets)
                        or end_index >= len(offsets)
                        or offsets[start_index] is None
                        or offsets[end_index] is None
                        or end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    char_start = offsets[start_index][0]
                    char_end = offsets[end_index][1]
                    if char_end <= char_start:
                        continue

                    score = float(starts[start_index] + ends[end_index])
                    if score > best_score:
                        best_score = score
                        best_answer = context[char_start:char_end]

        predictions.append(best_answer.strip())

    return predictions


def exact_match_accuracy(
    questions: list[dict[str, Any]],
    predictions: list[str],
) -> float:
    if not questions:
        return 0.0
    correct = sum(
        prediction == question["answer_text"]
        for question, prediction in zip(questions, predictions)
    )
    return correct / len(questions)

