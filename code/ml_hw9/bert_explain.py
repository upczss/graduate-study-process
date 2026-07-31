from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(__file__).resolve().parent / "outputs" / ".matplotlib")
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForQuestionAnswering, AutoTokenizer


def configure_cjk_font() -> None:
    for path in (
        Path("C:/Windows/Fonts/msyh.ttc"),
        Path("C:/Windows/Fonts/msyhbd.ttc"),
        Path("C:/Windows/Fonts/simhei.ttf"),
    ):
        if path.is_file():
            font_manager.fontManager.addfont(path)
            family = font_manager.FontProperties(fname=path).get_name()
            plt.rcParams["font.sans-serif"] = [family, "DejaVu Sans"]
            plt.rcParams["axes.unicode_minus"] = False
            return


configure_cjk_font()


def read_example(data_path: Path, index: int) -> tuple[str, str, str]:
    with data_path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    questions = data["questions"]
    paragraphs = data["paragraphs"]
    if not 0 <= index < len(questions):
        raise IndexError(f"Example index must be between 0 and {len(questions) - 1}")
    item = questions[index]
    return item["question_text"], paragraphs[item["paragraph_id"]], str(item.get("answer_text", ""))


def load_qa_model(model_dir: Path, device: torch.device):
    if not model_dir.is_dir():
        raise FileNotFoundError(f"HW7 model directory not found: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForQuestionAnswering.from_pretrained(
        model_dir, attn_implementation="eager"
    ).to(device).eval()
    return tokenizer, model


def prepare_inputs(tokenizer, question: str, context: str, device: torch.device, max_length: int):
    encoded = tokenizer(
        question,
        context,
        truncation="only_second",
        max_length=max_length,
        return_tensors="pt",
    )
    sequence_ids = encoded.sequence_ids(0)
    inputs = {key: value.to(device) for key, value in encoded.items()}
    tokens = tokenizer.convert_ids_to_tokens(encoded["input_ids"][0])
    return inputs, tokens, sequence_ids


def predict_answer(tokenizer, outputs, input_ids: torch.Tensor, sequence_ids: list[int | None]) -> str:
    context_mask = torch.tensor(
        [value == 1 for value in sequence_ids], device=input_ids.device, dtype=torch.bool
    )
    start = outputs.start_logits[0].masked_fill(~context_mask, -1e9).argmax().item()
    end_logits = outputs.end_logits[0].clone()
    end_logits[:start] = -1e9
    end_logits = end_logits.masked_fill(~context_mask, -1e9)
    end = end_logits.argmax().item()
    return tokenizer.decode(input_ids[0, start : end + 1], skip_special_tokens=True)


def save_attention(
    attention: torch.Tensor,
    tokens: list[str],
    layer: int,
    head: int,
    output_path: Path,
    max_tokens: int,
) -> None:
    matrix = attention[layer][0, head].detach().cpu().numpy()
    length = min(len(tokens), max_tokens)
    matrix = matrix[:length, :length]
    labels = tokens[:length]
    figure, axis = plt.subplots(figsize=(max(8, length * 0.32), max(7, length * 0.32)))
    image = axis.imshow(matrix, cmap="Blues", aspect="auto")
    axis.set_xticks(range(length), labels, rotation=90, fontsize=7)
    axis.set_yticks(range(length), labels, fontsize=7)
    axis.set_title(f"BERT attention | layer={layer}, head={head}")
    figure.colorbar(image, ax=axis, fraction=0.025)
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_embedding_pca(
    hidden_states: tuple[torch.Tensor, ...],
    tokens: list[str],
    layers: list[int],
    output_path: Path,
    max_tokens: int,
) -> None:
    length = min(len(tokens), max_tokens)
    figure, axes = plt.subplots(1, len(layers), figsize=(7 * len(layers), 6), squeeze=False)
    for column, layer in enumerate(layers):
        values = hidden_states[layer][0, :length].detach().cpu().numpy()
        centered = values - values.mean(axis=0, keepdims=True)
        _, _, right_vectors = np.linalg.svd(centered, full_matrices=False)
        points = centered @ right_vectors[:2].T
        axis = axes[0, column]
        axis.scatter(points[:, 0], points[:, 1], s=24)
        for token, (x, y) in zip(tokens[:length], points):
            axis.annotate(token, (x, y), fontsize=7, alpha=0.9)
        axis.set_title(f"Hidden-state PCA | layer={layer}")
        axis.set_xlabel("PC1")
        axis.set_ylabel("PC2")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def token_similarity(
    hidden: torch.Tensor, tokens: list[str], token_a: str, token_b: str
) -> dict[str, float | int | str]:
    def locate(token: str) -> int:
        if token in tokens:
            return tokens.index(token)
        matches = [index for index, value in enumerate(tokens) if token in value]
        if not matches:
            raise ValueError(f"Token '{token}' was not found. Available tokens: {tokens}")
        return matches[0]

    index_a, index_b = locate(token_a), locate(token_b)
    vector_a, vector_b = hidden[0, index_a], hidden[0, index_b]
    return {
        "token_a": tokens[index_a],
        "token_b": tokens[index_b],
        "index_a": index_a,
        "index_b": index_b,
        "euclidean_distance": float(torch.linalg.vector_norm(vector_a - vector_b)),
        "cosine_similarity": float(F.cosine_similarity(vector_a[None], vector_b[None])),
    }


def run_bert_explanations(
    model_dir: Path,
    data_path: Path,
    example_index: int,
    device: torch.device,
    output_dir: Path,
    methods: list[str],
    attention_layer: int,
    attention_head: int,
    pca_layers: list[int],
    token_a: str | None,
    token_b: str | None,
    max_length: int,
    plot_tokens: int,
) -> None:
    question, context, gold_answer = read_example(data_path, example_index)
    tokenizer, model = load_qa_model(model_dir, device)
    inputs, tokens, sequence_ids = prepare_inputs(tokenizer, question, context, device, max_length)
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_attentions=True,
            output_hidden_states=True,
            return_dict=True,
        )
    predicted_answer = predict_answer(tokenizer, outputs, inputs["input_ids"], sequence_ids)
    output_dir.mkdir(parents=True, exist_ok=True)

    layer_count = len(outputs.attentions)
    head_count = outputs.attentions[0].size(1)
    if not -layer_count <= attention_layer < layer_count:
        raise ValueError(f"attention-layer must be in [{-layer_count}, {layer_count - 1}]")
    if not 0 <= attention_head < head_count:
        raise ValueError(f"attention-head must be in [0, {head_count - 1}]")

    if "attention" in methods:
        save_attention(
            outputs.attentions,
            tokens,
            attention_layer,
            attention_head,
            output_dir / "attention.png",
            plot_tokens,
        )
    if "embedding" in methods:
        hidden_count = len(outputs.hidden_states)
        normalized_layers = [layer if layer >= 0 else hidden_count + layer for layer in pca_layers]
        if any(layer < 0 or layer >= hidden_count for layer in normalized_layers):
            raise ValueError(f"PCA layers must refer to hidden states 0..{hidden_count - 1}")
        save_embedding_pca(
            outputs.hidden_states,
            tokens,
            normalized_layers,
            output_dir / "embedding_pca.png",
            plot_tokens,
        )
    if "similarity" in methods:
        if token_a is None or token_b is None:
            selectable = [token for token in tokens if token not in tokenizer.all_special_tokens]
            if len(selectable) < 2:
                raise RuntimeError("Not enough ordinary tokens for similarity analysis")
            token_a, token_b = selectable[0], selectable[1]
        result = token_similarity(outputs.hidden_states[-1], tokens, token_a, token_b)
        (output_dir / "similarity.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    summary = {
        "example_index": example_index,
        "question": question,
        "gold_answer": gold_answer,
        "predicted_answer": predicted_answer,
        "token_count": len(tokens),
        "tokens": tokens,
    }
    (output_dir / "example.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
