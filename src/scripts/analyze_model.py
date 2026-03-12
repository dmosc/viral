"""
Model analysis script: inference examples, ablation study, permutation
importance, and tabular feature importance (weight magnitude).

Outputs 4 plots to docs/assets/.
"""

import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from safetensors.torch import load_file
from datasets import load_dataset
from sklearn.metrics import f1_score, roc_auc_score, average_precision_score

from transformers import AutoTokenizer, AutoImageProcessor

from src.config import Config
from src.model.virality_predictor import ViralityPredictor
from src.model.data_processor import DataProcessor

ASSETS_DIR = Path("docs/assets")
N_SAMPLES = 300
BATCH_SIZE = 4
THRESHOLD = 0.7

TABULAR_FEATURE_NAMES = [
    "follower_count", "following_count", "heart_count", "video_count",
    "friend_count", "duration", "width", "height", "aspect_ratio", "vq_score",
    "verified", "is_private", "is_ad", "share_enabled", "stitch_enabled",
    "sin_hour", "cos_hour", "sin_day", "cos_day",
]

# Finds latest checkpoint, alternatively can hardcode
def find_latest_checkpoint():
    """Auto-detect the latest checkpoint in data/checkpoints/."""
    checkpoints_dir = Path("data/checkpoints")
    checkpoint_dirs = sorted(
        [d for d in checkpoints_dir.iterdir()
         if d.is_dir() and d.name.startswith("checkpoint-")],
        key=lambda d: int(d.name.split("-")[-1]),
    )
    if not checkpoint_dirs:
        raise FileNotFoundError(f"No checkpoints found in {checkpoints_dir}")
    latest = checkpoint_dirs[-1] / "model.safetensors"
    print(f"Using checkpoint: {latest}")
    return str(latest)


def load_model_and_data(device):
    """Load model, processor, and test dataset."""
    config = Config()

    # Load model
    model = ViralityPredictor(config)
    # can hardcode the path if you want a specific checkpoint here
    state_dict = load_file(find_latest_checkpoint())
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # Load dataset once
    print("Loading dataset...")
    dataset = load_dataset(config.dataset_id)["train"]
    splits = dataset.train_test_split(
        train_size=config.train_size, test_size=config.test_size, seed=config.seed
    )
    test_dataset = splits["test"]

    # Set up processor, inject dataset to avoid calling load_dataset again
    processor = DataProcessor.__new__(DataProcessor)
    processor.config = config
    processor.dataset = dataset
    processor.tokenizer = AutoTokenizer.from_pretrained(config.text_model_id)
    processor.processor = AutoImageProcessor.from_pretrained(config.video_model_id)
    processor.image_mean = torch.tensor(
        processor.processor.image_mean).view(1, 3, 1, 1)
    processor.tabular_means = model.tabular_means.cpu().clone()
    processor.tabular_stds = model.tabular_stds.cpu().clone()

    return model, processor, test_dataset, device


def process_examples(processor, dataset, indices, device):
    """Process dataset examples in mini-batches, returns list of processed batch dicts."""
    all_batches = []
    for start in range(0, len(indices), BATCH_SIZE):
        batch_indices = indices[start : start + BATCH_SIZE]
        rows = dataset.select(batch_indices)
        batch_dict = {col: rows[col] for col in rows.column_names}
        processed = processor._process_batch(batch_dict)
        all_batches.append(processed)
    return all_batches


def concat_batches(batches):
    """Concatenate list of batch dicts into single tensors."""
    result = {}
    for key in batches[0]:
        result[key] = torch.cat([b[key] for b in batches], dim=0)
    return result


def to_device(batch, device, exclude=("labels",)):
    """Move batch tensors to device, excluding specified keys."""
    return {k: v.to(device) for k, v in batch.items() if k not in exclude}


def _run_inference(model, batches, device):
    """Run model inference on pre-processed batches, return viral_prob array."""
    all_probs = []
    for batch in batches:
        inference_batch = to_device(batch, device)
        preds = model.predict_scores(**inference_batch)
        all_probs.extend(preds["viral_prob"].cpu().tolist())
    return np.array(all_probs)


def _run_inference_from_combined(model, combined, device):
    """Run model inference on a single combined tensor dict, in mini-batches."""
    n = combined["input_ids"].shape[0]
    all_probs = []
    for start in range(0, n, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n)
        batch = {k: v[start:end].to(device) for k, v in combined.items()
                 if k != "labels"}
        preds = model.predict_scores(**batch)
        all_probs.extend(preds["viral_prob"].cpu().tolist())
    return np.array(all_probs)


# -- Analysis 1: Inference Examples --

def analysis_inference_examples(model, processor, test_dataset, device,
                                n_examples=200):
    """Find a TP, TN, and misclassification, plot a summary table."""
    print("Running Analysis 1: Inference Examples...")
    n = min(n_examples, len(test_dataset))
    indices = list(range(n))
    batches = process_examples(processor, test_dataset, indices, device)

    all_probs = []
    all_labels = []
    descriptions = []

    for i, batch in enumerate(batches):
        start = i * BATCH_SIZE
        end = min(start + BATCH_SIZE, n)
        inference_batch = to_device(batch, device)
        preds = model.predict_scores(**inference_batch)
        all_probs.extend(preds["viral_prob"].cpu().tolist())
        all_labels.extend(batch["labels"][:, 2].tolist())
        batch_indices = indices[start:end]
        for idx in batch_indices:
            desc = test_dataset[idx].get("description", "") or ""
            descriptions.append(desc[:80])

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    preds_binary = (all_probs >= THRESHOLD).astype(int)

    # Find TP, TN, misclassification examples
    tp_idx = tn_idx = mis_idx = None
    for i in range(len(all_labels)):
        gt = int(all_labels[i])
        pred = preds_binary[i]
        if gt == 1 and pred == 1 and tp_idx is None:
            tp_idx = i
        elif gt == 0 and pred == 0 and tn_idx is None:
            tn_idx = i
        elif gt != pred and mis_idx is None:
            mis_idx = i
        if tp_idx is not None and tn_idx is not None and mis_idx is not None:
            break

    # Fallback if any not found (shouldn't happen)
    if tp_idx is None:
        tp_idx = 0
    if tn_idx is None:
        tn_idx = 1
    if mis_idx is None:
        mis_idx = 2

    rows = [
        ("True Positive", tp_idx),
        ("True Negative", tn_idx),
        ("Misclassification", mis_idx),
    ]

    fig, ax = plt.subplots(figsize=(12, 3))
    ax.axis("off")
    table_data = []
    for label, idx in rows:
        gt = "Viral" if all_labels[idx] == 1 else "Not Viral"
        prob = all_probs[idx]
        pred = "Viral" if prob >= THRESHOLD else "Not Viral"
        desc = descriptions[idx] if descriptions[idx] else "(no description)"
        table_data.append([label, desc, gt, f"{prob:.2%}", pred])

    table = ax.table(
        cellText=table_data,
        colLabels=["Category", "Description", "Ground Truth", "Viral Prob", "Prediction"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.8)

    for j in range(5):
        table[0, j].set_facecolor("#4472C4")
        table[0, j].set_text_props(color="white", fontweight="bold")

    plt.title("Inference Examples", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "inference_examples.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved inference_examples.png")


# -- Analysis 2: Ablation Study --


def analysis_ablation(model, batches, labels_all, device):
    """Run ablation study: zero each modality, measure metric drop."""
    print("Running Analysis 2: Ablation Study...")

    # Full model pass
    full_probs = _run_inference(model, batches, device)

    # Zero-text pass
    def zero_text_hook(module, input, output):
        output.last_hidden_state = torch.zeros_like(output.last_hidden_state)
        return output

    handle = model.text_model.register_forward_hook(zero_text_hook)
    zero_text_probs = _run_inference(model, batches, device)
    handle.remove()

    # Zero-video pass
    def zero_video_hook(module, input, output):
        output.last_hidden_state = torch.zeros_like(output.last_hidden_state)
        return output

    handle = model.video_model.register_forward_hook(zero_video_hook)
    zero_video_probs = _run_inference(model, batches, device)
    handle.remove()

    # Zero-tabular pass
    def zero_tabular_hook(module, input, output):
        return torch.zeros_like(output)

    handle = model.tabular_mlp.register_forward_hook(zero_tabular_hook)
    zero_tabular_probs = _run_inference(model, batches, device)
    handle.remove()

    # Compute metrics
    conditions = {
        "Full Model": full_probs,
        "Zero Text": zero_text_probs,
        "Zero Video": zero_video_probs,
        "Zero Tabular": zero_tabular_probs,
    }
    metrics = {}
    for name, probs in conditions.items():
        preds_binary = (probs >= THRESHOLD).astype(int)
        metrics[name] = {
            "ROC-AUC": roc_auc_score(labels_all, probs),
            "PR-AUC": average_precision_score(labels_all, probs),
            "F1": f1_score(labels_all, preds_binary, zero_division=0),
        }

    _plot_ablation(metrics)


def _plot_ablation(metrics):
    """Grouped bar chart: 4 conditions × 3 metrics."""
    conditions = list(metrics.keys())
    metric_names = ["ROC-AUC", "PR-AUC", "F1"]
    x = np.arange(len(conditions))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#4472C4", "#ED7D31", "#70AD47"]

    for i, metric in enumerate(metric_names):
        values = [metrics[c][metric] for c in conditions]
        bars = ax.bar(x + i * width, values, width, label=metric, color=colors[i])
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )

    ax.set_xlabel("Condition")
    ax.set_ylabel("Score")
    ax.set_title("Ablation Study: Modality Contribution", fontweight="bold")
    ax.set_xticks(x + width)
    ax.set_xticklabels(conditions)
    ax.legend()
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(ASSETS_DIR / "ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved ablation_study.png")


# -- Analysis 3: Permutation Importance --


def analysis_permutation_importance(model, combined, labels_all, device):
    """Shuffle one tabular column at a time, measure metric drop."""
    print("Running Analysis 3: Permutation Importance...")

    # Baseline: run model on unmodified data
    baseline_probs = _run_inference_from_combined(model, combined, device)
    baseline_preds = (baseline_probs >= THRESHOLD).astype(int)
    baseline_f1 = f1_score(labels_all, baseline_preds, zero_division=0)
    print(f"  Baseline F1: {baseline_f1:.4f}")

    importances = []
    for col_idx, feat_name in enumerate(TABULAR_FEATURE_NAMES):
        # Clone tabular features and shuffle this one column
        original_col = combined["tabular_features"][:, col_idx].clone()
        perm = torch.randperm(combined["tabular_features"].shape[0])
        combined["tabular_features"][:, col_idx] = original_col[perm]

        # Run inference with shuffled column
        shuffled_probs = _run_inference_from_combined(model, combined, device)
        shuffled_preds = (shuffled_probs >= THRESHOLD).astype(int)
        shuffled_f1 = f1_score(labels_all, shuffled_preds, zero_division=0)

        importance = baseline_f1 - shuffled_f1
        importances.append(importance)
        print(f"  {feat_name}: F1 drop = {importance:+.4f}")

        # Restore original column
        combined["tabular_features"][:, col_idx] = original_col

    _plot_permutation_importance(importances)


def _plot_permutation_importance(importances):
    """Horizontal bar chart sorted by importance."""
    importances = np.array(importances)
    order = np.argsort(importances)
    sorted_names = [TABULAR_FEATURE_NAMES[i] for i in order]
    sorted_importances = importances[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    colors = ["#ED7D31" if v > 0 else "#4472C4" for v in sorted_importances]
    bars = ax.barh(sorted_names, sorted_importances, color=colors)
    ax.set_xlabel("F1 Drop (baseline - shuffled)")
    ax.set_title("Permutation Importance: Tabular Features", fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, sorted_importances):
        ax.text(
            bar.get_width() + 0.002 if val >= 0 else bar.get_width() - 0.002,
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.4f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(
        ASSETS_DIR / "permutation_importance.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  Saved permutation_importance.png")


# -- Analysis 4: Tabular Feature Importance (Weight Magnitude) --


def analysis_tabular_importance(model):
    """L2 norm of first-layer weights -> feature importance bar chart."""
    print("Running Analysis 4: Tabular Feature Importance (Weight Magnitude)...")
    weights = model.tabular_mlp[0].weight.detach().cpu().numpy()  # (512, 19)
    importance = np.linalg.norm(weights, axis=0)  # (19,)

    # Sort by importance
    order = np.argsort(importance)
    sorted_names = [TABULAR_FEATURE_NAMES[i] for i in order]
    sorted_importance = importance[order]

    fig, ax = plt.subplots(figsize=(8, 7))
    bars = ax.barh(sorted_names, sorted_importance, color="#4472C4")
    ax.set_xlabel("L2 Norm of Weight Column")
    ax.set_title("Tabular Feature Importance (First MLP Layer)", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    for bar, val in zip(bars, sorted_importance):
        ax.text(
            bar.get_width() + 0.02,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}",
            va="center",
            fontsize=8,
        )

    plt.tight_layout()
    plt.savefig(
        ASSETS_DIR / "tabular_feature_importance.png", dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("  Saved tabular_feature_importance.png")


# -- Main --


def main():
    test_mode = "--test" in sys.argv
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)

    if test_mode:
        print("=== TEST MODE: using tiny data subset to verify script logic ===\n")

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"Using device: {device}")

    model, processor, test_dataset, device = load_model_and_data(device)
    print(f"Test dataset size: {len(test_dataset)}")

    # Use --test flag to test functionality of script on small subset of data
    n_inference = 8 if test_mode else min(200, len(test_dataset))
    n_analysis = 8 if test_mode else min(N_SAMPLES, len(test_dataset))

    # Inference Examples
    analysis_inference_examples(model, processor, test_dataset, device,
                                n_examples=n_inference)

    # Pre-process shared data for Ablation Study and Permutation Importance
    indices = list(range(n_analysis))
    print(f"\nPre-processing {n_analysis} examples for ablation + permutation analyses...")
    batches = process_examples(processor, test_dataset, indices, device)
    combined = concat_batches(batches)
    labels_all = combined["labels"][:, 2].numpy()

    # Ablation Study
    analysis_ablation(model, batches, labels_all, device)

    # Permutation Importance
    analysis_permutation_importance(model, combined, labels_all, device)

    # Tabular Feature Importance (weight magnitude)
    analysis_tabular_importance(model)

    print("\nAll analyses complete")


if __name__ == "__main__":
    main()
