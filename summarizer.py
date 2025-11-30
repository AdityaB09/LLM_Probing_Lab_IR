import numpy as np


def summarize_single_run(dataset_name, task_name, sample_size, model_metrics):
    """
    model_metrics: dict[model_name] -> list[layer_metric_dict]
    """
    lines = []
    lines.append(
        f"For **{dataset_name}** ({task_name}) with {sample_size} samples, "
        f"we ran layerwise probes over {len(model_metrics)} transformer encoders."
    )

    # Best layer per model
    for model_name, layers in model_metrics.items():
        accs = np.array([m["accuracy"] for m in layers])
        f1s = np.array([m["f1"] for m in layers])
        eces = np.array([m["ece"] for m in layers])

        best_layer = int(accs.argmax())
        lines.append(
            f"- **{model_name}** peaks at layer **{best_layer}** "
            f"with accuracy ≈ {accs.max():.3f}, F1 ≈ {f1s.max():.3f}, "
            f"and ECE ≈ {eces[best_layer]:.3f}."
        )

    # Cross-model highlight
    all_records = []
    for model_name, layers in model_metrics.items():
        for m in layers:
            all_records.append((model_name, m["layer_index"], m["accuracy"]))
    best_global = max(all_records, key=lambda x: x[2])
    lines.append(
        f"Across all models, the best probe is **{best_global[0]} / layer {best_global[1]}** "
        f"with accuracy ≈ {best_global[2]:.3f}."
    )

    lines.append(
        "Overall, lower layers tend to capture surface/lexical cues, "
        "while middle–deep layers show stronger performance on this task, "
        "indicating that retrieval systems might benefit from extracting embeddings "
        "from those layers rather than the default final layer."
    )

    return "\n".join(lines)


def summarize_inline_comparison(
    dataset_name,
    task_name,
    sample_size,
    model_a,
    model_b,
    metrics_a,
    metrics_b,
):
    """
    metrics_a / metrics_b: list[layer_metric_dict] (output of run_layerwise_probe)
    """
    acc_a = np.array([m["accuracy"] for m in metrics_a])
    acc_b = np.array([m["accuracy"] for m in metrics_b])
    f1_a = np.array([m["f1"] for m in metrics_a])
    f1_b = np.array([m["f1"] for m in metrics_b])

    best_a = int(acc_a.argmax())
    best_b = int(acc_b.argmax())

    lines = []
    lines.append(
        f"Comparing **{model_a}** vs **{model_b}** on **{task_name}** "
        f"({dataset_name}) with {sample_size} samples."
    )
    lines.append(
        f"- {model_a} peaks at layer **{best_a}** with accuracy ≈ {acc_a.max():.3f} "
        f"and weighted F1 ≈ {f1_a.max():.3f}."
    )
    lines.append(
        f"- {model_b} peaks at layer **{best_b}** with accuracy ≈ {acc_b.max():.3f} "
        f"and weighted F1 ≈ {f1_b.max():.3f}."
    )

    if acc_a.max() > acc_b.max() + 0.01:
        lines.append(
            f"Overall, **{model_a}** is better calibrated for this task, "
            "suggesting its internal representations are more linearly separable."
        )
    elif acc_b.max() > acc_a.max() + 0.01:
        lines.append(
            f"Overall, **{model_b}** dominates on this dataset, "
            "indicating that its architecture captures richer task-specific cues."
        )
    else:
        lines.append(
            "Both models show comparable peak performance; in this regime, "
            "choice of **which layer** to export may matter more than which model family."
        )

    return "\n".join(lines)
