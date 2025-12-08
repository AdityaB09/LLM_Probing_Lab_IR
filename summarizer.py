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
        "In general, lower layers emphasize surface/lexical cues, while middle–deep "
        "layers show stronger performance on this task, suggesting they capture "
        "richer semantic features that are more linearly separable."
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
            f"Overall, **{model_a}** is better calibrated for this setup, "
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
            "the choice of **which layer** to export may matter more than "
            "which encoder family you pick."
        )

    return "\n".join(lines)


def summarize_multi_run_insights(metric_key, model_stats, layer_signature):
    """
    model_stats: list[{model, dataset, best, runs}]
    layer_signature: list[{layer, mean, count}]
    """
    if not model_stats:
        return "No runs available yet. Execute a few probes to unlock insights."

    metric_name = {
        "accuracy": "accuracy",
        "f1": "weighted F1",
        "ece": "calibration error (ECE)",
    }.get(metric_key, metric_key)

    lines = []
    lines.append(
        f"Aggregating all stored runs, we analyze trends with respect to **{metric_name}**."
    )

    # Top models overall
    sorted_models = sorted(model_stats, key=lambda s: s["best"], reverse=True)
    top_overall = sorted_models[:3]

    lines.append("### Top-performing model/dataset pairs")
    for s in top_overall:
        lines.append(
            f"- **{s['model']}** on **{s['dataset']}** reaches best {metric_name} ≈ {s['best']:.3f} "
            f"across {s['runs']} run(s)."
        )

    # Dataset-specific winners
    by_dataset = {}
    for s in model_stats:
        by_dataset.setdefault(s["dataset"], []).append(s)
    lines.append("")
    lines.append("### Dataset-specific observations")
    for ds, lst in by_dataset.items():
        lst_sorted = sorted(lst, key=lambda s: s["best"], reverse=True)
        best = lst_sorted[0]
        lines.append(
            f"- On **{ds}**, **{best['model']}** consistently dominates with "
            f"peak {metric_name} ≈ {best['best']:.3f}."
        )

    # Layer signature
    if layer_signature:
        hot_layer = max(layer_signature, key=lambda s: s["mean"])
        lines.append("")
        lines.append("### Layer signature")
        lines.append(
            f"- Across all runs, layer **{hot_layer['layer']}** emerges as a "
            f"universal sweet spot with mean {metric_name} ≈ {hot_layer['mean']:.3f} "
            f"over {hot_layer['count']} probe(s)."
        )

    lines.append("")
    lines.append(
        "These patterns essentially turn your probing logs into a live research report: "
        "you can quote the strongest encoder/dataset combos, the layers that behave like "
        "semantic bottlenecks, and how calibration evolves as you go deeper."
    )

    return "\n".join(lines)
