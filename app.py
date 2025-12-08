from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    jsonify,
    flash,
)
from config import Config
from storage import init_db, save_run, list_runs, get_run, get_run_metrics
from summarizer import summarize_single_run, summarize_inline_comparison, summarize_multi_run_insights
from probing_core import load_dataset, run_layerwise_probe
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
# ----------------------------------------------------
# APP + DB
# ----------------------------------------------------

app = Flask(__name__)
app.config.from_object(Config)
init_db()

# ----------------------------------------------------
# NAV BAR
# ----------------------------------------------------

@app.context_processor
def inject_globals():
    return {
        "NAV_ITEMS": [
            ("overview", "Overview"),
            ("explorer", "Explorer"),
            ("run_probes", "Run Probes"),
            ("history", "History"),
            ("compare", "Compare"),
            ("interpret", "Interpret"),
            ("insights_ai", "Insights AI"),
        ],
        "config": app.config,
    }


def _train_probe_and_predict(texts, labels, info, model_name, text_query, max_train=400):
    """
    Train a tiny CLS-based logistic probe on a subset of (texts, labels),
    then predict for `text_query`.

    This powers the Live API prediction in Interpret.
    """
    if len(texts) == 0:
        return None

    # Make labels contiguous 0..K-1
    unique_ids = sorted(set(labels))
    id_to_idx = {lab: i for i, lab in enumerate(unique_ids)}
    y = np.array([id_to_idx[l] for l in labels], dtype=np.int64)

    # Label names (fall back to raw ids)
    label_names = info.get("label_names")
    if not label_names:
        label_names = [str(u) for u in unique_ids]

    # Sub-sample for fast training
    n = len(texts)
    max_train = min(max_train, n)
    if n > max_train:
        idx = np.random.choice(n, max_train, replace=False)
        train_texts = [texts[i] for i in idx]
        y_train = y[idx]
    else:
        train_texts = texts
        y_train = y

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(
        model_name,
        output_hidden_states=False,
    ).to(DEVICE)
    model.eval()

    def embed(batch_texts):
        enc = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt",
        ).to(DEVICE)
        with torch.no_grad():
            out = model(**enc)
            cls = out.last_hidden_state[:, 0, :].detach().cpu().numpy()
        return cls

    X_train = embed(train_texts)

    clf = LogisticRegression(
        max_iter=250,
        multi_class="auto",
        solver="lbfgs",
    )
    clf.fit(X_train, y_train)

    X_q = embed([text_query])
    proba = clf.predict_proba(X_q)[0]

    pred_idx = int(proba.argmax())
    # Map back through continuous label ids if needed
    if pred_idx < len(label_names):
        pred_label = label_names[pred_idx]
    else:
        pred_label = str(pred_idx)

    pairs = list(zip(label_names, proba.tolist()))

    return {
        "pred_label": pred_label,
        "probs": proba.tolist(),
        "label_names": label_names,
        "pairs": pairs,
    }

# ----------------------------------------------------
# PAGES
# ----------------------------------------------------

@app.route("/")
def overview():
    runs = list_runs(limit=5)
    total_runs = len(list_runs(limit=1000))
    return render_template(
        "overview.html",
        runs=runs,
        total_runs=total_runs,
        datasets=Config.KAGGLE_DATASETS,
        models=Config.TRANSFORMER_MODELS,
    )


@app.route("/explorer", methods=["GET", "POST"])
def explorer():
    dataset_key = request.form.get("dataset") or "amazon_reviews"
    max_samples = int(request.form.get("max_samples") or Config.DEFAULT_MAX_SAMPLES)

    texts, labels, info = load_dataset(dataset_key, max_samples)
    sample_preview = list(zip(texts[:5], labels[:5]))

    return render_template(
        "explorer.html",
        dataset_key=dataset_key,
        info=info,
        preview=sample_preview,
        datasets=Config.KAGGLE_DATASETS,
        max_samples=max_samples,
    )


# ----------------------------------------------------
# RUN PROBES (unchanged)
# ----------------------------------------------------

@app.route("/run-probes", methods=["GET", "POST"])
def run_probes():
    if request.method == "GET":
        return render_template(
            "run_probes.html",
            datasets=Config.KAGGLE_DATASETS,
            models=Config.TRANSFORMER_MODELS,
            default_max_samples=Config.DEFAULT_MAX_SAMPLES,
        )

    dataset_key = request.form.get("dataset")
    max_samples = int(request.form.get("max_samples") or Config.DEFAULT_MAX_SAMPLES)
    model_names = request.form.getlist("models") or [Config.TRANSFORMER_MODELS[0]]

    texts, labels, info = load_dataset(dataset_key, max_samples)

    all_metrics = {}
    for m in model_names:
        metrics = run_layerwise_probe(m, texts, labels)
        all_metrics[m] = metrics

    summary_md = summarize_single_run(
        dataset_name=dataset_key,
        task_name=info["task_name"],
        sample_size=info["num_samples"],
        model_metrics=all_metrics,
    )

    run_id = save_run(
        dataset_key=dataset_key,
        dataset_name=info["task_name"],
        models=model_names,
        sample_size=info["num_samples"],
        mode="single",
        summary=summary_md,
        metrics=all_metrics,
    )

    flash("Probe run completed!", "success")
    return redirect(url_for("results", run_id=run_id))


@app.route("/results/<int:run_id>")
def results(run_id):
    run = get_run(run_id)
    if run is None:
        flash("Run not found", "danger")
        return redirect(url_for("overview"))

    metrics = get_run_metrics(run_id)
    return render_template("results.html", run=run, metrics=metrics)


@app.route("/history")
def history():
    runs = list_runs(limit=100)
    return render_template("history.html", runs=runs)


# ----------------------------------------------------
# COMPARE (unchanged)
# ----------------------------------------------------

@app.route("/compare", methods=["GET", "POST"])
def compare():
    result = None

    if request.method == "POST":
        dataset_key = request.form.get("dataset")
        max_samples = int(request.form.get("max_samples") or Config.DEFAULT_MAX_SAMPLES)
        model_a = request.form.get("model_a")
        model_b = request.form.get("model_b")

        if not dataset_key or not model_a or not model_b:
            flash("Please choose a dataset and two models.", "danger")
            return redirect(url_for("compare"))

        if model_a == model_b:
            flash("Please pick two different models.", "danger")
            return redirect(url_for("compare"))

        texts, labels, info = load_dataset(dataset_key, max_samples)

        metrics_a = run_layerwise_probe(model_a, texts, labels)
        metrics_b = run_layerwise_probe(model_b, texts, labels)

        summary = summarize_inline_comparison(
            dataset_name=dataset_key,
            task_name=info["task_name"],
            sample_size=info["num_samples"],
            model_a=model_a,
            model_b=model_b,
            metrics_a=metrics_a,
            metrics_b=metrics_b,
        )

        result = {
            "dataset_key": dataset_key,
            "task_name": info["task_name"],
            "sample_size": info["num_samples"],
            "model_a": model_a,
            "model_b": model_b,
            "metrics_a": metrics_a,
            "metrics_b": metrics_b,
            "summary": summary,
        }

    return render_template(
        "compare.html",
        datasets=Config.KAGGLE_DATASETS,
        models=Config.TRANSFORMER_MODELS,
        result=result,
    )


# ----------------------------------------------------
# CHART API
# ----------------------------------------------------

@app.route("/api/run/<int:run_id>/metrics")
def api_run_metrics(run_id):
    metrics = get_run_metrics(run_id)
    return jsonify(metrics)


# ----------------------------------------------------
# HELPER — normalize attribution format
# ----------------------------------------------------

def normalize_attrib(tokens, raw_layers):
    """
    Converts variable backend output to strict frontend format:
    {
        "tokens": [...],
        "layers": {"0":[...], "1":[...], ...}
    }
    """

    layers = {}

    # Case: raw_layers = list of per-layer lists
    if isinstance(raw_layers, list) and isinstance(raw_layers[0], list):
        for i, scores in enumerate(raw_layers):
            layers[str(i)] = scores
        return {"tokens": tokens, "layers": layers}

    return {"tokens": tokens, "layers": {}}


# ----------------------------------------------------
# ATTRIBUTION ENGINE — activation, attention, grad
# ----------------------------------------------------

def compute_attributions(method, model, tokenizer, text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=128)
    outputs = model(**inputs, output_hidden_states=True, output_attentions=True)

    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])

    # ------------------------------
    # Activation Energy
    # ------------------------------
    if method == "activation":
        norms = [
            torch.norm(h[0], dim=-1).detach().tolist()
            for h in outputs.hidden_states
        ]
        return normalize_attrib(tokens, norms)

    # ------------------------------
    # Attention Rollout
    # ------------------------------
    if method == "attention":
        rollout = []
        for layer in outputs.attentions:
            a = layer[0].mean(0).detach().numpy()
            rollout.append(a.mean(axis=0).tolist())
        return normalize_attrib(tokens, rollout)

    # ------------------------------
    # Grad × Input
    # ------------------------------
    if method == "grad":
        model.zero_grad()
        probs = outputs.logits.softmax(dim=-1)[0]
        top = probs.max()
        top.backward()

        # fix gradient retrieval
        grads = []
        for h in outputs.hidden_states:
            if h.requires_grad:
                grads.append(h.grad.abs().mean(-1).detach().tolist())
            else:
                grads.append([0.0] * len(tokens))

        return normalize_attrib(tokens, grads)

    return normalize_attrib(tokens, [])


# ----------------------------------------------------
# INTERPRET PAGE
# ----------------------------------------------------

# ----------------------------------------------------
# INTERPRET (fixed)
# ----------------------------------------------------
from attribution import compute_token_attributions, DEVICE  # use your file

@app.route("/interpret", methods=["GET", "POST"])
def interpret():
    """
    Token-level heatmap + Live API prediction.
    """
    datasets = Config.KAGGLE_DATASETS
    models = Config.TRANSFORMER_MODELS
    methods = {
        "activation": "Activation norm",
        "attention": "Attention rollout",
        "grad_input": "Grad × Input",
    }

    attribution = None
    meta = None
    prediction = None

    if request.method == "POST":
        dataset_key = request.form.get("dataset") or next(iter(datasets.keys()))
        model_name = request.form.get("model") or None
        method = request.form.get("method") or "activation"
        sample_idx = int(request.form.get("sample_idx") or 0)
        max_samples = int(
            request.form.get("max_samples") or Config.DEFAULT_MAX_SAMPLES
        )
        custom_text = (request.form.get("custom_text") or "").strip()

        if not model_name:
            flash("Please pick a model to interpret.", "danger")
            return redirect(url_for("interpret"))

        # Always load dataset so we can train a tiny probe for prediction
        texts, labels, info = load_dataset(dataset_key, max_samples)

        if custom_text:
            text_source = "custom"
            text = custom_text
        else:
            text_source = "dataset"
            if sample_idx < 0 or sample_idx >= len(texts):
                sample_idx = 0
            text = texts[sample_idx]

        # Token-level attribution
        attribution = compute_token_attributions(
            text=text,
            model_name=model_name,
            method=method,
        )

        # Live API prediction using small CLS probe
        try:
            prediction = _train_probe_and_predict(
                texts=texts,
                labels=labels,
                info=info,
                model_name=model_name,
                text_query=text,
                max_train=400,
            )
        except Exception as e:  # don't crash UI if probe fails
            prediction = None

        meta = {
            "dataset_key": dataset_key,
            "task_name": info.get("task_name", dataset_key),
            "model_name": model_name,
            "method": method,
            "method_readable": attribution["method_readable"],
            "sample_idx": sample_idx,
            "text_source": text_source,
            "raw_text": text,
        }

    return render_template(
        "interpret.html",
        datasets=datasets,
        models=models,
        methods=methods,
        attribution=attribution,
        meta=meta,
        prediction=prediction,
    )


@app.route("/insights", methods=["GET", "POST"])
@app.route("/insights-ai", methods=["GET", "POST"])
def insights_ai():
    metric = request.form.get("metric") or "accuracy"
    dataset_filter = request.form.get("dataset") or "all"

    runs = list_runs(limit=None)
    datasets = Config.KAGGLE_DATASETS

    if not runs:
        return render_template(
            "insights.html",
            runs=[],
            datasets=datasets,
            metric=metric,
            dataset_filter=dataset_filter,
            insights_summary=None,
            insights_data=None,
        )

    model_stats = {}  # (model, dataset) -> {model, dataset, best, runs}
    layer_stats = {}  # layer -> [metric values]

    for r in runs:
        run_id = r["id"]
        dataset_key = r.get("dataset_key") or r.get("dataset_name")

        if dataset_filter != "all" and dataset_key != dataset_filter:
            continue

        metrics = get_run_metrics(run_id)
        if not metrics:
            continue

        for model_name, layers in metrics.items():
            key = (model_name, dataset_key)
            best_val = max(l[metric] for l in layers if l[metric] is not None)

            stat = model_stats.setdefault(
                key,
                {
                    "model": model_name,
                    "dataset": dataset_key,
                    "best": best_val,
                    "runs": 0,
                },
            )
            stat["best"] = max(stat["best"], best_val)
            stat["runs"] += 1

            for lm in layers:
                layer_idx = lm["layer_index"]
                val = lm[metric]
                if val is None:
                    continue
                layer_stats.setdefault(layer_idx, []).append(val)

    model_stats_list = list(model_stats.values())
    layer_signature = []
    for layer_idx, vals in sorted(layer_stats.items()):
        if not vals:
            continue
        mean_val = float(sum(vals) / len(vals))
        layer_signature.append(
            {
                "layer": int(layer_idx),
                "mean": mean_val,
                "count": len(vals),
            }
        )

    insights_summary = summarize_multi_run_insights(
        metric_key=metric,
        model_stats=model_stats_list,
        layer_signature=layer_signature,
    )

    insights_data = {
        "metric": metric,
        "models": model_stats_list,
        "layers": layer_signature,
    }

    return render_template(
        "insights.html",
        runs=runs,
        datasets=datasets,
        metric=metric,
        dataset_filter=dataset_filter,
        insights_summary=insights_summary,
        insights_data=insights_data,
    )


# ----------------------------------------------------
# MAIN
# ----------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
