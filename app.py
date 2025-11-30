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
from summarizer import summarize_single_run, summarize_inline_comparison
from probing_core import load_dataset, run_layerwise_probe
# (rest of imports unchanged)

app = Flask(__name__)
app.config.from_object(Config)

# Initialise DB
init_db()


@app.context_processor
def inject_globals():
    return {
        "NAV_ITEMS": [
            ("overview", "Overview"),
            ("explorer", "Explorer"),
            ("run_probes", "Run Probes"),
            ("history", "History"),
            ("compare", "Compare"),
        ],
        "config": app.config,
    }


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
    max_samples = int(
        request.form.get("max_samples") or Config.DEFAULT_MAX_SAMPLES
    )

    texts, labels, info = load_dataset(dataset_key, max_samples)
    # Simple stats
    sample_preview = list(zip(texts[:5], labels[:5]))
    return render_template(
        "explorer.html",
        dataset_key=dataset_key,
        info=info,
        preview=sample_preview,
        datasets=Config.KAGGLE_DATASETS,
        max_samples=max_samples,
    )


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
    max_samples = int(
        request.form.get("max_samples") or Config.DEFAULT_MAX_SAMPLES
    )
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
    return render_template(
        "results.html",
        run=run,
        metrics=metrics,
    )


@app.route("/history")
def history():
    runs = list_runs(limit=100)
    return render_template("history.html", runs=runs)


@app.route("/compare", methods=["GET", "POST"])
def compare():
    """
    Instant comparison:
    - user picks dataset, sample size, model A, model B
    - we run probes for both on the fly
    - show summary + graphs
    No dependence on existing runs in the DB.
    """
    result = None

    if request.method == "POST":
        dataset_key = request.form.get("dataset")
        max_samples = int(
            request.form.get("max_samples") or Config.DEFAULT_MAX_SAMPLES
        )
        model_a = request.form.get("model_a")
        model_b = request.form.get("model_b")

        if not dataset_key or not model_a or not model_b:
            flash("Please choose a dataset and two models.", "danger")
            return redirect(url_for("compare"))

        if model_a == model_b:
            flash("Please pick two different models to compare.", "danger")
            return redirect(url_for("compare"))

        texts, labels, info = load_dataset(dataset_key, max_samples)

        # Run probes (same data, two encoders)
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



@app.route("/api/run/<int:run_id>/metrics")
def api_run_metrics(run_id):
    # JSON endpoint to feed Chart.js
    metrics = get_run_metrics(run_id)
    return jsonify(metrics)


if __name__ == "__main__":
    app.run(debug=True)
