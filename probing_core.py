import os
from pathlib import Path
import re
import bz2  # handle .bz2 archives

import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

from config import Config

DATA_RAW_DIR = Path(Config.DATA_RAW_DIR)
DATA_CACHE_DIR = Path(Config.DATA_CACHE_DIR)
DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
DATA_CACHE_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def ensure_kaggle_download(dataset_key):
    """
    Use kaggle CLI to download if the expected file is missing.
    Handles special cases like amazon_reviews train.ft.txt.bz2.
    """
    meta = Config.KAGGLE_DATASETS[dataset_key]
    target_file = DATA_RAW_DIR / dataset_key / meta["file"]
    ds_dir = target_file.parent
    ds_dir.mkdir(parents=True, exist_ok=True)

    # Already there?
    if target_file.exists():
        return target_file

    slug = meta["slug"]
    cmd = f'kaggle datasets download -d {slug} -p "{ds_dir}" --unzip'
    print("[KAGGLE]", cmd)
    exit_code = os.system(cmd)
    if exit_code != 0:
        raise RuntimeError("Kaggle download failed. Check API key and slug.")

    # Special handling for Amazon reviews (train.ft.txt / .bz2)
    if dataset_key == "amazon_reviews":
        candidates = list(ds_dir.glob("train.ft*"))
        if not candidates:
            raise FileNotFoundError(
                f"No train.ft* file found in {ds_dir} after Kaggle download."
            )

        txt = [c for c in candidates if c.name == "train.ft.txt"]
        if txt:
            return txt[0]

        bz_candidates = [c for c in candidates if c.suffix == ".bz2"]
        if not bz_candidates:
            return candidates[0]

        bz_path = bz_candidates[0]
        out_path = ds_dir / "train.ft.txt"
        print(f"[KAGGLE] Decompressing {bz_path.name} -> {out_path.name}")
        with bz2.open(bz_path, "rb") as f_in, open(out_path, "wb") as f_out:
            f_out.write(f_in.read())
        return out_path

    # Other datasets: expect the configured file, but be forgiving
    if not target_file.exists():
        wildcard = list(ds_dir.glob(meta["file"]))
        if wildcard:
            return wildcard[0]
        raise FileNotFoundError(
            f"Expected file {target_file} not found after kaggle download."
        )

    return target_file


def clean_text(s):
    s = str(s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def load_dataset(dataset_key, max_samples):
    """
    Returns texts, labels, and a small info dict.
    """
    meta = Config.KAGGLE_DATASETS[dataset_key]
    file_path = ensure_kaggle_download(dataset_key)

    if dataset_key == "sarcasm":
        df = pd.read_json(file_path, lines=True)
        df = df[[meta["text_col"], meta["label_col"]]]
        df[meta["text_col"]] = df[meta["text_col"]].map(clean_text)

    elif dataset_key == "fake_news":
        # Fake.csv + True.csv -> 0/1 labels
        fake_path = file_path
        true_path = file_path.parent / "True.csv"
        if not true_path.exists():
            raise FileNotFoundError(
                f"True.csv not found next to {fake_path}. "
                "Check the downloaded files in data/raw/fake_news."
            )
        df_fake = pd.read_csv(fake_path)
        df_true = pd.read_csv(true_path)
        df_fake["label_int"] = 0
        df_true["label_int"] = 1
        df = pd.concat(
            [
                df_fake.rename(columns={"text": "text"}),
                df_true.rename(columns={"text": "text"}),
            ],
            ignore_index=True,
        )
        df = df[["text", "label_int"]]
        df["text"] = df["text"].map(clean_text)
        meta_text_col, meta_label_col = "text", "label_int"
        meta = {**meta, "text_col": meta_text_col, "label_col": meta_label_col}

    elif dataset_key == "amazon_reviews":
        # fastText format; use errors='ignore' to avoid UnicodeDecodeError
        texts = []
        labels = []
        with open(file_path, encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split(" ", 1)
                if len(parts) != 2:
                    continue
                label_str, text = parts
                # __label__1 or __label__2 -> 0/1
                label = 1 if "2" in label_str else 0
                texts.append(clean_text(text))
                labels.append(label)
        df = pd.DataFrame({"text": texts, "label": labels})
        meta = {**meta, "text_col": "text", "label_col": "label"}

    elif dataset_key == "hate_speech":
        df = pd.read_csv(file_path)
        df = df[[meta["text_col"], meta["label_col"]]]
        df[meta["text_col"]] = df[meta["text_col"]].map(clean_text)

    else:
        raise ValueError(f"Unknown dataset {dataset_key}")

    # Shuffle & sample
    df = df.dropna().sample(frac=1.0, random_state=42).reset_index(drop=True)
    max_samples = min(max_samples, Config.MAX_SAMPLES_HARD_LIMIT, len(df))
    df = df.iloc[:max_samples]

    texts = df[meta["text_col"]].tolist()
    labels = df[meta["label_col"]].astype(int).tolist()

    info = {
        "num_samples": len(df),
        "num_classes": len(set(labels)),
        "class_counts": df[meta["label_col"]].value_counts().to_dict(),
        "task_name": meta["task_name"],
    }
    return texts, labels, info


def expected_calibration_error(y_true, probs, num_bins=10):
    """
    Simple ECE implementation for multiclass (using max prob).
    """
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)

    ece = 0.0
    bin_boundaries = np.linspace(0.0, 1.0, num_bins + 1)
    for i in range(num_bins):
        start, end = bin_boundaries[i], bin_boundaries[i + 1]
        mask = (confidences > start) & (confidences <= end)
        if not np.any(mask):
            continue
        bin_conf = confidences[mask].mean()
        bin_acc = correct[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * (mask.sum() / len(confidences))
    return ece


def run_layerwise_probe(model_name, texts, labels, random_state=42):
    """
    Train independent logistic regression probes over each layer.
    Returns list of metric dicts (per layer).
    """
    print(f"[MODEL] Loading {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name, output_hidden_states=True)
    model.to(DEVICE)
    model.eval()

    print("[ENCODING] Computing hidden states...")
    all_hidden = []
    all_labels = np.array(labels)

    with torch.no_grad():
        for batch_start in tqdm(range(0, len(texts), 16)):
            batch_texts = texts[batch_start : batch_start + 16]
            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt",
            ).to(DEVICE)

            outputs = model(**enc)
            hidden_states = outputs.hidden_states  # tuple: (layer0,...,layerL)

            batch_layers = [h[:, 0, :].cpu().numpy() for h in hidden_states]
            all_hidden.append(batch_layers)

    num_layers = len(all_hidden[0])
    layer_arrays = []
    for layer_idx in range(num_layers):
        layer_batches = [b[layer_idx] for b in all_hidden]
        layer_arrays.append(np.vstack(layer_batches))

    metrics = []

    for layer_idx in range(num_layers):
        X = layer_arrays[layer_idx]
        y = all_labels

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=random_state, stratify=y
        )

        clf = LogisticRegression(
            max_iter=200, n_jobs=-1, multi_class="auto", solver="lbfgs"
        )
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_prob = clf.predict_proba(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="weighted")
        ece = expected_calibration_error(y_test, y_prob)

        metrics.append(
            {
                "layer_index": layer_idx,
                "accuracy": float(acc),
                "f1": float(f1),
                "ece": float(ece),
                "extra": {
                    "n_test": len(y_test),
                },
            }
        )

    return metrics
