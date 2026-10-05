# predict.py
"""
Predict stage.

Final production classifiers: for each embedding source (domain_sequence,
orf_sequence) and category, fits PCA + LogisticRegression on the ENTIRE
expanded_dataset.csv (no train/test split — validate.py already measured
accuracy on held-out splits; this stage exists to produce real annotations),
using the same pooled layer range and fixed classifier_C
(config.PARAMS["classifier_C"]) as validate.py's pooled substage — no
per-category/source tuning. Predicts probabilities/entropy for every
sequence in unknowns.csv and writes updated_unknowns.csv — unknowns.csv
itself is left untouched.
"""
import os
import numpy as np
import pandas as pd
import config

from utils import inference_utils as inf
from utils.feature_cache import FeatureCache

_SOURCE_LABEL = {"domain_sequence": "domain", "orf_sequence": "orf"}


def run(paths, params, hpc):
    esm_settings = config.PARAMS["inference_settings"]
    categories   = list(config.CATEGORIES.keys())
    source_param = params.get("embedding_source", "both")
    layer_range  = params.get("layer_range", [11, 19])
    seed         = params.get("seed", 42)
    layers       = list(range(layer_range[0], layer_range[1] + 1))

    out_dir = paths["predict_dir"]
    os.makedirs(out_dir, exist_ok=True)

    expanded_df  = pd.read_csv(os.path.join(paths["expand_dir"], "expanded_dataset.csv"))
    train_orf_df = inf.orf_level_frame(expanded_df)
    print(f"[predict] Loaded expanded_dataset.csv: {len(expanded_df)} rows, "
          f"{len(train_orf_df)} unique ORFs (training data — entire dataset, no split)")

    unknown_df     = pd.read_csv(paths["unknowns"])
    unknown_orf_df = inf.orf_level_frame(unknown_df)
    print(f"[predict] Loaded unknowns.csv: {len(unknown_df)} rows, "
          f"{len(unknown_orf_df)} unique ORFs; pooling layers {layer_range[0]}-{layer_range[1]}, "
          f"C={config.PARAMS['classifier_C']}")

    train_frames = {
        "domain_sequence": (expanded_df,  "domain_id", "domain_sequence"),
        "orf_sequence":    (train_orf_df, "orf_id",    "orf_sequence"),
    }
    unknown_frames = {
        "domain_sequence": (unknown_df,     "domain_id", "domain_sequence"),
        "orf_sequence":    (unknown_orf_df, "orf_id",    "orf_sequence"),
    }
    sources = ["domain_sequence", "orf_sequence"] if source_param == "both" else [source_param]

    pred_df = unknown_df.copy()

    for sequence_source in sources:
        print(f"\n{'=' * 60}")
        print(f"[predict] Source: {sequence_source}")
        print(f"{'=' * 60}")

        train_df, train_id_col, train_seq_col = train_frames[sequence_source]
        unk_df, unk_id_col, unk_seq_col       = unknown_frames[sequence_source]
        source_label = _SOURCE_LABEL[sequence_source]
        embed_dir    = os.path.join(paths["embeddings"], sequence_source)

        # Training embeddings: reuse validate's already-warm cache (pure I/O).
        train_ids      = train_df[train_id_col].tolist()
        train_seq_dict = dict(zip(train_df[train_id_col], train_df[train_seq_col]))
        inf.prepare_embeddings(
            train_ids, train_seq_dict, paths, esm_settings, hpc, layers,
            embed_dir=embed_dir, cache_key="validate",
        )
        train_cache = FeatureCache(os.path.join(embed_dir, "csv", "validate"), train_ids, layers)

        # Unknown embeddings: own cache_key — real new ESM2 extraction likely.
        unk_ids      = unk_df[unk_id_col].tolist()
        unk_seq_dict = dict(zip(unk_df[unk_id_col], unk_df[unk_seq_col]))
        inf.prepare_embeddings(
            unk_ids, unk_seq_dict, paths, esm_settings, hpc, layers,
            embed_dir=embed_dir, cache_key="predict",
        )
        unk_cache = FeatureCache(os.path.join(embed_dir, "csv", "predict"), unk_ids, layers)

        missing_train = [l for l in layers if train_cache.get_all(l) is None]
        missing_unk   = [l for l in layers if unk_cache.get_all(l) is None]
        if missing_train or missing_unk:
            print(f"  [predict] Source {sequence_source}: missing cached layers "
                  f"(train={missing_train}, unknown={missing_unk}), skipping source")
            continue

        X_train_pooled = np.concatenate([train_cache.get_all(l) for l in layers], axis=1)
        X_unk_pooled   = np.concatenate([unk_cache.get_all(l) for l in layers], axis=1)

        C = config.PARAMS["classifier_C"]
        for category in categories:
            labels_full = train_df[category].values
            valid = pd.notna(labels_full)
            if valid.sum() == 0 or len(np.unique(labels_full[valid])) < 2:
                print(f"  [predict] {category}: fewer than 2 labeled classes, skipping")
                continue

            X_train_v = X_train_pooled[valid]
            y_train_v = labels_full[valid]

            print(f"  [predict] {category}: n_train={len(y_train_v)}, n_unknown={len(unk_ids)}")

            result = inf.fit_and_predict(X_train_v, y_train_v, X_unk_pooled, C, seed)

            pred_col    = f"{category}_pred_{source_label}"
            entropy_col = f"{category}_entropy_{source_label}"
            if pred_col not in pred_df.columns:
                pred_df[pred_col] = None
                pred_df[entropy_col] = np.nan
                for cls in sorted(np.unique(y_train_v)):
                    pred_df[f"{category}_prob_{cls}_{source_label}"] = np.nan

            # merge predictions back onto pred_df (unknowns-grain), broadcasting
            # orf-grain predictions to every domain row sharing that orf_id —
            # same pattern as validate.py's pooled_substage.
            id_to_row = {uid: i for i, uid in enumerate(unk_ids)}
            mask_in_pred = pred_df[unk_id_col].isin(unk_ids)
            for idx in pred_df.index[mask_in_pred]:
                i = id_to_row[pred_df.at[idx, unk_id_col]]
                pred_df.at[idx, pred_col] = result["preds"][i]
                pred_df.at[idx, entropy_col] = result["entropies"][i]
                for ci, cls in enumerate(result["classes"]):
                    pred_df.at[idx, f"{category}_prob_{cls}_{source_label}"] = result["probs"][i, ci]

    output_path = os.path.join(out_dir, "updated_unknowns.csv")
    pred_df.to_csv(output_path, index=False)
    print(f"\n[predict] Wrote {output_path}: {len(pred_df)} rows")
