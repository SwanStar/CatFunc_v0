# validate.py
"""
Validate stage.

Both substages use a single fixed logistic-regression C
(config.PARAMS["classifier_C"]) for every embedding source and category —
no per-combination CV tuning. Tuning was dropped after finding it let C
vary across categories for no real accuracy benefit (7 of 8
(embedding_source, category) combinations independently converged to the
same C anyway) while making per-row entropy incomparable between
categories, since entropy's magnitude is directly shaped by
regularization strength, not just genuine label uncertainty.

trajectory substage: for each embedding source (domain_sequence,
orf_sequence) and category, sweeps the fixed C across all 33 layers in
both train/test directions (vg1->vg2 and vg2->vg1), recording PCA
dimensionality and test accuracy.

pooled substage: pools a config-set layer range (concatenated features)
and runs the same two-direction logistic regression once per layer range,
at the same fixed C. Writes per-row predictions/probabilities/entropy back onto
updated_expanded_dataset.csv (domain-source columns, then orf-source
columns), sorted by validation_group then descending misclassification
count, plus validation_summary.csv with per-class recall/precision/AUROC
and macro-averaged recall/precision/AUROC (each class weighted equally,
unlike overall_accuracy which is implicitly weighted by class size --
report macro_recall, not overall_accuracy, as the headline number under
class imbalance), plus a label-shuffled null comparison for all of these.
"""
import os
import numpy as np
import pandas as pd
import config

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from utils import inference_utils as inf
from utils import lda_utils
from utils.feature_cache import FeatureCache

_SOURCE_LABEL = {"domain_sequence": "domain", "orf_sequence": "orf"}


def _fit_and_score(X_train, y_train, X_test, y_test, C, seed):
    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    return float((preds == y_test).mean())


def _shuffle_within_groups(labels, groups, seed):
    """Permute labels independently within each group value — preserves each
    group's label marginals exactly, breaks the label-feature association."""
    rng = np.random.default_rng(seed)
    shuffled = labels.copy()
    for g in np.unique(groups):
        idx = np.where(groups == g)[0]
        shuffled[idx] = rng.permutation(labels[idx])
    return shuffled


def _classify_direction(X_raw, labels, vg, train_group, test_group, C, seed):
    """
    PCA-fit on train_group's rows, apply to test_group's rows, fit/predict
    a single LogisticRegression(C=C). Returns None if there isn't
    enough data to fit or score meaningfully.
    """
    train_mask = vg == train_group
    test_mask  = vg == test_group
    y_train, y_test = labels[train_mask], labels[test_mask]
    n_classes = len(np.unique(y_train))

    if n_classes < 2 or train_mask.sum() < n_classes + 1 or test_mask.sum() == 0:
        return None

    scaler, pca, X_train_pca = lda_utils.pca_fit(X_raw[train_mask])
    X_test_pca = lda_utils.pca_apply(scaler, pca, X_raw[test_mask])

    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    clf.fit(X_train_pca, y_train)
    preds   = clf.predict(X_test_pca)
    probs   = clf.predict_proba(X_test_pca)
    classes = clf.classes_

    accuracy  = float((preds == y_test).mean())
    entropies = np.array([inf.shannon_entropy_row(p) for p in probs])

    per_class = []
    aucs = []
    for ci, cls in enumerate(classes):
        y_bin = (y_test == cls).astype(int)
        if 0 < y_bin.sum() < len(y_bin):
            auc = float(roc_auc_score(y_bin, probs[:, ci]))
            aucs.append(auc)
        else:
            auc = float("nan")
        recall_mask = y_test == cls
        recall = float((preds[recall_mask] == cls).mean()) if recall_mask.sum() else float("nan")
        pred_mask = preds == cls
        precision = float((y_test[pred_mask] == cls).mean()) if pred_mask.sum() else float("nan")
        per_class.append({"class": cls, "ovr_auc": auc, "recall": recall, "precision": precision})

    macro_auc       = float(np.mean(aucs)) if aucs else float("nan")
    # Macro-averaged recall/precision: mean of the per-class values above,
    # each class weighted equally regardless of support -- unlike plain
    # accuracy, which is implicitly weighted by class size. This is the
    # number that should be treated as "the" headline summary metric for
    # this task, not overall_accuracy (see stage docstring / manuscript
    # decision: report macro recall, not accuracy, given class imbalance).
    # nanmean (not mean): unlike `aucs` above, per_class always has one
    # entry per class even when that class's recall/precision is NaN (zero
    # test/predicted examples of it) -- a plain mean would let one such
    # class poison the whole macro average.
    macro_recall    = float(np.nanmean([pc["recall"] for pc in per_class])) if per_class else float("nan")
    macro_precision = float(np.nanmean([pc["precision"] for pc in per_class])) if per_class else float("nan")

    return {
        "test_mask":       test_mask,
        "preds":           preds,
        "probs":           probs,
        "classes":         classes,
        "entropies":       entropies,
        "accuracy":        accuracy,
        "macro_auc":       macro_auc,
        "macro_recall":    macro_recall,
        "macro_precision": macro_precision,
        "per_class":       per_class,
        "n_train":         int(train_mask.sum()),
        "n_test":          int(test_mask.sum()),
    }


def pooled_substage(paths, params, hpc):
    categories    = list(config.CATEGORIES.keys())
    source_param  = params.get("embedding_source", "both")
    layer_range   = params.get("layer_range", [11, 19])
    shuffle_count = params.get("shuffle_count", 5)
    seed          = params.get("seed", 42)
    layers        = list(range(layer_range[0], layer_range[1] + 1))
    C             = config.PARAMS["classifier_C"]

    out_dir = paths["validate_dir"]
    os.makedirs(out_dir, exist_ok=True)

    expanded_df = pd.read_csv(os.path.join(paths["expand_dir"], "expanded_dataset.csv"))
    orf_df = inf.orf_level_frame(expanded_df)
    print(f"[validate.pooled] Loaded expanded_dataset.csv: {len(expanded_df)} rows, "
          f"{len(orf_df)} unique ORFs; pooling layers {layer_range[0]}-{layer_range[1]}, C={C}")

    source_frames = {
        "domain_sequence": (expanded_df, "domain_id", "domain_sequence"),
        "orf_sequence":    (orf_df,      "orf_id",    "orf_sequence"),
    }
    sources = ["domain_sequence", "orf_sequence"] if source_param == "both" else [source_param]

    pred_df = expanded_df.copy()
    summary_rows = []

    for sequence_source in sources:
        print(f"\n{'=' * 60}")
        print(f"[validate.pooled] Source: {sequence_source}")
        print(f"{'=' * 60}")

        df, id_col, seq_col = source_frames[sequence_source]
        embed_dir = os.path.join(paths["embeddings"], sequence_source)
        csv_dir   = os.path.join(embed_dir, "csv", "validate")
        all_ids   = df[id_col].tolist()
        source_label = _SOURCE_LABEL[sequence_source]

        cache = FeatureCache(csv_dir, all_ids, layers)
        missing_layers = [l for l in layers if cache.get_all(l) is None]
        if missing_layers:
            print(f"  [validate.pooled] Source {sequence_source}: missing cached layers "
                  f"{missing_layers}, skipping source")
            continue
        X_pooled_full = np.concatenate([cache.get_all(l) for l in layers], axis=1)

        for category in categories:
            labels_full = df[category].values
            vg_full     = df["validation_group"].values
            valid = pd.notna(labels_full) & pd.notna(vg_full)

            if valid.sum() == 0 or len(np.unique(labels_full[valid])) < 2:
                print(f"  [validate.pooled] {category}: fewer than 2 labeled classes, skipping")
                continue

            ids_v    = np.array(all_ids, dtype=object)[valid]
            labels_v = labels_full[valid]
            vg_v     = vg_full[valid]
            X_raw    = X_pooled_full[valid]

            print(f"  [validate.pooled] {category}: n={len(ids_v)}, "
                  f"pooled layers {layers[0]}-{layers[-1]}")

            direction_results = {}
            for train_group, test_group in (("vg1", "vg2"), ("vg2", "vg1")):
                result = _classify_direction(X_raw, labels_v, vg_v, train_group, test_group, C, seed)
                if result is None:
                    continue
                direction_results[(train_group, test_group)] = result

                task_id = f"train_{train_group}"
                summary_rows.append({
                    "embedding_source":  sequence_source,
                    "category":          category,
                    "task_id":           task_id,
                    "row_type":          "overall",
                    "class":             None,
                    "n_train":           result["n_train"],
                    "n_test":            result["n_test"],
                    "overall_accuracy":  result["accuracy"],
                    "macro_auc":         result["macro_auc"],
                    "macro_recall":      result["macro_recall"],
                    "macro_precision":   result["macro_precision"],
                })
                for pc in result["per_class"]:
                    summary_rows.append({
                        "embedding_source": sequence_source,
                        "category":         category,
                        "task_id":          task_id,
                        "row_type":         "per_class",
                        "class":            pc["class"],
                        "n_train":          result["n_train"],
                        "n_test":           result["n_test"],
                        "ovr_auc":          pc["ovr_auc"],
                        "recall":           pc["recall"],
                        "precision":        pc["precision"],
                    })

                # merge predictions back onto pred_df, broadcasting orf-grain
                # predictions to every domain row sharing that orf_id
                test_ids = ids_v[result["test_mask"]]
                pred_col    = f"{category}_pred_{source_label}"
                entropy_col = f"{category}_entropy_{source_label}"
                if pred_col not in pred_df.columns:
                    pred_df[pred_col] = None
                    pred_df[entropy_col] = np.nan
                    for cls in sorted(np.unique(labels_v)):
                        pred_df[f"{category}_prob_{cls}_{source_label}"] = np.nan

                id_to_row = {tid: i for i, tid in enumerate(test_ids)}
                mask_in_pred = pred_df[id_col].isin(test_ids)
                for idx in pred_df.index[mask_in_pred]:
                    i = id_to_row[pred_df.at[idx, id_col]]
                    pred_df.at[idx, pred_col] = result["preds"][i]
                    pred_df.at[idx, entropy_col] = result["entropies"][i]
                    for ci, cls in enumerate(result["classes"]):
                        pred_df.at[idx, f"{category}_prob_{cls}_{source_label}"] = result["probs"][i, ci]

            if shuffle_count > 0 and direction_results:
                shuffle_accum = {}
                for shuffle_idx in range(shuffle_count):
                    shuffled_labels = _shuffle_within_groups(labels_v, vg_v, seed + shuffle_idx + 1)
                    for train_group, test_group in (("vg1", "vg2"), ("vg2", "vg1")):
                        shuffle_result = _classify_direction(
                            X_raw, shuffled_labels, vg_v, train_group, test_group, C, seed
                        )
                        if shuffle_result is None:
                            continue
                        task_id = f"train_{train_group}"
                        bucket = shuffle_accum.setdefault(task_id, {
                            "accuracy": [], "macro_auc": [], "macro_recall": [], "macro_precision": [],
                        })
                        bucket["accuracy"].append(shuffle_result["accuracy"])
                        bucket["macro_auc"].append(shuffle_result["macro_auc"])
                        bucket["macro_recall"].append(shuffle_result["macro_recall"])
                        bucket["macro_precision"].append(shuffle_result["macro_precision"])

                for row in summary_rows:
                    if (row["embedding_source"] == sequence_source and row["category"] == category
                            and row["row_type"] == "overall" and row["task_id"] in shuffle_accum):
                        accs  = shuffle_accum[row["task_id"]]["accuracy"]
                        aucs  = shuffle_accum[row["task_id"]]["macro_auc"]
                        recs  = shuffle_accum[row["task_id"]]["macro_recall"]
                        precs = shuffle_accum[row["task_id"]]["macro_precision"]
                        row["shuffle_overall_accuracy_mean"] = float(np.mean(accs)) if accs else float("nan")
                        row["shuffle_overall_accuracy_std"]  = float(np.std(accs)) if accs else float("nan")
                        row["shuffle_macro_auc_mean"] = float(np.nanmean(aucs)) if aucs else float("nan")
                        row["shuffle_macro_auc_std"]  = float(np.nanstd(aucs)) if aucs else float("nan")
                        row["shuffle_macro_recall_mean"] = float(np.nanmean(recs)) if recs else float("nan")
                        row["shuffle_macro_recall_std"]  = float(np.nanstd(recs)) if recs else float("nan")
                        row["shuffle_macro_precision_mean"] = float(np.nanmean(precs)) if precs else float("nan")
                        row["shuffle_macro_precision_std"]  = float(np.nanstd(precs)) if precs else float("nan")

    def _count_misclassified(row):
        n = 0
        for cat in categories:
            for lbl in _SOURCE_LABEL.values():
                col = f"{cat}_pred_{lbl}"
                if col in pred_df.columns and pd.notna(row[col]) and row[col] != row[cat]:
                    n += 1
        return n

    pred_df["n_misclassified"] = pred_df.apply(_count_misclassified, axis=1)
    pred_df = pred_df.sort_values(["validation_group", "n_misclassified"], ascending=[True, False])

    updated_path = os.path.join(out_dir, "updated_expanded_dataset.csv")
    pred_df.to_csv(updated_path, index=False)
    print(f"\n[validate.pooled] Wrote {updated_path}: {len(pred_df)} rows")

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(out_dir, "validation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"[validate.pooled] Wrote {summary_path}: {len(summary_df)} rows")


def trajectory_substage(paths, params, hpc):
    esm_settings   = config.PARAMS["inference_settings"]
    categories     = list(config.CATEGORIES.keys())
    source_param   = params.get("embedding_source", "both")
    seed           = params.get("seed", 42)
    layers         = list(range(33))
    C              = config.PARAMS["classifier_C"]

    out_dir = paths["validate_dir"]
    os.makedirs(out_dir, exist_ok=True)

    expanded_df = pd.read_csv(os.path.join(paths["expand_dir"], "expanded_dataset.csv"))
    print(f"[validate] Loaded expanded_dataset.csv: {len(expanded_df)} rows")

    orf_df = inf.orf_level_frame(expanded_df)
    print(f"[validate] {len(orf_df)} unique ORFs (orf_sequence source)")

    source_frames = {
        "domain_sequence": (expanded_df, "domain_id", "domain_sequence"),
        "orf_sequence":    (orf_df,      "orf_id",    "orf_sequence"),
    }
    sources = ["domain_sequence", "orf_sequence"] if source_param == "both" else [source_param]

    all_rows = []

    for sequence_source in sources:
        print(f"\n{'=' * 60}")
        print(f"[validate] Source: {sequence_source}")
        print(f"{'=' * 60}")

        df, id_col, seq_col = source_frames[sequence_source]
        seq_dict  = dict(zip(df[id_col], df[seq_col]))
        all_ids   = df[id_col].tolist()
        embed_dir = os.path.join(paths["embeddings"], sequence_source)
        csv_dir   = os.path.join(embed_dir, "csv", "validate")

        inf.prepare_embeddings(
            all_ids, seq_dict, paths, esm_settings, hpc, layers,
            embed_dir=embed_dir, cache_key="validate",
        )
        cache = FeatureCache(csv_dir, all_ids, layers)

        for category in categories:
            labels = df[category].values
            vg     = df["validation_group"].values
            valid  = pd.notna(labels) & pd.notna(vg)

            if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
                print(f"  [validate] {category}: fewer than 2 labeled classes, skipping")
                continue
            labels_v, vg_v = labels[valid], vg[valid]

            for layer in layers:
                X_layer = cache.get_all(layer)
                if X_layer is None:
                    continue
                X_layer_v = X_layer[valid]

                for train_group, test_group in (("vg1", "vg2"), ("vg2", "vg1")):
                    train_mask = vg_v == train_group
                    test_mask  = vg_v == test_group

                    y_train = labels_v[train_mask]
                    y_test  = labels_v[test_mask]
                    n_classes = len(np.unique(y_train))

                    if n_classes < 2 or train_mask.sum() < n_classes + 1 or test_mask.sum() == 0:
                        continue

                    scaler, pca, X_train_pca = lda_utils.pca_fit(X_layer_v[train_mask])
                    X_test_pca = lda_utils.pca_apply(scaler, pca, X_layer_v[test_mask])

                    accuracy = _fit_and_score(X_train_pca, y_train, X_test_pca, y_test, C, seed)

                    all_rows.append({
                        "embedding_source":    sequence_source,
                        "category":            category,
                        "layer":               layer,
                        "train_group":         train_group,
                        "test_group":          test_group,
                        "n_train":             int(train_mask.sum()),
                        "n_test":              int(test_mask.sum()),
                        "n_classes":           n_classes,
                        "pca_dims_at_95":      pca.n_components_,
                        "C":                   C,
                        "accuracy":            accuracy,
                    })

    trajectory_df = pd.DataFrame(all_rows)
    # Group so each individual trajectory (one source/category/direction,
    # swept across all layers) lands in sequential rows — rows are computed
    # layer-outer/direction-inner above, which interleaves the two
    # directions layer-by-layer; this reorders for clean copy-paste.
    trajectory_df = trajectory_df.sort_values(
        ["embedding_source", "category", "train_group", "layer"]
    ).reset_index(drop=True)
    output_path = os.path.join(out_dir, "validation_trajectory.csv")
    trajectory_df.to_csv(output_path, index=False)
    print(f"\n[validate] Wrote {output_path}: {len(trajectory_df)} rows")


def run(paths, params, hpc, substages=None):
    if substages is None:
        substages = config.STAGES["validate"].get("substages", {})

    if substages.get("trajectory", False):
        print("\n[validate] Running trajectory substage...")
        trajectory_substage(paths, params["trajectory"], hpc)
        print("\n[validate] Trajectory substage complete.")

    if substages.get("pooled", False):
        print("\n[validate] Running pooled substage...")
        pooled_substage(paths, params["pooled"], hpc)
        print("\n[validate] Pooled substage complete.")
