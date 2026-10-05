# analyze.py
"""
Analyze stage: LDA/PCA trajectory analysis of literature.csv embeddings.

Reads literature.csv, embeds domain and ORF sequences with ESM2 across all
33 layers, and tracks per-layer class separation for each configured
category using PCA_dims_at_95, Bartlett's chi-square, and partial
eta-squared (all derived from Wilks' Lambda). Also assigns validation
groups. The literature-vs-expanded comparison 3D LDA visualization lives
in expand.py's visualize substage; this stage has its own separate,
self-mapped LDA visualization instead — for each configured category,
projecting literature embeddings into that category's own discriminant
space and coloring by that SAME category (using that category's matched
color palette), for both domain_sequence and orf_sequence embedding
sources.

Outputs (under paths["analyze_dir"]):
  - literature_analysis.csv           (counts block, then one row per layer)
  - updated_literature.csv            (validation_group + placeholder for later stages)
  - literature_lda_visualizations.pdf (one page per source x category x layer)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import config

from utils import inference_utils as inf
from utils import lda_utils
from utils import seq_utils
from utils.feature_cache import FeatureCache

# Short labels used in literature_analysis.csv column names
# (e.g. PCA_dims_domain_category_1) instead of the full "domain_sequence".
_SOURCE_LABEL = {"domain_sequence": "domain", "orf_sequence": "orf"}


def _assign_validation_groups(lit_df, categories, seed, stratify_category=None):
    """
    Stratified ~50/50 split of literature ORFs into two validation groups
    ('vg1'/'vg2'), stratified on whichever category has the most classes
    (auto-detected from `categories`, unless `stratify_category` overrides
    it) — the finest-grained split keeps both groups balanced across it.

    Returns a Series of 'vg1'/'vg2' aligned to lit_df's index. Every domain
    derived from a given literature ORF (via BLAST/HMMer) later inherits
    this same value, so a query and its own homologs never end up split
    across train/test — the whole point being to avoid data leakage.
    """
    from sklearn.model_selection import train_test_split

    if stratify_category is None:
        stratify_category = max(categories, key=lambda c: lit_df[c].nunique())

    labels = lit_df[stratify_category]
    idx_vg1, idx_vg2 = train_test_split(
        lit_df.index, test_size=0.5, stratify=labels, random_state=seed
    )

    groups = pd.Series(index=lit_df.index, dtype=object)
    groups.loc[idx_vg1] = "vg1"
    groups.loc[idx_vg2] = "vg2"

    print(f"[analyze] Assigned validation groups stratified on {stratify_category} "
          f"({lit_df[stratify_category].nunique()} classes): "
          f"vg1={len(idx_vg1)}, vg2={len(idx_vg2)}")
    return groups


def _counts_table(domain_df, lit_df, categories):
    """
    Human-readable counts block: one "<category>_total" row per category
    followed by one row per class value in that category (ordered by
    domain-level frequency), stepping through `categories` in order.
    orf_counts/domain_counts sit side by side on the same row — these can
    differ whenever a literature.csv row has more than one domain_bounds
    entry (domain_df has one row per domain instance, lit_df one per ORF).
    """
    rows = []
    for category in categories:
        domain_valid = domain_df[category].dropna()
        orf_valid    = lit_df[category].dropna()
        domain_vc    = domain_valid.value_counts()
        orf_vc       = orf_valid.value_counts()

        rows.append({
            "category":     f"{category}_total",
            "orf_counts":   int(len(orf_valid)),
            "domain_counts": int(len(domain_valid)),
        })

        all_labels = domain_vc.index.union(orf_vc.index)
        for label in sorted(all_labels, key=lambda l: -domain_vc.get(l, 0)):
            rows.append({
                "category":      label,
                "orf_counts":    int(orf_vc.get(label, 0)),
                "domain_counts": int(domain_vc.get(label, 0)),
            })

    return pd.DataFrame(rows)


def _layer_trajectory(cache, df, categories, layers, sequence_source):
    """
    For each layer and category: PCA_dims_at_95, then Bartlett's
    chi-square and partial eta-squared derived from the full-data Wilks'
    Lambda in that PCA-reduced space. Returns a list of trajectory rows.
    """
    rows = []

    for layer in layers:
        X_full = cache.get_all(layer)
        if X_full is None:
            continue

        for category in categories:
            labels = df[category].values
            valid = pd.notna(labels)
            X = X_full[valid]
            y = labels[valid]

            n_classes = len(np.unique(y))
            if n_classes < 2 or len(y) < n_classes + 1:
                continue

            X_pca, n_dims = lda_utils.pca_reduce_at_95(X)
            wilks_full = lda_utils.wilks_lambda(X_pca, y)
            bartlett_chi2 = lda_utils.bartlett_chi_square(wilks_full, len(y), n_dims, n_classes)
            eta_partial_sq = lda_utils.partial_eta_squared(wilks_full, n_dims, n_classes)

            rows.append({
                "sequence_source": sequence_source,
                "category":        category,
                "layer":           layer,
                "pca_dims_at_95":  n_dims,
                "bartlett_chi2":   bartlett_chi2,
                "eta_partial_sq":  eta_partial_sq,
            })

    return rows


def _build_layer_table(trajectory_rows, categories, sources):
    """
    One row per layer. Columns are grouped by source first (all domain_*
    columns, then all orf_* columns), and within each source, stepping
    through categories: PCA_dims_{source}_{category},
    Bartlett_chi2_{source}_{category}, eta_partial_sq_{source}_{category}.
    """
    if not trajectory_rows:
        return pd.DataFrame()

    traj_df = pd.DataFrame(trajectory_rows)
    layers = sorted(traj_df["layer"].unique())

    rows = []
    for layer in layers:
        row = {"layer": layer}
        for source in sources:
            label = _SOURCE_LABEL.get(source, source)
            for category in categories:
                match = traj_df[
                    (traj_df["layer"] == layer)
                    & (traj_df["category"] == category)
                    & (traj_df["sequence_source"] == source)
                ]
                suffix = f"{label}_{category}"
                if len(match):
                    m = match.iloc[0]
                    row[f"PCA_dims_{suffix}"] = m["pca_dims_at_95"]
                    row[f"Bartlett_chi2_{suffix}"] = m["bartlett_chi2"]
                    row[f"eta_partial_sq_{suffix}"] = m["eta_partial_sq"]
                else:
                    row[f"PCA_dims_{suffix}"] = None
                    row[f"Bartlett_chi2_{suffix}"] = None
                    row[f"eta_partial_sq_{suffix}"] = None
        rows.append(row)

    return pd.DataFrame(rows)


def _lda_visualize(paths, params, hpc, lit_df, domain_df):
    """
    For each embedding source (domain_sequence, orf_sequence) and each
    configured category spec, fits an LDA discriminant projection on
    literature embeddings for that category, then plots those same points
    colored by that SAME category (self-mapped — how well does this
    category separate in its own discriminant space), using that
    category's own matched color palette. One PDF page per
    (embedding_source, category, layer) — a single-population scatter, no
    second/expanded population here unlike expand.py's visualize substage.
    """
    esm_settings   = config.PARAMS["inference_settings"]
    layers         = params.get("layers", [2, 12, 22, 32])
    source_param   = params.get("embedding_source", "both")
    category_specs = params.get("categories", [])

    sources = ["domain_sequence", "orf_sequence"] if source_param == "both" else [source_param]

    # domain_sequence is keyed by domain_id (domain_df, one row per
    # domain_bounds entry); orf_sequence is keyed by orf_id directly off
    # lit_df (one row per ORF) — same convention as run()'s main
    # trajectory loop, so an ORF's sequence is only ever embedded once.
    source_frames = {
        "domain_sequence": (domain_df, "domain_id", "domain_sequence"),
        "orf_sequence":    (lit_df,    "orf_id",    "orf_sequence"),
    }

    pdf_path = os.path.join(paths["analyze_dir"], "literature_lda_visualizations.pdf")
    with PdfPages(pdf_path) as pdf:
        for sequence_source in sources:
            df, id_col, seq_col = source_frames[sequence_source]
            source_label = _SOURCE_LABEL.get(sequence_source, sequence_source)

            embed_dir = os.path.join(paths["embeddings"], sequence_source)
            csv_dir   = os.path.join(embed_dir, "csv", "analyze")
            all_ids   = df[id_col].tolist()
            seq_dict  = dict(zip(df[id_col], df[seq_col]))
            inf.prepare_embeddings(
                all_ids, seq_dict, paths, esm_settings, hpc, layers,
                embed_dir=embed_dir, cache_key="analyze",
            )
            cache = FeatureCache(csv_dir, all_ids, layers)

            for spec in category_specs:
                category      = spec["category"]
                color_palette = spec.get("color_palette", {})
                cat_name = config.CATEGORIES.get(category, {}).get("name", category)

                labels = df[category].values
                valid = pd.notna(labels)
                if valid.sum() == 0 or len(np.unique(labels[valid])) < 2:
                    print(f"  [analyze.lda] {sequence_source}/{category}: "
                          f"fewer than 2 labeled classes, skipping")
                    continue

                for layer in layers:
                    X = cache.get_all(layer)
                    if X is None:
                        print(f"  [analyze.lda] Skipping {sequence_source}/{category} "
                              f"layer {layer}: missing cached embeddings")
                        continue

                    X_valid, y_valid = X[valid], labels[valid]

                    scaler, pca, X_pca = lda_utils.pca_fit(X_valid)
                    X_lda, _ = lda_utils.fit_lda_projection(X_pca, y_valid, n_components=3)

                    if X_lda.shape[1] < 2:
                        print(f"  [analyze.lda] Skipping {sequence_source}/{category} "
                              f"layer {layer}: only {X_lda.shape[1]} LD axis available")
                        continue

                    title = f"{cat_name} LDA space ({source_label}) — layer {layer}"
                    lda_utils.plot_lda_3d(pdf, X_lda, y_valid, color_palette, title)
                    print(f"  [analyze.lda] Added PDF page: {title}")

    print(f"[analyze.lda] Wrote {pdf_path}")


def run(paths, params, hpc):
    """Analyze stage entry point."""
    esm_settings       = config.PARAMS["inference_settings"]
    categories         = params.get("categories", list(config.CATEGORIES.keys()))
    embedding_source   = params.get("embedding_source", "both")
    layers             = params.get("layers", esm_settings["repr_layers"])
    vg_stratify_category = params.get("validation_group_category")  # None = auto-detect
    vg_seed            = params.get("validation_group_seed", 42)

    out_dir = paths["analyze_dir"]
    os.makedirs(out_dir, exist_ok=True)

    lit_df = pd.read_csv(paths["literature"])
    print(f"[analyze] Loaded literature.csv: {len(lit_df)} rows")

    lit_df["validation_group"] = _assign_validation_groups(lit_df, categories, vg_seed, vg_stratify_category)

    domain_df = seq_utils.literature_domains_with_categories(lit_df, categories)
    print(f"[analyze] Built {len(domain_df)} domain rows from literature.csv")

    counts_df = _counts_table(domain_df, lit_df, categories)

    # domain_sequence is keyed by domain_id (domain_df, one row per
    # domain_bounds entry); orf_sequence is keyed by orf_id directly off
    # lit_df (one row per ORF) — NOT domain_df, so an ORF's sequence is
    # only ever embedded once even if it has multiple domain_bounds.
    source_frames = {
        "domain_sequence": (domain_df, "domain_id", "domain_sequence"),
        "orf_sequence":    (lit_df,    "orf_id",    "orf_sequence"),
    }

    sources = ["domain_sequence", "orf_sequence"] if embedding_source == "both" else [embedding_source]

    all_trajectory_rows = []

    for sequence_source in sources:
        print(f"\n{'=' * 60}")
        print(f"[analyze] Source: {sequence_source}")
        print(f"{'=' * 60}")

        df, id_col, seq_col = source_frames[sequence_source]

        seq_dict  = dict(zip(df[id_col], df[seq_col]))
        all_ids   = df[id_col].tolist()
        embed_dir = os.path.join(paths["embeddings"], sequence_source)
        csv_dir   = os.path.join(embed_dir, "csv", "analyze")

        inf.prepare_embeddings(
            all_ids, seq_dict, paths,
            esm_settings, hpc, layers,
            embed_dir=embed_dir,
            cache_key="analyze",
        )

        cache = FeatureCache(csv_dir, all_ids, layers)

        trajectory_rows = _layer_trajectory(cache, df, categories, layers, sequence_source)
        all_trajectory_rows.extend(trajectory_rows)

    layer_df = _build_layer_table(all_trajectory_rows, categories, sources)

    # counts block first, then one row per layer — concat with sort=False
    # keeps column order (category, orf_counts, domain_counts, then layer,
    # PCA_dims_*/Wilks_lambda_*/flag_*), filling the other block's columns
    # with NaN on each side.
    analysis_df   = pd.concat([counts_df, layer_df], ignore_index=True, sort=False)
    analysis_path = os.path.join(out_dir, "literature_analysis.csv")
    analysis_df.to_csv(analysis_path, index=False)
    print(f"\n[analyze] Wrote {analysis_path}")

    updated_literature = lit_df.copy()
    updated_literature["analyze_stage_complete"] = True
    updated_path = os.path.join(out_dir, "updated_literature.csv")
    updated_literature.to_csv(updated_path, index=False)
    print(f"[analyze] Wrote {updated_path}")

    lda_params = params.get("lda_visualize", {})
    if lda_params.get("run", True):
        _lda_visualize(paths, lda_params, hpc, lit_df, domain_df)
