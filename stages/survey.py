# survey.py
"""
Survey stage.

Genome-wide HMM scan + classification — unlike every other stage, this
does NOT touch the curated literature/BLAST-expanded universe as a
candidate set. hmm_scan scans whole external proteomes (config-listed)
for AAA-domain-like hits using the same HMM profile expand.search already
built, classifying HMM_signal (strong/weak — both kept, unlike
expand.search's hard filter) and applying keyword filters (e.g.
is_abc_transporter, a heuristic for likely false-positive hits sharing
the Walker A motif with genuine AAA+ ATPases).

predict trains the same fixed-C classifiers as predict.py on the ENTIRE
expanded_dataset.csv (used purely as training data here) and classifies
every hit at both domain- and ORF-level, adds hierarchy-violation flags
(does a row's predicted child-category label belong under its own
predicted parent-category label, per a parent/child mapping inferred
empirically from the training data's true labels — not a hardcoded
dict), then writes keyword-split hit/label-distribution CSVs, a
hierarchy violation-rate summary, and an entropy-vs-hierarchy-status
comparison for a config-set subset of proteomes.
"""
import os
import numpy as np
import pandas as pd
import config

from utils import inference_utils as inf
from utils import io_utils, seq_utils, subprocesses
from utils.feature_cache import FeatureCache

_SOURCE_LABEL = {"domain_sequence": "domain", "orf_sequence": "orf"}


def _apply_keyword_filters(df, keyword_filters):
    """OR-across-keywords boolean column is_{name} per configured filter,
    matched case-insensitively against protein_description."""
    for kf in keyword_filters:
        col = f"is_{kf['name']}"
        df[col] = False
        for kw in kf["keywords"]:
            df[col] = df[col] | df["protein_description"].str.contains(kw, case=False, na=False)
    return df


def _add_parsed_header_columns(df):
    """
    Parse UniProt-format protein_id (sp|ACCESSION|ENTRY_NAME) into
    accession/common_name, and protein_description
    (... OS=Species OX=... GN=... PE=... SV=...) into species.
    """
    if "protein_id" in df.columns:
        parts = df["protein_id"].str.split("|", expand=True)
        df["accession"]   = parts[1] if parts.shape[1] > 1 else df["protein_id"]
        df["common_name"] = parts[2] if parts.shape[1] > 2 else ""
    if "protein_description" in df.columns:
        df["species"] = df["protein_description"].str.extract(
            r"OS=(.+?)(?:\s+OX=|\s+GN=|\s+PE=|\s+SV=|$)"
        )
    return df


def _sort_survey_hits(df, categories, proteome_order=None):
    """
    Order rows: proteome (priority order first) -> keyword-match bucket
    (non-matches first) -> mechanism (category_1) prediction ->
    HMM_signal (strong before weak) -> protein_id -> domain_identifier.
    """
    if df.empty:
        return df

    sort_df = df.copy()
    sort_cols, ascending = [], []

    if "proteome_id" in sort_df.columns:
        if proteome_order:
            order_map = {pid: i for i, pid in enumerate(proteome_order)}
            sort_df["_prot_order"] = sort_df["proteome_id"].map(order_map).fillna(len(proteome_order))
            sort_cols.append("_prot_order")
        else:
            sort_cols.append("proteome_id")
        ascending.append(True)

    keyword_cols = [c for c in sort_df.columns if c.startswith("is_")]
    if keyword_cols:
        sort_df["_keyword_match"] = sort_df[keyword_cols].any(axis=1)
        sort_cols.append("_keyword_match")
        ascending.append(True)

    pred_cols = [f"{cat}_pred_domain" for cat in categories]
    mechanism_col = ("category_1_pred_domain" if "category_1_pred_domain" in sort_df.columns
                      else next((c for c in pred_cols if c in sort_df.columns), None))
    if mechanism_col is not None:
        sort_cols.append(mechanism_col)
        ascending.append(True)

    if "HMM_signal" in sort_df.columns:
        sort_df["_signal_order"] = sort_df["HMM_signal"].map({"strong": 0, "weak": 1})
        sort_cols.append("_signal_order")
        ascending.append(True)

    for col in ("protein_id", "domain_identifier"):
        if col in sort_df.columns:
            sort_cols.append(col)
            ascending.append(True)

    sort_df = sort_df.sort_values(sort_cols, ascending=ascending, na_position="last")
    drop_cols = [c for c in ("_prot_order", "_keyword_match", "_signal_order") if c in sort_df.columns]
    return sort_df.drop(columns=drop_cols).reset_index(drop=True)


def hmm_scan(paths, params, hpc):
    """
    params is the FULL PARAMS["survey"] dict (not scoped to just
    hmm_scan) since proteomes/keyword_filters are shared with predict.
    """
    proteomes       = params.get("proteomes", [])
    keyword_filters = params.get("keyword_filters", [])
    scan_params     = params.get("hmm_scan", {})
    filter_params   = scan_params.get("filter_params", {})
    hmm_settings    = scan_params.get("hmm_settings", {})

    active = [p for p in proteomes if p.get("active", False)]
    if not active:
        print("[survey.hmm_scan] No active proteomes configured")
        return

    hmm_profile = subprocesses.build_hmm_profile(paths["msa"], paths["expand_hmm_dir"], hpc)

    output_dir = paths["survey_hits_dir"]
    os.makedirs(output_dir, exist_ok=True)

    for proteome_cfg in active:
        prot_id, prot_fasta = proteome_cfg["id"], proteome_cfg["fasta"]
        stage_dir = os.path.join(output_dir, prot_id)
        os.makedirs(stage_dir, exist_ok=True)

        print(f"\n{'=' * 60}")
        print(f"[survey.hmm_scan] Scanning: {prot_id}")
        print(f"  Proteome: {os.path.basename(prot_fasta)}")
        print(f"{'=' * 60}")

        fasta_df = io_utils.read_fasta(prot_fasta, truncate_header=True)
        print(f"  Proteins in proteome: {len(fasta_df)}")

        success = subprocesses.run_hmmer(prot_id, stage_dir, prot_fasta, hmm_profile, hmm_settings, hpc)
        if not success:
            print(f"  No HMM hits for {prot_id}")
            continue

        hmm_df = io_utils.read_hmmsearch_domtblout(os.path.join(stage_dir, f"{prot_id}.domtblout"))
        fasta_df["protein_id"]          = fasta_df["header"].str.split().str[0]
        fasta_df["protein_description"] = fasta_df["header_info"].fillna("")
        print(f"  HMM hits: {len(hmm_df)}")

        hmm_df["c-value"] = pd.to_numeric(hmm_df["c-value"], errors="coerce")
        hmm_df = seq_utils.remove_overlapping_domains(hmm_df)
        print(f"  After overlap removal: {len(hmm_df)}")
        if len(hmm_df) == 0:
            print(f"  No domains after overlap removal for {prot_id}")
            continue

        domain_df = pd.merge(hmm_df, fasta_df, left_on="target_name", right_on="protein_id", how="inner")

        domain_df["boundaries"]        = domain_df["env_from"].astype(str) + "-" + domain_df["env_to"].astype(str)
        domain_df["domain_identifier"] = domain_df["target_name"] + "_" + domain_df["boundaries"]
        domain_df["domain_sequence"]   = list(
            map(seq_utils.extract_domain, domain_df["sequence"], domain_df["boundaries"])
        )

        domain_df["domain_length"] = domain_df["domain_sequence"].str.len()
        domain_df["walkerA_score"] = domain_df["domain_sequence"].apply(seq_utils.score_walkerA)

        domain_df["HMM_signal"] = "weak"
        passes = (
            (domain_df["c-value"] < filter_params["c_value_max"]) &
            (domain_df["domain_length"] >= filter_params["length_min"]) &
            (domain_df["domain_length"] <= filter_params["length_max"]) &
            (domain_df["walkerA_score"] >= filter_params["walker_a_min"])
        )
        domain_df.loc[passes, "HMM_signal"] = "strong"
        n_strong = int((domain_df["HMM_signal"] == "strong").sum())
        n_weak   = int((domain_df["HMM_signal"] == "weak").sum())
        print(f"  Signal classification: {n_strong} strong, {n_weak} weak")

        _apply_keyword_filters(domain_df, keyword_filters)
        for kf in keyword_filters:
            col = f"is_{kf['name']}"
            print(f"  Keyword '{kf['name']}' matches: {int(domain_df[col].sum())}")

        domain_df["proteome_id"] = prot_id

        # sort: strong first, weak second, keyword matches last
        domain_df["sort_key"] = domain_df["HMM_signal"].map({"strong": 0, "weak": 1})
        for kf in keyword_filters:
            col = f"is_{kf['name']}"
            domain_df.loc[domain_df[col], "sort_key"] = domain_df.loc[domain_df[col], "sort_key"].clip(lower=2)
        domain_df = domain_df.sort_values("sort_key").drop(columns=["sort_key"])

        output_path = os.path.join(output_dir, f"{prot_id}_survey_hits.csv")
        domain_df.to_csv(output_path, index=False)
        print(f"  Written: {output_path} ({len(domain_df)} domains)")


def _empirical_parent_map(train_df, parent_col, child_col):
    """Majority-vote parent label observed for each child label in the
    training data's TRUE labels — used as the hierarchy ground truth
    instead of porting the old hardcoded CATEGORY_HIERARCHY dict."""
    valid = train_df[[parent_col, child_col]].dropna()
    return valid.groupby(child_col)[parent_col].agg(lambda s: s.mode().iloc[0]).to_dict()


def _label_distribution(df, categories, sources):
    """One row per (category, source, proteome_id, label) with predicted-
    label counts, plus a proteome_id="combined" row per (category,
    source, label) pooled across all proteomes."""
    rows = []
    for category in categories:
        for source_label in sources:
            pred_col = f"{category}_pred_{source_label}"
            if pred_col not in df.columns:
                continue
            for proteome_id, grp in df.groupby("proteome_id"):
                for label, count in grp[pred_col].value_counts().items():
                    rows.append({"category": category, "source": source_label,
                                 "proteome_id": proteome_id, "label": label, "count": int(count)})
            for label, count in df[pred_col].value_counts().items():
                rows.append({"category": category, "source": source_label,
                             "proteome_id": "combined", "label": label, "count": int(count)})
    return pd.DataFrame(rows)


def _add_hierarchy_ok_columns(df, edges, sources, parent_map_source_df):
    """
    Adds {child_cat}_hier_ok_{source} (per-edge) and hier_ok_{source}
    (combined -- False if EITHER edge was violated, True only if both are
    defined and both pass, NaN if either is unscorable; matches the
    predecessor pipeline's check_hierarchy_violations(), domain_inference/
    utils/inference_utils.py:895-954) columns to df, using
    empirically-derived parent maps built from parent_map_source_df's TRUE
    category labels. Mutates and returns df. Reused for both the survey
    hit pool (hierarchy_substage) and the held-out validation experiment
    (updated_expanded_dataset.csv), so the two are scored identically.
    """
    for child_cat, parent_cat in edges:
        parent_map = _empirical_parent_map(parent_map_source_df, parent_cat, child_cat)
        for source_label in sources:
            child_pred_col  = f"{child_cat}_pred_{source_label}"
            parent_pred_col = f"{parent_cat}_pred_{source_label}"
            hier_col        = f"{child_cat}_hier_ok_{source_label}"
            if child_pred_col not in df.columns or parent_pred_col not in df.columns:
                continue
            expected_parent = df[child_pred_col].map(parent_map)
            df[hier_col] = (expected_parent == df[parent_pred_col]).astype("boolean")
            unscorable = df[child_pred_col].isna() | df[parent_pred_col].isna()
            df.loc[unscorable, hier_col] = pd.NA

    for source_label in sources:
        c1 = df.get(f"category_1_hier_ok_{source_label}")
        c2 = df.get(f"category_2_hier_ok_{source_label}")
        if c1 is None or c2 is None:
            continue
        both_defined = c1.notna() & c2.notna()
        combined = pd.Series(pd.NA, index=df.index, dtype="boolean")
        combined[both_defined] = c1[both_defined].astype(bool) & c2[both_defined].astype(bool)
        df[f"hier_ok_{source_label}"] = combined
    return df


def _hierarchy_violation_row(df, edges, sources, proteome_id, keyword_state, orf_id_col=None):
    """
    orf_id_col: column identifying the parent ORF/protein (e.g. "accession"
    for the survey hit pool, "orf_id" for updated_expanded_dataset.csv).
    df is domain-hit/domain-row grain, so *_hier_ok_orf / hier_ok_orf are
    per-ORF values broadcast onto every row sharing that ORF (a multi-
    domain protein has one row per domain, all carrying the same orf-level
    value) -- computing orf-source rates/n_scorable directly on df would
    silently weight multi-domain proteins by how many domains they have.
    Dedupe to one row per orf_id_col before scoring the "orf" source only;
    "domain" source rows are already one-per-domain, no dedup needed.
    """
    row = {"proteome_id": proteome_id, "keyword_state": keyword_state}
    orf_df = df.drop_duplicates(subset=orf_id_col) if orf_id_col and orf_id_col in df.columns else df

    def _source_df(source_label):
        return orf_df if source_label == "orf" else df

    for child_cat, _parent_cat in edges:
        for source_label in sources:
            sdf = _source_df(source_label)
            hier_col = f"{child_cat}_hier_ok_{source_label}"
            if hier_col not in sdf.columns:
                continue
            scorable = sdf[hier_col].notna()
            n_scorable = int(scorable.sum())
            viol_rate = float((~sdf.loc[scorable, hier_col]).mean()) if n_scorable else float("nan")
            row[f"{child_cat}_viol_rate_{source_label}"] = viol_rate
            row[f"{child_cat}_n_scorable_{source_label}"] = n_scorable
    # Combined (any-level) rate -- only present when the caller has already
    # computed hier_ok_{source} (currently just hierarchy_substage; absent
    # for predict_substage's domain-only summary, silently skipped here).
    for source_label in sources:
        sdf = _source_df(source_label)
        combined_col = f"hier_ok_{source_label}"
        if combined_col not in sdf.columns:
            continue
        scorable = sdf[combined_col].notna()
        n_scorable = int(scorable.sum())
        viol_rate = float((~sdf.loc[scorable, combined_col]).mean()) if n_scorable else float("nan")
        row[f"combined_viol_rate_{source_label}"] = viol_rate
        row[f"combined_n_scorable_{source_label}"] = n_scorable
    return row


def predict_substage(paths, params, hpc):
    """
    params is the FULL PARAMS["survey"] dict. Domain-level classification
    ONLY, across every `active` proteome -- no ORF-level inference, no
    *_pred_orf/*_prob_*_orf/*_entropy_orf/*_hier_ok_orf columns anywhere
    in this substage's output. ORF-level classification (restricted to
    PARAMS["survey"]["predict"]["hierarchy_proteomes"]) lives in
    hierarchy_substage below, as its own separate output files.
    """
    categories      = list(config.CATEGORIES.keys())
    proteomes       = params.get("proteomes", [])
    predict_params  = params.get("predict", {})
    layer_range     = predict_params.get("layer_range", [11, 19])
    seed            = predict_params.get("seed", 42)
    priority_proteomes = predict_params.get("priority_proteomes", ["h_sapiens", "e_coli"])
    layers = list(range(layer_range[0], layer_range[1] + 1))
    C = config.PARAMS["classifier_C"]
    esm_settings = config.PARAMS["inference_settings"]

    out_dir = paths["survey_dir"]
    os.makedirs(out_dir, exist_ok=True)

    expanded_df = pd.read_csv(os.path.join(paths["expand_dir"], "expanded_dataset.csv"))
    print(f"[survey.predict] Loaded expanded_dataset.csv: {len(expanded_df)} rows (training data)")

    active_ids = [p["id"] for p in proteomes if p.get("active", False)]
    hit_frames = []
    for prot_id in active_ids:
        hit_path = os.path.join(paths["survey_hits_dir"], f"{prot_id}_survey_hits.csv")
        if not os.path.exists(hit_path):
            print(f"  [survey.predict] {hit_path} not found, skipping {prot_id}")
            continue
        hit_frames.append(pd.read_csv(hit_path))
    if not hit_frames:
        print("[survey.predict] No survey hit CSVs found — run hmm_scan first")
        return
    hits_df = pd.concat(hit_frames, ignore_index=True)
    hits_df = _add_parsed_header_columns(hits_df)
    print(f"[survey.predict] Loaded {len(hits_df)} hits across {len(hit_frames)} proteomes; "
          f"pooling layers {layer_range[0]}-{layer_range[1]}, C={C}")

    sources_present = ["domain"]
    embed_dir = os.path.join(paths["embeddings"], "domain_sequence")

    pred_df = hits_df.copy()

    print(f"\n{'=' * 60}")
    print("[survey.predict] Source: domain_sequence")
    print(f"{'=' * 60}")

    train_ids      = expanded_df["domain_id"].tolist()
    train_seq_dict = dict(zip(expanded_df["domain_id"], expanded_df["domain_sequence"]))
    inf.prepare_embeddings(
        train_ids, train_seq_dict, paths, esm_settings, hpc, layers,
        embed_dir=embed_dir, cache_key="validate",
    )
    train_cache = FeatureCache(os.path.join(embed_dir, "csv", "validate"), train_ids, layers)

    hit_ids      = hits_df["domain_identifier"].tolist()
    hit_seq_dict = dict(zip(hits_df["domain_identifier"], hits_df["domain_sequence"]))
    inf.prepare_embeddings(
        hit_ids, hit_seq_dict, paths, esm_settings, hpc, layers,
        embed_dir=embed_dir, cache_key="survey",
    )
    hit_cache = FeatureCache(os.path.join(embed_dir, "csv", "survey"), hit_ids, layers)

    missing_train = [l for l in layers if train_cache.get_all(l) is None]
    missing_hit   = [l for l in layers if hit_cache.get_all(l) is None]
    if missing_train or missing_hit:
        print(f"  [survey.predict] missing cached layers "
              f"(train={missing_train}, hits={missing_hit}), aborting")
        return

    X_train_pooled = np.concatenate([train_cache.get_all(l) for l in layers], axis=1)
    X_hit_pooled   = np.concatenate([hit_cache.get_all(l) for l in layers], axis=1)

    for category in categories:
        labels_full = expanded_df[category].values
        valid = pd.notna(labels_full)
        if valid.sum() == 0 or len(np.unique(labels_full[valid])) < 2:
            print(f"  [survey.predict] {category}: fewer than 2 labeled classes, skipping")
            continue

        X_train_v = X_train_pooled[valid]
        y_train_v = labels_full[valid]

        print(f"  [survey.predict] {category}: n_train={len(y_train_v)}, n_hits={len(hit_ids)}")

        result = inf.fit_and_predict(X_train_v, y_train_v, X_hit_pooled, C, seed)

        pred_col    = f"{category}_pred_domain"
        entropy_col = f"{category}_entropy_domain"
        pred_df[pred_col] = None
        pred_df[entropy_col] = np.nan
        for cls in sorted(np.unique(y_train_v)):
            pred_df[f"{category}_prob_{cls}_domain"] = np.nan

        id_to_row = {hid: i for i, hid in enumerate(hit_ids)}
        mask_in_pred = pred_df["domain_identifier"].isin(hit_ids)
        for idx in pred_df.index[mask_in_pred]:
            i = id_to_row[pred_df.at[idx, "domain_identifier"]]
            pred_df.at[idx, pred_col] = result["preds"][i]
            pred_df.at[idx, entropy_col] = result["entropies"][i]
            for ci, cls in enumerate(result["classes"]):
                pred_df.at[idx, f"{category}_prob_{cls}_domain"] = result["probs"][i, ci]

    # Hierarchy check (domain-source only, since sources_present is just
    # ["domain"] here): empirically infer parent-label-per-child-label
    # maps from the training data's TRUE labels, then flag whether each
    # hit row's predicted child label is consistent with its own
    # predicted parent label.
    edges = [(cat, config.CATEGORIES[cat]["parent"]) for cat in categories if config.CATEGORIES[cat]["parent"]]
    for child_cat, parent_cat in edges:
        parent_map = _empirical_parent_map(expanded_df, parent_cat, child_cat)
        for source_label in sources_present:
            child_pred_col  = f"{child_cat}_pred_{source_label}"
            parent_pred_col = f"{parent_cat}_pred_{source_label}"
            hier_col        = f"{child_cat}_hier_ok_{source_label}"
            if child_pred_col not in pred_df.columns or parent_pred_col not in pred_df.columns:
                continue
            expected_parent = pred_df[child_pred_col].map(parent_map)
            # Nullable "boolean" dtype (not plain bool) — assigning NaN
            # into a plain-bool column silently upcasts it to object
            # dtype, which breaks `~` (bitwise-complements True/False to
            # -2/-1 instead of negating) and corrupts every violation
            # rate computed from it downstream.
            pred_df[hier_col] = (expected_parent == pred_df[parent_pred_col]).astype("boolean")
            unscorable = pred_df[child_pred_col].isna() | pred_df[parent_pred_col].isna()
            pred_df.loc[unscorable, hier_col] = pd.NA

    pred_df = _sort_survey_hits(pred_df, categories, proteome_order=priority_proteomes)

    # Keyword split -> separate CSVs (hits + label distributions).
    keyword_cols = [c for c in pred_df.columns if c.startswith("is_")]
    keyword_mask = pred_df[keyword_cols].any(axis=1) if keyword_cols else pd.Series(False, index=pred_df.index)
    false_df = pred_df.loc[~keyword_mask].reset_index(drop=True)
    true_df  = pred_df.loc[keyword_mask].reset_index(drop=True)

    false_path = os.path.join(out_dir, "survey_hits_keyword_false.csv")
    true_path  = os.path.join(out_dir, "survey_hits_keyword_true.csv")
    false_df.to_csv(false_path, index=False)
    true_df.to_csv(true_path, index=False)
    print(f"\n[survey.predict] Wrote {false_path}: {len(false_df)} rows")
    print(f"[survey.predict] Wrote {true_path}: {len(true_df)} rows")

    label_dist_false = _label_distribution(false_df, categories, sources_present)
    label_dist_true  = _label_distribution(true_df, categories, sources_present)
    label_dist_false.to_csv(os.path.join(out_dir, "label_distribution_keyword_false.csv"), index=False)
    label_dist_true.to_csv(os.path.join(out_dir, "label_distribution_keyword_true.csv"), index=False)
    print(f"[survey.predict] Wrote label_distribution_keyword_false.csv: {len(label_dist_false)} rows")
    print(f"[survey.predict] Wrote label_distribution_keyword_true.csv: {len(label_dist_true)} rows")

    # FASTA export for priority proteomes, keyword=False rows only.
    for prot_id in priority_proteomes:
        prot_hits = false_df[false_df["proteome_id"] == prot_id]
        if prot_hits.empty:
            print(f"  [survey.predict] No keyword=False hits for {prot_id}, skipping FASTA export")
            continue
        dom_df = prot_hits.drop_duplicates(subset=["domain_identifier"])
        io_utils.write_fasta(os.path.join(out_dir, f"{prot_id}_domains.fasta"),
                              dom_df["domain_identifier"], dom_df["domain_sequence"])
        orf_df = prot_hits.drop_duplicates(subset=["accession"])
        io_utils.write_fasta(os.path.join(out_dir, f"{prot_id}_orfs.fasta"),
                              orf_df["accession"], orf_df["sequence"])
        print(f"  [survey.predict] Wrote {prot_id}_domains.fasta ({len(dom_df)}), "
              f"{prot_id}_orfs.fasta ({len(orf_df)})")

    # Hierarchy violation-rate summary: by proteome x keyword_state x edge x source.
    summary_rows = []
    for proteome_id in pred_df["proteome_id"].unique():
        for keyword_state, subset in (("false", false_df), ("true", true_df), ("all", pred_df)):
            sub = subset[subset["proteome_id"] == proteome_id]
            if sub.empty:
                continue
            summary_rows.append(_hierarchy_violation_row(sub, edges, sources_present, proteome_id, keyword_state))
    hier_summary_df = pd.DataFrame(summary_rows)
    hier_summary_path = os.path.join(out_dir, "hierarchy_violation_summary.csv")
    hier_summary_df.to_csv(hier_summary_path, index=False)
    print(f"[survey.predict] Wrote {hier_summary_path}: {len(hier_summary_df)} rows")


def hierarchy_substage(paths, params, hpc):
    """
    params is the FULL PARAMS["survey"] dict. Domain+ORF-level
    classification restricted to PARAMS["survey"]["predict"]
    ["hierarchy_proteomes"] (human/E. coli/S. pombe by default) --
    produces the domain-vs-parent and orf-vs-parent hierarchy-violation
    comparison, plus the entropy-vs-hierarchy-status comparison, as
    hierarchy_analysis_summary.csv / entropy_by_hierarchy_status.csv.

    Self-contained: reads the raw per-proteome hit CSVs directly (same as
    predict_substage does) rather than depending on predict_substage's
    output file, matching the existing pattern of substages independently
    loading what they need (e.g. validate.py's trajectory/pooled
    substages). Uses its own cache_key ("hierarchy", not "survey") for
    both sources' hit embeddings so its narrower 3-proteome hit pool never
    collides with predict_substage's separate 10-proteome domain-only
    "survey" cache -- training-side embeddings still reuse cache_key
    "validate" for both sources, since those were already corrected by
    this session's ESM window-truncation fix.
    """
    categories      = list(config.CATEGORIES.keys())
    predict_params  = params.get("predict", {})
    layer_range     = predict_params.get("layer_range", [11, 19])
    seed            = predict_params.get("seed", 42)
    hierarchy_proteomes = predict_params.get("hierarchy_proteomes", ["h_sapiens", "e_coli", "s_pombe"])
    layers = list(range(layer_range[0], layer_range[1] + 1))
    C = config.PARAMS["classifier_C"]
    esm_settings = config.PARAMS["inference_settings"]

    out_dir = paths["survey_dir"]
    os.makedirs(out_dir, exist_ok=True)

    expanded_df  = pd.read_csv(os.path.join(paths["expand_dir"], "expanded_dataset.csv"))
    train_orf_df = inf.orf_level_frame(expanded_df)
    print(f"[survey.hierarchy] Loaded expanded_dataset.csv: {len(expanded_df)} rows, "
          f"{len(train_orf_df)} unique ORFs (training data)")

    hit_frames = []
    for prot_id in hierarchy_proteomes:
        hit_path = os.path.join(paths["survey_hits_dir"], f"{prot_id}_survey_hits.csv")
        if not os.path.exists(hit_path):
            print(f"  [survey.hierarchy] {hit_path} not found, skipping {prot_id}")
            continue
        hit_frames.append(pd.read_csv(hit_path))
    if not hit_frames:
        print("[survey.hierarchy] No survey hit CSVs found for hierarchy_proteomes — run hmm_scan first")
        return
    hits_df = pd.concat(hit_frames, ignore_index=True)
    hits_df = _add_parsed_header_columns(hits_df)
    print(f"[survey.hierarchy] Loaded {len(hits_df)} hits across {len(hit_frames)} proteomes "
          f"(scope: {hierarchy_proteomes}); pooling layers {layer_range[0]}-{layer_range[1]}, C={C}")

    hits_orf_df = hits_df.drop_duplicates(subset="accession").reset_index(drop=True)

    train_frames = {
        "domain_sequence": (expanded_df,  "domain_id", "domain_sequence"),
        "orf_sequence":    (train_orf_df, "orf_id",    "orf_sequence"),
    }
    hit_frames_by_source = {
        "domain_sequence": (hits_df,     "domain_identifier", "domain_sequence"),
        "orf_sequence":    (hits_orf_df, "accession",          "sequence"),
    }
    sources = ["domain_sequence", "orf_sequence"]
    sources_present = [_SOURCE_LABEL[s] for s in sources]

    pred_df = hits_df.copy()

    for sequence_source in sources:
        print(f"\n{'=' * 60}")
        print(f"[survey.hierarchy] Source: {sequence_source}")
        print(f"{'=' * 60}")

        train_df, train_id_col, train_seq_col = train_frames[sequence_source]
        hit_df, hit_id_col, hit_seq_col       = hit_frames_by_source[sequence_source]
        source_label = _SOURCE_LABEL[sequence_source]
        embed_dir    = os.path.join(paths["embeddings"], sequence_source)

        train_ids      = train_df[train_id_col].tolist()
        train_seq_dict = dict(zip(train_df[train_id_col], train_df[train_seq_col]))
        inf.prepare_embeddings(
            train_ids, train_seq_dict, paths, esm_settings, hpc, layers,
            embed_dir=embed_dir, cache_key="validate",
        )
        train_cache = FeatureCache(os.path.join(embed_dir, "csv", "validate"), train_ids, layers)

        hit_ids      = hit_df[hit_id_col].tolist()
        hit_seq_dict = dict(zip(hit_df[hit_id_col], hit_df[hit_seq_col]))
        inf.prepare_embeddings(
            hit_ids, hit_seq_dict, paths, esm_settings, hpc, layers,
            embed_dir=embed_dir, cache_key="hierarchy",
        )
        hit_cache = FeatureCache(os.path.join(embed_dir, "csv", "hierarchy"), hit_ids, layers)

        missing_train = [l for l in layers if train_cache.get_all(l) is None]
        missing_hit   = [l for l in layers if hit_cache.get_all(l) is None]
        if missing_train or missing_hit:
            print(f"  [survey.hierarchy] Source {sequence_source}: missing cached layers "
                  f"(train={missing_train}, hits={missing_hit}), skipping source")
            continue

        X_train_pooled = np.concatenate([train_cache.get_all(l) for l in layers], axis=1)
        X_hit_pooled   = np.concatenate([hit_cache.get_all(l) for l in layers], axis=1)

        for category in categories:
            labels_full = train_df[category].values
            valid = pd.notna(labels_full)
            if valid.sum() == 0 or len(np.unique(labels_full[valid])) < 2:
                print(f"  [survey.hierarchy] {category}: fewer than 2 labeled classes, skipping")
                continue

            X_train_v = X_train_pooled[valid]
            y_train_v = labels_full[valid]

            print(f"  [survey.hierarchy] {category}: n_train={len(y_train_v)}, n_hits={len(hit_ids)}")

            result = inf.fit_and_predict(X_train_v, y_train_v, X_hit_pooled, C, seed)

            pred_col    = f"{category}_pred_{source_label}"
            entropy_col = f"{category}_entropy_{source_label}"
            if pred_col not in pred_df.columns:
                pred_df[pred_col] = None
                pred_df[entropy_col] = np.nan
                for cls in sorted(np.unique(y_train_v)):
                    pred_df[f"{category}_prob_{cls}_{source_label}"] = np.nan

            id_to_row = {hid: i for i, hid in enumerate(hit_ids)}
            mask_in_pred = pred_df[hit_id_col].isin(hit_ids)
            for idx in pred_df.index[mask_in_pred]:
                i = id_to_row[pred_df.at[idx, hit_id_col]]
                pred_df.at[idx, pred_col] = result["preds"][i]
                pred_df.at[idx, entropy_col] = result["entropies"][i]
                for ci, cls in enumerate(result["classes"]):
                    pred_df.at[idx, f"{category}_prob_{cls}_{source_label}"] = result["probs"][i, ci]

    # Hierarchy check for BOTH sources (domain-vs-parent-domain,
    # orf-vs-parent-orf), same logic as predict_substage -- per-edge plus
    # combined (any-level) hier_ok, via the shared helper so the held-out
    # validation experiment below is scored identically.
    edges = [(cat, config.CATEGORIES[cat]["parent"]) for cat in categories if config.CATEGORIES[cat]["parent"]]
    pred_df = _add_hierarchy_ok_columns(pred_df, edges, sources_present, expanded_df)

    pred_df = _sort_survey_hits(pred_df, categories, proteome_order=hierarchy_proteomes)

    pred_path = os.path.join(out_dir, "hierarchy_predictions.csv")
    pred_df.to_csv(pred_path, index=False)
    print(f"\n[survey.hierarchy] Wrote {pred_path}: {len(pred_df)} rows")

    keyword_cols = [c for c in pred_df.columns if c.startswith("is_")]
    keyword_mask = pred_df[keyword_cols].any(axis=1) if keyword_cols else pd.Series(False, index=pred_df.index)
    false_df = pred_df.loc[~keyword_mask].reset_index(drop=True)
    true_df  = pred_df.loc[keyword_mask].reset_index(drop=True)

    # Hierarchy violation-rate summary (both sources, hierarchy_proteomes
    # only): by proteome x keyword_state x edge x source. Distinct from
    # predict_substage's own hierarchy_violation_summary.csv, which covers
    # domain-source only, across all active proteomes.
    summary_rows = []
    for proteome_id in pred_df["proteome_id"].unique():
        for keyword_state, subset in (("false", false_df), ("true", true_df), ("all", pred_df)):
            sub = subset[subset["proteome_id"] == proteome_id]
            if sub.empty:
                continue
            summary_rows.append(
                _hierarchy_violation_row(sub, edges, sources_present, proteome_id, keyword_state,
                                          orf_id_col="accession")
            )

    # Baseline row: same hier_ok/violation-rate computation, but on
    # validate.py's held-out cross-validated predictions
    # (updated_expanded_dataset.csv) instead of the survey hit pool --
    # every row there already carries its single held-out prediction from
    # whichever vg1/vg2 direction tested it, so this is a genuine pooled
    # held-out evaluation, not survey-hit-pool-specific. Lets the paper
    # compare survey-hit violation rates against a proper validation
    # baseline in the same file, same columns, same rate definition.
    validate_path = os.path.join(paths["validate_dir"], "updated_expanded_dataset.csv")
    if os.path.exists(validate_path):
        validate_df = pd.read_csv(validate_path)
        validate_df = _add_hierarchy_ok_columns(validate_df, edges, sources_present, expanded_df)
        summary_rows.append(
            _hierarchy_violation_row(validate_df, edges, sources_present, "validation_experiment", "n/a",
                                      orf_id_col="orf_id")
        )
        print(f"[survey.hierarchy] Added validation_experiment baseline row from {validate_path} "
              f"({len(validate_df)} rows)")
    else:
        print(f"[survey.hierarchy] {validate_path} not found, skipping validation_experiment baseline row")

    hier_summary_df = pd.DataFrame(summary_rows)
    hier_summary_path = os.path.join(out_dir, "hierarchy_analysis_summary.csv")
    hier_summary_df.to_csv(hier_summary_path, index=False)
    print(f"[survey.hierarchy] Wrote {hier_summary_path}: {len(hier_summary_df)} rows")

    # Entropy-vs-hierarchy-status comparison (moved here from
    # predict_substage, same computation it already had, same scope --
    # this substage's hit pool IS already restricted to hierarchy_proteomes,
    # so no further proteome filtering is needed here).
    entropy_base = false_df
    entropy_rows = []
    for child_cat, parent_cat in edges:
        for source_label in sources_present:
            hier_col    = f"{child_cat}_hier_ok_{source_label}"
            entropy_col = f"{child_cat}_entropy_{source_label}"
            if hier_col not in entropy_base.columns or entropy_col not in entropy_base.columns:
                continue
            groups = list(entropy_base.groupby("proteome_id")) + [("combined", entropy_base)]
            for proteome_id, grp in groups:
                for hier_ok_value, sub in grp.groupby(hier_col):
                    vals = sub[entropy_col].dropna()
                    if len(vals) == 0:
                        continue
                    entropy_rows.append({
                        "proteome_id":    proteome_id,
                        "edge":           f"{child_cat}_vs_{parent_cat}",
                        "source":         source_label,
                        "hier_ok":        bool(hier_ok_value),
                        "mean_entropy":   float(vals.mean()),
                        "median_entropy": float(vals.median()),
                        "std_entropy":    float(vals.std()),
                        "n":              int(len(vals)),
                    })
    entropy_df = pd.DataFrame(entropy_rows)
    entropy_path = os.path.join(out_dir, "entropy_by_hierarchy_status.csv")
    entropy_df.to_csv(entropy_path, index=False)
    print(f"[survey.hierarchy] Wrote {entropy_path}: {len(entropy_df)} rows")


def run(paths, params, hpc, substages=None):
    if substages is None:
        substages = config.STAGES["survey"].get("substages", {})

    if substages.get("hmm_scan", False):
        print("\n[survey] Running hmm_scan substage...")
        hmm_scan(paths, params, hpc)
        print("\n[survey] hmm_scan substage complete.")

    if substages.get("predict", False):
        print("\n[survey] Running predict substage...")
        predict_substage(paths, params, hpc)
        print("\n[survey] predict substage complete.")

    if substages.get("hierarchy", False):
        print("\n[survey] Running hierarchy substage...")
        hierarchy_substage(paths, params, hpc)
        print("\n[survey] hierarchy substage complete.")
