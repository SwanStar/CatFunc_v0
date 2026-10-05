# inference_utils.py
"""
ESM2 embedding extraction and caching, ported from domain_inference's
utils/inference_utils.py and trimmed to only the embedding-prep pipeline
used by stages/analyze.py, plus a couple of small shared ML helpers
(orf_level_frame, shannon_entropy_row, fit_and_predict) used by
validate.py/predict.py/survey.py. The old pipeline's MODEL_REGISTRY/
tuning-file/hierarchy machinery intentionally not ported — see
domain_inference/utils/inference_utils.py for the full version.
"""

import os
import subprocess
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression

from utils import lda_utils

# In-memory cache of layer CSVs for the lifetime of the process.
_CSV_DF_CACHE = {}

# Separator between a real sequence id and its window index in pseudo-ids
# used only during extraction (e.g. "KAG2379976.1__w0"). Verified no real
# domain_id/orf_id in this dataset contains "__", so these can't collide
# with a real id. Window ids never leave generate_esm_input/run_esm_extract/
# pool_windowed_embeddings -- every other function (build_csv_cache,
# check_pt_layer_completeness, check_csv_cache, load_layer) only ever sees
# real ids from id_list, so nothing downstream needs to know windowing exists.
WINDOW_ID_SEP = "__w"

# -----------------------------------------------------------------------------
# EMBEDDING EXTRACTION
# -----------------------------------------------------------------------------

def generate_esm_input(id_list, pt_dir, fasta_path):
    """
    Check which sequences are missing .pt files and write a fasta for ESM2 input.
    Returns number of sequences needing computation.
    """
    to_compute = [i for i in id_list if not os.path.exists(os.path.join(pt_dir, f"{i}.pt"))]

    if to_compute:
        print(f"  {len(to_compute)} sequences need embeddings")
    else:
        print("  All .pt files already exist")

    return to_compute


def _sliding_windows(seq_len, window_size, stride):
    """
    (start, end) half-open slices tiling [0, seq_len). Consecutive windows
    overlap by (window_size - stride) residues. The final window is anchored
    to end exactly at seq_len (start = seq_len - window_size) rather than
    left as a short ragged remainder, so every window -- including the last
    -- has length == window_size and full C-terminal coverage is guaranteed.
    Only called when seq_len > window_size.
    """
    starts = list(range(0, seq_len - window_size + 1, stride))
    last_start = seq_len - window_size
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [(s, s + window_size) for s in starts]


def _expand_for_windowing(to_compute, seq_dict, pt_dir, window_size, window_overlap):
    """
    Preprocessing step for run_esm_extract(). ESM2 silently truncates any
    input over window_size residues (extract.py's --truncation_seq_length
    default), so sequences longer than that are split into overlapping
    windows here instead of being handed to extract.py whole.

    Returns:
      fasta_ids, fasta_seqs: what should actually be written to the ESM
        input fasta this call -- short sequences (<=window_size) unchanged
        (identical to the pre-windowing behavior), long sequences as
        per-window pseudo-entries "{real_id}__w{i}" (skipping any window
        .pt that already exists on disk, so an interrupted run resumes
        without re-embedding completed windows).
      window_plan: {real_id: [(window_id, window_length), ...]} for every
        long id in to_compute, so pooling always runs for them even if all
        their windows were already cached from a prior partial run.
    """
    assert window_size <= 1022, "window_size must not exceed extract.py's truncation default (1022)"
    stride = window_size - window_overlap
    fasta_ids, fasta_seqs = [], []
    window_plan = {}
    for real_id in to_compute:
        seq = seq_dict[real_id]
        if len(seq) <= window_size:
            fasta_ids.append(real_id)
            fasta_seqs.append(seq)
            continue
        plan = []
        for i, (start, end) in enumerate(_sliding_windows(len(seq), window_size, stride)):
            win_id = f"{real_id}{WINDOW_ID_SEP}{i}"
            plan.append((win_id, end - start))
            if not os.path.exists(os.path.join(pt_dir, f"{win_id}.pt")):
                fasta_ids.append(win_id)
                fasta_seqs.append(seq[start:end])
        window_plan[real_id] = plan
    return fasta_ids, fasta_seqs, window_plan


def pool_windowed_embeddings(pt_dir, window_plan, layers):
    """
    Postprocessing step, run after run_esm_extract(). For each real_id in
    window_plan, combines its windows' mean_representations into one vector
    per layer, weighted by each window's residue length, and writes
    {pt_dir}/{real_id}.pt in the exact {"label", "mean_representations"}
    shape extract.py itself produces -- so build_csv_cache() and everything
    downstream needs no changes. Length-weighting (rather than an
    unweighted mean) means this stays correct even if window_size/overlap
    are changed such that windows end up unequal length; with the current
    settings every window is the same length, so it's numerically a plain
    mean today, but the weighting isn't relying on that holding forever.
    Window .pt files themselves are left on disk as a resumable
    intermediate cache -- they're invisible to every other function since
    those only ever iterate real ids from id_list, never window ids.
    """
    import torch
    for real_id, windows in window_plan.items():
        weighted_sums = {l: None for l in layers}
        total_weight = 0
        ok = True
        for win_id, weight in windows:
            win_pt = os.path.join(pt_dir, f"{win_id}.pt")
            if not os.path.exists(win_pt):
                print(f"  Warning: window {win_id} missing, cannot pool {real_id}")
                ok = False
                break
            data = torch.load(win_pt, weights_only=True)
            total_weight += weight
            for l in layers:
                contribution = data["mean_representations"][l] * weight
                weighted_sums[l] = contribution if weighted_sums[l] is None else weighted_sums[l] + contribution
        if not ok:
            continue
        mean_reprs = {l: weighted_sums[l] / total_weight for l in layers}
        torch.save({"label": real_id, "mean_representations": mean_reprs},
                    os.path.join(pt_dir, f"{real_id}.pt"))
        print(f"  Pooled {len(windows)} windows -> {real_id}.pt ({total_weight} total residues)")


def check_pt_layer_completeness(pt_dir, id_list, layers):
    """
    Check which .pt files exist but are missing required layer data.
    Returns list of IDs whose .pt files don't contain all requested layers.
    """
    import torch
    incomplete = []
    for identifier in id_list:
        pt_file = os.path.join(pt_dir, f"{identifier}.pt")
        if not os.path.exists(pt_file):
            continue
        try:
            data = torch.load(pt_file, weights_only=True)
            for layer in layers:
                if layer not in data["mean_representations"]:
                    incomplete.append(identifier)
                    break
        except Exception:
            incomplete.append(identifier)
    if incomplete:
        print(f"  {len(incomplete)} .pt files have incomplete layer data")
    else:
        print("  All .pt files have complete layer data")
    return incomplete


def run_esm_extract(fasta_path, pt_dir, esm_settings, hpc):
    """
    Run ESM2 extract.py as a subprocess.
    """
    extract_py   = esm_settings["extract_script"]
    model        = esm_settings["pretrained_model"]
    layers       = esm_settings["repr_layers"]
    include      = esm_settings["include"]
    layers_str   = " ".join(str(l) for l in layers)
    module_load  = "\n".join(hpc["module_cmd"])

    command = f"""
{module_load}
python3 {extract_py} {model} {fasta_path} {pt_dir} --repr_layers {layers_str} --include {include} --nogpu
"""
    try:
        subprocess.run(command, shell=True, check=True, executable="/bin/bash")
        print("  ESM2 extraction complete")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  ESM2 extraction failed: {e}")
        return False


# -----------------------------------------------------------------------------
# CACHE MANAGEMENT
# -----------------------------------------------------------------------------

def check_csv_cache(csv_dir, id_list, layers, cache_key=None):
    """
    Check which layers need to be rebuilt based on whether all identifiers
    are present as columns in the cached CSV.
    Returns list of layers that need rebuilding.
    """
    if cache_key:
        csv_dir = os.path.join(csv_dir, cache_key)
    layers_to_rebuild = []

    for layer in layers:
        csv_path = os.path.join(csv_dir, f"layer_{layer}.csv")
        if not os.path.exists(csv_path):
            layers_to_rebuild.append(layer)
            continue

        cached_ids = set(pd.read_csv(csv_path, nrows=0).columns.tolist())
        missing    = set(id_list) - cached_ids
        if missing:
            print(f"  Layer {layer}: {len(missing)} sequences missing from cache")
            layers_to_rebuild.append(layer)

    return layers_to_rebuild


def build_csv_cache(pt_dir, csv_dir, id_list, layers, cache_key=None):
    """
    Build or rebuild layer CSVs from .pt files.
    CSVs are stored as (features x sequences) with identifiers as column headers.
    """
    import torch
    if cache_key:
        csv_dir = os.path.join(csv_dir, cache_key)
    os.makedirs(csv_dir, exist_ok=True)

    for layer in layers:
        print(f"  Building cache for layer {layer}...")
        embeddings  = []
        valid_ids   = []

        for identifier in id_list:
            pt_file = os.path.join(pt_dir, f"{identifier}.pt")
            if os.path.exists(pt_file):
                try:
                    data = torch.load(pt_file, weights_only=True)
                    embeddings.append(data["mean_representations"][layer].numpy())
                    valid_ids.append(identifier)
                except KeyError:
                    print(f"  Warning: {identifier}.pt missing layer {layer}, skipping")
            else:
                print(f"  Warning: {identifier}.pt not found, skipping")

        array = np.vstack(embeddings).T
        df    = pd.DataFrame(array, columns=valid_ids)
        df.to_csv(os.path.join(csv_dir, f"layer_{layer}.csv"), index=False)
        print(f"  Layer {layer} cached: {array.shape[0]} features x {len(valid_ids)} sequences")


# -----------------------------------------------------------------------------
# LOADING
# -----------------------------------------------------------------------------

def load_layer(csv_dir, layer, id_list):
    """
    Load a single layer CSV and return array in (sequences x features) orientation.
    Filters to only the requested id_list in the correct order.
    """
    csv_path = os.path.join(csv_dir, f"layer_{layer}.csv")
    df       = _CSV_DF_CACHE.get(csv_path)
    if df is None:
        df = pd.read_csv(csv_path)
        _CSV_DF_CACHE[csv_path] = df

    df       = df[id_list]
    array    = df.values.T
    return array


# -----------------------------------------------------------------------------
# TOP-LEVEL ORCHESTRATION
# -----------------------------------------------------------------------------

def prepare_embeddings(id_list, seq_dict, paths, esm_settings, hpc, layers,
                        layers_subset=None, embed_dir=None, cache_key=None):
    """
    Full embedding preparation pipeline:
    1. Check which .pt files are missing, run ESM2 if needed
    2. Check CSV cache, rebuild layers if needed

    If layers_subset is provided (e.g. [14,15,16,17,18]), only build CSV cache
    for those layers, saving disk I/O. Otherwise builds for all layers.

    If cache_key is provided, CSV cache is stored in {embed_dir}/csv/{cache_key}/
    for per-group isolation (smaller caches). .pt files remain shared.
    """
    from utils import io_utils as io

    if embed_dir is None:
        embed_dir = os.path.join(paths["embeddings"])

    pt_dir  = os.path.join(embed_dir, "pt")
    csv_dir = os.path.join(embed_dir, "csv")
    os.makedirs(pt_dir,  exist_ok=True)
    os.makedirs(csv_dir, exist_ok=True)

    # --- Step 1: Check CSV cache first (cheap) ---
    cache_layers = layers_subset if layers_subset is not None else layers
    layers_to_rebuild = check_csv_cache(csv_dir, id_list, cache_layers, cache_key)

    # --- Step 2: ESM2 extraction only if CSV needs rebuilding ---
    to_compute = []
    if layers_to_rebuild:
        fasta_path = os.path.join(embed_dir, "esm_input.fasta")
        to_compute = generate_esm_input(id_list, pt_dir, fasta_path)

        # Also check .pt files that exist but may be missing required layers
        incomplete_pts = check_pt_layer_completeness(pt_dir, id_list, cache_layers)
        to_compute.extend(incomplete_pts)
        to_compute = list(dict.fromkeys(to_compute))

        if to_compute:
            window_size    = esm_settings.get("window_size", 1022)
            window_overlap = esm_settings.get("window_overlap", 150)
            fasta_ids, fasta_seqs, window_plan = _expand_for_windowing(
                to_compute, seq_dict, pt_dir, window_size, window_overlap)

            if fasta_ids:
                io.write_fasta(fasta_path, fasta_ids, fasta_seqs)
                success = run_esm_extract(fasta_path, pt_dir, esm_settings, hpc)
                if not success:
                    raise RuntimeError("ESM2 extraction failed, cannot proceed")

            if window_plan:
                pool_windowed_embeddings(pt_dir, window_plan, esm_settings["repr_layers"])

    # --- Step 3: Build CSV cache ---
    if layers_to_rebuild:
        build_csv_cache(pt_dir, csv_dir, id_list, layers_to_rebuild, cache_key)
    else:
        print("  CSV cache is up to date")


# -----------------------------------------------------------------------------
# SHARED ML HELPERS (used by both validate.py and predict.py)
# -----------------------------------------------------------------------------

def orf_level_frame(expanded_df):
    """One row per unique orf_id — a multi-domain ORF's orf_sequence is
    otherwise embedded/classified redundantly under each of its domain_ids."""
    return expanded_df.drop_duplicates(subset="orf_id").reset_index(drop=True)


def shannon_entropy_row(probs_row):
    with np.errstate(divide="ignore", invalid="ignore"):
        terms = probs_row * np.log(probs_row)
    return float(-np.nansum(terms))


def fit_and_predict(X_train, y_train, X_unknown, C, seed):
    """
    PCA-fit on X_train, apply to X_unknown, fit LogisticRegression(C=C) on
    the full labeled training set (no train/test split) and predict on
    X_unknown. Used by predict.py and survey.py — both train on the
    entire expanded_dataset.csv and classify genuinely new sequences with
    no ground truth to score against (unlike validate.py's
    _classify_direction, which holds out a test split to measure
    accuracy).
    """
    scaler, pca, X_train_pca = lda_utils.pca_fit(X_train)
    X_unknown_pca = lda_utils.pca_apply(scaler, pca, X_unknown)

    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed)
    clf.fit(X_train_pca, y_train)
    preds     = clf.predict(X_unknown_pca)
    probs     = clf.predict_proba(X_unknown_pca)
    classes   = clf.classes_
    entropies = np.array([shannon_entropy_row(p) for p in probs])

    return {"preds": preds, "probs": probs, "classes": classes, "entropies": entropies}
