# config.py
"""
User-facing configuration for the CatFunc pipeline (successor to
domain_inference, rebuilt stage by stage). Only what the `analyze` stage
needs is defined here for now — see domain_inference/config.py for the
full pipeline's config shape this is modeled on.
"""
import os

# base path for all input and output files
EXPERIMENT_DIR = "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/CatFunc/AAA"
# data directories
DATA_DIR      = os.path.join(EXPERIMENT_DIR, "data")
METADATA_DIR  = os.path.join(EXPERIMENT_DIR, "metadata")
ANALYSIS_DIR  = os.path.join(EXPERIMENT_DIR, "analysis")

# User provides these
INPUT_FILES = {
    "literature": "literature.csv",
    "msa":        "lit_unk_domains_hc50.aln",
    "unknowns":   "unknowns.csv",
}

# -----------------------------------------------------------------------------
# Category hierarchy
# Each category column in literature.csv gets a display name and a parent
# category (None if it's a root / independent axis). Enshrines the
# hierarchy for later use (e.g. hierarchy-violation checks) and gives
# categories human-readable names for reporting/plots.
# -----------------------------------------------------------------------------
CATEGORIES = {
    "category_0": {"name": "class",         "parent": None},
    "category_1": {"name": "mechanism",     "parent": "category_0"},
    "category_2": {"name": "archetype",     "parent": "category_1"},
    "category_3": {"name": "stoichiometry", "parent": None},
}

# -----------------------------------------------------------------------------
# Stages
# 'run' is the master switch for each stage
# -----------------------------------------------------------------------------
STAGES = {
    # Regenerating literature_lda_visualizations.pdf etc. -- stale since
    # Sep 12/16, predates every ESM window-truncation fix this session.
    # 5 literature orf_ids (NOD2_hs, dhcA_dd, NLRP3_hs, RPS4_at, NAIP_hs)
    # were long (>1022) but never covered by the earlier expanded_dataset/
    # unknowns fixes (not present in either file) -- their stale .pt +
    # the whole orf_sequence/csv/analyze cache were backed up and cleared
    # separately before this run.
    "analyze": {"run": True},
    "expand": {
        "run": False,
        "substages": {
            "search":    False,  # already run — all 103 queries processed
            "combine":   False,  # already run
            "visualize": False,  # already run
        },
    },
    # Already re-run and verified earlier tonight with the corrected
    # orf_sequence embeddings (window-pooling fix) -- off now so the
    # overnight survey run doesn't redo this.
    "validate": {
        "run": False,
        "substages": {
            "trajectory": False,
            "pooled":     False,
        },
    },
    "predict": {"run": False},
    # hmm_scan off -- hit data for the 10 active proteomes already exists.
    # predict: domain-only now (see stages/survey.py), all 10 active
    # proteomes, fast (no ORF windowing needed at all for this substage).
    # hierarchy: domain+ORF classification restricted to
    # PARAMS["survey"]["predict"]["hierarchy_proteomes"] (human/E.coli/
    # S.pombe) -- this is the expensive one (158 ORFs need the sliding-
    # window extraction fix), meant to run overnight.
    # predict already ran and verified last night, unaffected by the
    # combined hier_ok addition (domain-only) -- off so this re-run only
    # touches hierarchy_substage.
    # Already re-run and verified earlier -- off so this run only touches analyze.
    "survey": {
        "run": False,
        "substages": {
            "hmm_scan":  False,
            "predict":   False,
            "hierarchy": False,
        },
    },
}

# Shared mechanism (category_1) color palette — used by both analyze's and
# expand's LDA visualizations so the same label always gets the same color.
MECHANISM_COLOR_PALETTE = {
    "rotary":    "cornflowerblue",
    "recip":     "gold",
    "initiator": "mediumorchid",
    "loader":    "#d62728",
    "signaling": "darkblue",
    "unknown":   "gray",
}

# Archetype (category_2) color palette — used by analyze's self-mapped
# archetype LDA visualization (fit on archetype, colored by archetype).
ARCHETYPE_COLOR_PALETTE = {
    "unfoldase":     "#2271b4",
    "remodeling":    "#6baed5",
    "locomotion":    "#fdf7bc",
    "disassembly":   "#d86127",
    "replication":   "#bcbddb",
    "transposition": "#412b79",
    "loading":       "#d62c28",
    "signaling":     "#2d307b",
    "unknown":       "gray",
}

# -----------------------------------------------------------------------------
# HPC Environment
# Module commands required for subprocess calls in CHPC environment
# -----------------------------------------------------------------------------
HPC = {
    "module_cmd": [
        "module use $HOME/MyModules",
        "module load miniforge3/latest"
    ]
}

# -----------------------------------------------------------------------------
# Parameters
# -----------------------------------------------------------------------------
PARAMS = {
    "inference_settings": {
        "extract_script": "/uufs/chpc.utah.edu/common/home/u6060853/software/scripts/classification/extract.py",
        "pretrained_model": "esm2_t33_650M_UR50D",
        "repr_layers":      list(range(33)),   # extract all 33 layers
        "include":          "mean",
        # ESM2 has a hard 1022-residue input limit (extract.py's
        # --truncation_seq_length default, left untouched). Sequences longer
        # than window_size are split into overlapping windows (overlap =
        # window_overlap residues) that are each embedded separately and
        # pooled back into one whole-sequence vector -- see
        # inference_utils.prepare_embeddings/_expand_for_windowing. Must not
        # exceed 1022 or windows would themselves get silently truncated.
        "window_size":    1022,
        "window_overlap": 150,
    },
    # Single fixed logistic-regression C used everywhere (validate's
    # trajectory/pooled substages and predict) — no per-category/source CV
    # tuning. Tuning was dropped because (a) 7 of 8 (embedding_source,
    # category) combinations independently converged to C=0.01 anyway with
    # no accuracy benefit from picking otherwise, and (b) letting C vary
    # across categories makes per-row entropy incomparable between them,
    # since entropy's magnitude is directly shaped by regularization
    # strength, not just genuine label uncertainty.
    "classifier_C": 0.1,
    "analyze": {
        # Defaults to all of CATEGORIES above; override here to analyze a subset.
        "embedding_source":        "both",     # "domain_sequence", "orf_sequence", or "both"
        "layers":                  list(range(33)),
        # ~50/50 stratified split of literature ORFs into vg1/vg2, written
        # into updated_literature.csv and carried through to every domain
        # derived from a given ORF (same mechanism as category transfer) —
        # a query and its own homologs never end up split across train/test.
        "validation_group_category": None,  # None = auto-detect (category with the most classes)
        "validation_group_seed":     42,
        # Separate LDA experiment: for each entry in `categories`, project
        # literature embeddings into that category's own discriminant
        # space and color by that SAME category (self-mapped, not
        # cross-category), using that category's matched palette — the
        # standard "how well does this category separate in its own LDA
        # space" plot, for both domain_sequence and orf_sequence. Single
        # population (literature only) — unlike expand.visualize, there's
        # no expanded-dataset comparison at this stage yet.
        "lda_visualize": {
            "run":              True,
            "layers":           [2, 12, 22, 32],
            "embedding_source": "both",  # "domain_sequence", "orf_sequence", or "both"
            "categories": [
                {"category": "category_1", "color_palette": MECHANISM_COLOR_PALETTE},
                {"category": "category_2", "color_palette": ARCHETYPE_COLOR_PALETTE},
            ],
        },
    },
    "expand": {
        "search": {
            "blast_settings": {
                "hit_limit":    20000,
                "database":     "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/nr",
                "num_threads":  28
            },
            "blast_filter_params": {
                "coverage_min":  0.8,
                "coverage_max":  1.1,
                "homology_max":  85,
                "homology_min":  25
            },
            "hmm_settings": {
                "num_threads": 8
            },
            "skip_hmmer": False,   # reuse a cached .domtblout instead of re-running hmmsearch
            # no mmseqs_settings here — the intra-query redundancy filter is deferred
            "c_value_max":  1.0e-4,
            "length_min":   190,
            "length_max":   500,
            "walker_a_min": 12,
        },
        "combine": {
            # cascade: each round clusters only the previous round's
            # representatives, not the full pool (avoids connected-component
            # clustering chaining distant sequences together in a huge pool).
            "homology_cutoffs":  [0.7, 0.5, 0.3],
            "mmseqs_settings": {
                "cluster_mode": 1,
                "num_threads":  8,
                "k":            5,
            },
            "stratify_category": "category_1",  # 5-class mechanism level
            # For each label of this category (e.g. stoichiometry:
            # homomer/heteromer), recomputes label_distribution.csv's
            # per-category breakdown on that label's subset for every
            # OTHER category and appends the cross-tab to the bottom of
            # the file. Set to None to skip this extra analysis.
            "label_distribution_pivot_category": "category_3",
            # mafft alignment of the final tier's expanded_domains.fasta /
            # expanded_orfs.fasta, run automatically at the end of combine.
            # Set to False to skip (e.g. while iterating on clustering
            # params and not wanting to wait on/regenerate alignments).
            "run_mafft": True,
            "mafft_settings": {
                "maxiterate": 1000,
                "retree":     3,
                "threads":    20,  # per mafft process; domains + orfs run concurrently
            },
        },
        "visualize": {
            # LDA space is fit on literature embeddings only; non-query
            # (BLAST-derived) expanded_dataset.csv domains are projected
            # into that same fitted space, not re-fit. Each layer/range gets
            # a paired page: literature-only, then literature+expanded,
            # both using the identical fitted axes for direct comparison.
            "embedding_source": "domain_sequence",  # only source supported right now
            "layers":         [2, 12, 16, 22, 32],
            # extra pages using pooled (concatenated) features across each
            # range, in addition to the single layers above
            "layer_ranges":   [[11, 19]],
            "lda_category":   "category_1",  # mechanism space
            "color_category": "category_1",  # colored by mechanism too, for this experiment
            "color_palette":  MECHANISM_COLOR_PALETTE,
        },
    },
    "validate": {
        "trajectory": {
            # Single logistic regression at the fixed classifier_C (see
            # above), swept per-layer — no layer pooling yet, deferred
            # until it's decided whether pooling actually helps.
            "embedding_source": "both",  # "domain_sequence", "orf_sequence", or "both"
            "seed":             42,
        },
        "pooled": {
            # Pools this layer range (concatenated features) instead of a
            # single layer, at the same fixed classifier_C.
            "layer_range":      [11, 19],
            "embedding_source": "both",  # "domain_sequence", "orf_sequence", or "both"
            "shuffle_count":    5,       # label-shuffled null repeats per (source, category)
            "seed":             42,
        },
    },
    "predict": {
        # Final production classifiers: trains on the ENTIRE
        # expanded_dataset.csv (no train/test split), predicts on
        # unknowns.csv. Same pooled layer range and fixed classifier_C as
        # validate's pooled substage.
        "embedding_source": "both",  # "domain_sequence", "orf_sequence", or "both"
        "layer_range":      [11, 19],
        "seed":             42,
    },
    "survey": {
        # Shared by both substages: hmm_scan computes is_* keyword columns
        # and writes one hits CSV per active proteome; predict re-reads
        # those same proteomes/filters, so both live here (not duplicated
        # per-substage) to avoid the two substages drifting apart.
        "proteomes": [
            {"id": "e_coli",         "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/ecoli_canonical.fasta",         "active": True},
            {"id": "d_melanogaster", "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/dmelanogaster_canonical.fasta", "active": True},
            {"id": "d_discoideum",   "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/ddiscoideum_canonical.fasta",   "active": True},
            {"id": "c_elegans",      "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/celegans_canonical.fasta",      "active": True},
            {"id": "h_sapiens",      "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/hsapiens_canonical.fasta",      "active": True},
            {"id": "a_thaliana",     "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/athaliana_canonical.fasta",     "active": True},
            {"id": "b_subtilis",     "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/bsubtilis_canonical.fasta",     "active": True},
            {"id": "s_cerevisiae",   "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/scerevisiae_canonical.fasta",   "active": True},
            {"id": "s_elongatus",    "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/selongatus_canonical.fasta",    "active": False},
            {"id": "s_pombe",        "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/spombe_canonical.fasta",        "active": True},
            {"id": "s_solfataricus", "fasta": "/uufs/chpc.utah.edu/common/home/cryoem-group3/hill/swan/databases/proteomes/ssolfataricus_canonical.fasta", "active": True},
        ],
        "keyword_filters": [
            {"name": "abc_transporter", "keywords": ["ABC", "transporter", "transporting"]},
        ],
        "hmm_scan": {
            # Whole-proteome HMM scan (not a per-query BLAST/HMMer flow
            # like expand.search) — both "strong" and "weak" HMM_signal
            # hits are kept (unlike expand.search's hard filter).
            "filter_params": {
                "c_value_max":  1.0e-4,
                "length_min":   190,
                "length_max":   500,
                "walker_a_min": 12,
            },
            "hmm_settings": {
                "num_threads": 8,
            },
        },
        "predict": {
            # Same fixed classifier_C + pooled layer_range as predict.py;
            # no per-category tuning. predict_substage always classifies
            # domain_sequence only, across every `active` proteome above;
            # hierarchy_substage always classifies both domain_sequence
            # and orf_sequence, restricted to hierarchy_proteomes below --
            # no embedding_source knob needed, each substage's sources are
            # fixed by what it's for.
            "layer_range":                  [11, 19],
            "seed":                         42,
            "priority_proteomes":           ["h_sapiens", "e_coli"],            # FASTA export + sort priority
            "hierarchy_proteomes":          ["h_sapiens", "e_coli", "s_pombe"], # hierarchy_substage's proteome scope
        },
    },
}
