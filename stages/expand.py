# expand.py
"""
Expand stage.

search substage: BLAST homology search -> HMMer domain scanning, ported
from domain_inference/stages/expand.py's search_substage and helpers (the
MMseqs-based intra-query redundancy filter is still deferred).

combine substage: gathers search's per-query filtered domains + the
literature domains, clusters the combined set once via MMseqs at a single
homology cutoff, and picks representatives with stratified_homology_filter
(literature-priority + category-label balancing) instead of
domain_inference's homology-blind three-tier hc30/hc50/hc70 cascade.
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
from matplotlib.backends.backend_pdf import PdfPages
import config

from utils import io_utils, seq_utils, subprocesses
from utils import inference_utils as inf
from utils import lda_utils
from utils.feature_cache import FeatureCache


def filter_blast_hits(identifier, query_dir, filter_params):
    """Filter BLAST results for a single query. Writes filtered CSV and returns stats."""
    min_coverage = filter_params['coverage_min']
    max_coverage = filter_params['coverage_max']
    min_homology = filter_params['homology_min']
    max_homology = filter_params['homology_max']

    blast_tsv   = os.path.join(query_dir, f"{identifier}_BLAST.tsv")
    blast_fasta = os.path.join(query_dir, f"{identifier}_BLAST.fasta")
    query_fasta = os.path.join(query_dir, f"{identifier}.fasta")
    output_csv  = os.path.join(query_dir, f"{identifier}_BLAST_proc.csv")

    blast_df     = io_utils.read_blast_tsv(blast_tsv)
    blast_total  = len(blast_df)
    query_length = len(io_utils.read_fasta(query_fasta)['sequence'].iloc[0])

    blast_df['coverage'] = blast_df['alignment_length'] / query_length
    blast_df = blast_df[blast_df['coverage'] >= min_coverage]
    after_coverage_highpass = len(blast_df)

    blast_df = blast_df[blast_df['coverage'] <= max_coverage]
    after_coverage_lowpass = len(blast_df)

    blast_df = blast_df[blast_df['homology_score'] >= min_homology]
    after_homology_highpass = len(blast_df)

    blast_df = blast_df[blast_df['homology_score'] <= max_homology]
    after_homology_lowpass = len(blast_df)

    blast_df       = blast_df.drop_duplicates(subset=['target_name'])
    blast_fasta_df = io_utils.read_fasta(blast_fasta, truncate_header=True)
    blast_fasta_df = blast_fasta_df.drop_duplicates(subset=['header'])

    processed_df = pd.merge(blast_df, blast_fasta_df,
                            left_on='target_name',
                            right_on='header',
                            how='inner')
    processed_df.to_csv(output_csv, index=False)

    return {
        'query':                               identifier,
        'blast_total':                         blast_total,
        'sequences_removed_coverage_highpass': blast_total - after_coverage_highpass,
        'sequences_removed_coverage_lowpass':  after_coverage_highpass - after_coverage_lowpass,
        'sequences_removed_highpass':          after_coverage_lowpass - after_homology_highpass,
        'sequences_removed_lowpass':           after_homology_highpass - after_homology_lowpass,
        'final_sequences':                     after_homology_lowpass
    }


def prepare_hmmer_input(identifier, query_dir):
    """Read BLAST_proc.csv and write FASTA for HMMer input."""
    blast_proc_csv = os.path.join(query_dir, f"{identifier}_BLAST_proc.csv")
    df             = pd.read_csv(blast_proc_csv, keep_default_na=False)
    output_fasta   = os.path.join(query_dir, f"{identifier}_BLAST_proc.fasta")
    io_utils.write_fasta(output_fasta, df['header'], df['sequence'])
    return output_fasta


def extract_domains_from_hmm_hits(identifier, query_dir, input_fasta, query_row, annotation_cols):
    """
    Extract domains from HMMer output and associate with query annotations.
    Renamed from domain_inference's `extract_domains` — that name collides
    with the different, existing seq_utils.extract_domains here.

    annotation_cols: columns to copy from query_row onto every extracted
    domain (category_0..category_3 plus validation_group) — the same
    mechanism that lets a literature ORF's validation_group flow onto all
    of its BLAST/HMMer-derived hits, so a query and its own homologs never
    end up split across train/test.
    """
    input_domtblout = os.path.join(query_dir, f"{identifier}.domtblout")
    output_csv_path = os.path.join(query_dir, f"{identifier}_domains_unfiltered.csv")

    hmm_df   = io_utils.read_hmmsearch_domtblout(input_domtblout)
    fasta_df = io_utils.read_fasta(input_fasta)

    print(f"  {identifier}: {len(hmm_df)} HMM hits found")

    domain_df = pd.merge(hmm_df, fasta_df,
                         left_on="target_name",
                         right_on="header")

    # Generate domain identifiers ("-" bounds separator, matching seq_utils.extract_domain)
    domain_df['boundaries']        = domain_df['env_from'].astype(str) + "-" + domain_df['env_to'].astype(str)
    domain_df['domain_identifier'] = domain_df['target_name'] + "_" + domain_df['boundaries']
    domain_df['blast_query']       = identifier

    # Extract domain sequences
    domain_df['domain_sequence'] = list(
        map(seq_utils.extract_domain, domain_df['sequence'], domain_df['boundaries'])
    )

    # Transfer annotations (categories + validation_group) from query row
    for col in annotation_cols:
        if col in query_row.index:
            domain_df[col] = query_row[col]

    output_cols = ['domain_identifier', 'domain_sequence', 'sequence', 'target_name', 'c-value', 'blast_query'] + list(annotation_cols)
    output_df   = domain_df[output_cols].drop_duplicates(subset=['domain_identifier'])

    print(f"  {identifier}: {len(output_df)} domains extracted")
    output_df.to_csv(output_csv_path, index=False)


def filter_domains(identifier, query_dir, params):
    """
    Filter domains for a single query by c-value, length, and Walker-A
    motif score. Returns filtered dataframe and stats.

    No MMseqs redundancy filter here (domain_inference's filter_domains
    also dedupes near-identical hits via MMseqs at this point) — deferred
    alongside the combine substage.
    """
    c_value_max  = params['c_value_max']
    length_min   = params['length_min']
    length_max   = params['length_max']
    walker_a_min = params['walker_a_min']

    domain_df = pd.read_csv(os.path.join(query_dir, f"{identifier}_domains_unfiltered.csv"))

    stats = {'identifier': identifier, 'domains_start': len(domain_df)}

    # c-value filter
    before = len(domain_df)
    domain_df = domain_df[domain_df['c-value'] < c_value_max]
    stats['cvalue_removed'] = before - len(domain_df)

    # length filters
    before = len(domain_df)
    domain_df = domain_df[domain_df['domain_sequence'].str.len() >= length_min]
    stats['length_highpass_removed'] = before - len(domain_df)

    before = len(domain_df)
    domain_df = domain_df[domain_df['domain_sequence'].str.len() <= length_max]
    stats['length_lowpass_removed'] = before - len(domain_df)

    # walker_a filter
    domain_df['walkerA_score'] = [seq_utils.score_walkerA(s) for s in domain_df['domain_sequence']]
    before = len(domain_df)
    domain_df = domain_df[domain_df['walkerA_score'] >= walker_a_min]
    stats['walkerA_removed'] = before - len(domain_df)

    stats['domains_final'] = len(domain_df)

    filtered_csv = os.path.join(query_dir, f"{identifier}_domains_filtered.csv")
    domain_df.to_csv(filtered_csv, index=False)

    print(f"  {identifier}: {stats['domains_start']} → {stats['domains_final']} domains after filtering")

    return domain_df, stats


def write_filtering_summary(all_stats, out_dir):
    """Write per-query filtering statistics to a CSV summary file."""
    summary_df  = pd.DataFrame(all_stats)
    output_path = os.path.join(out_dir, "filtering_summary.csv")
    summary_df.to_csv(output_path, index=False)
    print(f"  ✓ Filtering summary written: {output_path}")


# ===== combine substage =====

def stratified_homology_filter(cluster_tsv_path, label_map, category, priority_ids=None, vg_map=None):
    """
    Pick one representative per MMseqs cluster, choosing the member whose
    label is currently least represented in the running representative set
    (a greedy global balancing pass) instead of MMseqs' own homology-blind
    default representative.

    cluster_tsv_path: path to an mmseqs easy-cluster *_cluster.tsv file
                       (tab-separated 'representative, member', no header —
                       same format priority_combine reads in domain_inference).
    label_map: dict-like domain_id -> label value for `category`. Every
               domain already carries this, inherited from its query ORF
               during the search substage's extract_domains_from_hmm_hits.
    category: name of the category being stratified on (for logging only).
    priority_ids: optional set of domain_ids (e.g. literature-derived
                  domains) that should always win the representative slot
                  when present in a cluster — label-balancing then applies
                  only among the priority members present (if more than
                  one), so a curated literature domain is never dropped in
                  favor of a homology-derived one. Falls back to balancing
                  over all members when no priority member is present.
    vg_map: optional dict-like domain_id -> validation_group. When multiple
            candidates tie on category-label balance, the tie is broken by
            whichever validation_group currently has the lowest running
            count, instead of arbitrary cluster.tsv order — keeps the two
            validation groups roughly balanced in the final representative
            set too.

    Returns a DataFrame with columns [domain_id, cluster, label] — one row
    per cluster, `domain_id` being the chosen representative.
    """
    cluster_df = pd.read_csv(cluster_tsv_path, sep='\t', names=['representative', 'member'], header=None)
    priority_ids = priority_ids or set()
    vg_map = vg_map or {}

    label_counts = {label: 0 for label in set(label_map.values())}
    vg_counts    = {vg: 0 for vg in set(vg_map.values())} if vg_map else {}

    representatives = []
    for cluster_id, group in cluster_df.groupby('representative'):
        members = group['member'].tolist()
        member_labels = {m: label_map.get(m) for m in members if label_map.get(m) is not None}

        if not member_labels:
            # No labeled members at all (shouldn't happen — every domain
            # inherits a label during search) — fall back to mmseqs' pick.
            representatives.append({'domain_id': cluster_id, 'cluster': cluster_id, 'label': None})
            continue

        priority_members = {m: label for m, label in member_labels.items() if m in priority_ids}
        candidates = priority_members if priority_members else member_labels

        # Members whose label currently has the lowest running count among
        # labels present in this cluster (restricted to priority members
        # first, if any are present).
        min_count = min(label_counts[label] for label in candidates.values())
        tied = [m for m in candidates if label_counts[candidates[m]] == min_count]

        if len(tied) > 1 and vg_map:
            # Break the tie by whichever validation_group is currently
            # least represented; members with no known validation_group
            # sort last.
            chosen_member = min(tied, key=lambda m: vg_counts.get(vg_map.get(m), float('inf')))
        else:
            chosen_member = tied[0]

        chosen_label = candidates[chosen_member]
        label_counts[chosen_label] += 1

        chosen_vg = vg_map.get(chosen_member)
        if vg_map and chosen_vg is not None:
            vg_counts[chosen_vg] = vg_counts.get(chosen_vg, 0) + 1

        representatives.append({'domain_id': chosen_member, 'cluster': cluster_id, 'label': chosen_label})

    rep_df = pd.DataFrame(representatives)
    print(f"  [stratified_homology_filter] {category}: {len(rep_df)} clusters, "
          f"final label counts: {label_counts}")
    return rep_df


def _category_count_rows(lit_df, tier_dfs, final_tier_id, category, pivot_category=None, pivot_label=None):
    """
    Builds the "<category>_total" row plus one row per label present in
    tier_dfs[final_tier_id][category] (ordered by final-tier frequency),
    each with literature/<tier>/expanded/vg1/vg2 counts. lit_df/tier_dfs
    may already be filtered to a pivot-label subset; pivot_category/
    pivot_label are stamped onto every returned row (left None for the
    unfiltered top-level section).
    """
    final_df = tier_dfs[final_tier_id]

    lit_valid = lit_df[category].dropna()
    lit_vc = lit_valid.value_counts()

    tier_valid = {tid: df[category].dropna() for tid, df in tier_dfs.items()}
    tier_vc    = {tid: v.value_counts() for tid, v in tier_valid.items()}

    vg1_valid = final_df.loc[final_df['validation_group'] == 'vg1', category].dropna()
    vg2_valid = final_df.loc[final_df['validation_group'] == 'vg2', category].dropna()
    vg1_vc, vg2_vc = vg1_valid.value_counts(), vg2_valid.value_counts()

    def _col_name(tid):
        return 'expanded' if tid == final_tier_id else tid

    rows = []
    total_row = {
        'label':           f'{category}_total',
        'literature':      int(len(lit_valid)),
        'pivot_category':  pivot_category,
        'pivot_label':     pivot_label,
    }
    for tid in tier_dfs:
        total_row[_col_name(tid)] = int(len(tier_valid[tid]))
    total_row['vg1'] = int(len(vg1_valid))
    total_row['vg2'] = int(len(vg2_valid))
    rows.append(total_row)

    final_vc = tier_vc[final_tier_id]
    all_labels = final_vc.index.union(lit_vc.index)
    for label in sorted(all_labels, key=lambda l: -final_vc.get(l, 0)):
        row = {
            'label':          label,
            'literature':     int(lit_vc.get(label, 0)),
            'pivot_category': pivot_category,
            'pivot_label':    pivot_label,
        }
        for tid in tier_dfs:
            row[_col_name(tid)] = int(tier_vc[tid].get(label, 0))
        row['vg1'] = int(vg1_vc.get(label, 0))
        row['vg2'] = int(vg2_vc.get(label, 0))
        rows.append(row)

    return rows


def write_label_distribution_summary(lit_domains, tier_dfs, final_tier_id, categories, out_dir, pivot_category=None):
    """
    label, literature, <tier columns in cascade order e.g. hc70, hc50>,
    expanded (the final tier), vg1, vg2, pivot_category, pivot_label —
    one "<category>_total" row per category (counts of non-missing values)
    followed by one row per class label present in the final tier, ordered
    by final-tier frequency, stepping through categories. pivot_category/
    pivot_label are None for this top-level, unfiltered section.

    tier_dfs: dict of cutoff_id -> dataframe of representatives at that
              tier (subset of combined_df), in cascade order (e.g.
              {"hc70": df70, "hc50": df50, "hc30": df30}) — every tier
              gets its own column, except final_tier_id which is labeled
              "expanded" (vg1/vg2 are also computed from that tier).

    pivot_category: optional category (e.g. "category_3") whose label
              values partition the dataset. When set, for every label of
              that category the same per-category breakdown above is
              recomputed on that label's subset for every OTHER category
              and appended to the bottom — a cross-tab of pivot label vs.
              every other category's label distribution — tagged via the
              pivot_category/pivot_label columns so it's easy to filter in
              a spreadsheet.
    """
    final_df = tier_dfs[final_tier_id]

    rows = []
    for category in categories:
        rows.extend(_category_count_rows(lit_domains, tier_dfs, final_tier_id, category))

    if pivot_category:
        pivot_labels = sorted(final_df[pivot_category].dropna().unique())
        other_categories = [c for c in categories if c != pivot_category]
        for pivot_label in pivot_labels:
            lit_sub  = lit_domains[lit_domains[pivot_category] == pivot_label]
            tier_sub = {tid: df[df[pivot_category] == pivot_label] for tid, df in tier_dfs.items()}
            for other_category in other_categories:
                rows.extend(_category_count_rows(
                    lit_sub, tier_sub, final_tier_id, other_category,
                    pivot_category=pivot_category, pivot_label=pivot_label,
                ))

    df = pd.DataFrame(rows)
    output_path = os.path.join(out_dir, "label_distribution.csv")
    df.to_csv(output_path, index=False)
    print(f"  ✓ label_distribution.csv written")
    return df


def gather_blast_domains(query_dir_base):
    """
    Load all per-query filtered domain CSVs written by search_substage and
    standardize their columns to match
    seq_utils.literature_domains_with_categories' schema (domain_id,
    orf_id, orf_sequence, domain_sequence, categories), keeping the
    query/c-value/walkerA_score provenance columns too.
    """
    all_domains   = []
    ready_count   = 0
    skipped_count = 0

    for qdir in sorted(os.listdir(query_dir_base)):
        dirpath = os.path.join(query_dir_base, qdir)
        if not os.path.isdir(dirpath):
            continue

        filtered_csv = os.path.join(dirpath, f"{qdir}_domains_filtered.csv")
        if not os.path.exists(filtered_csv):
            skipped_count += 1
            continue

        df = pd.read_csv(filtered_csv)
        if len(df) == 0:
            skipped_count += 1
            continue

        df = df.rename(columns={
            'domain_identifier': 'domain_id',
            'blast_query':       'query',
            'target_name':       'orf_id',
            'sequence':          'orf_sequence',
        })

        required = {'domain_id', 'domain_sequence', 'orf_id', 'orf_sequence'}
        if not required.issubset(df.columns):
            skipped_count += 1
            print(f"    Skipping {qdir}: missing required columns ({required - set(df.columns)})")
            continue

        all_domains.append(df)
        ready_count += 1

    combined = pd.concat(all_domains, ignore_index=True)
    combined = combined.drop_duplicates(subset=['domain_id'])

    print(f"  Loaded {ready_count} queries, skipped {skipped_count} (no filtered CSV found)")
    print(f"  ✓ {len(combined)} blast domains gathered (deduped)")
    return combined


def write_expanded_fastas(final_df, out_dir):
    """
    Writes domain- and ORF-grain FASTA files for the final (e.g. hc30) tier
    of expanded_dataset.csv — domain_id/domain_sequence as-is (already
    unique), orf_id/orf_sequence deduplicated by orf_id (a multi-domain ORF
    only needs one alignment-input copy of its full sequence).
    """
    domains_fasta = os.path.join(out_dir, "expanded_domains.fasta")
    io_utils.write_fasta(domains_fasta, final_df['domain_id'], final_df['domain_sequence'])
    print(f"  ✓ {domains_fasta} written: {len(final_df)} domains")

    orf_df = final_df.drop_duplicates(subset='orf_id')
    orfs_fasta = os.path.join(out_dir, "expanded_orfs.fasta")
    io_utils.write_fasta(orfs_fasta, orf_df['orf_id'], orf_df['orf_sequence'])
    print(f"  ✓ {orfs_fasta} written: {len(orf_df)} unique ORFs")

    return domains_fasta, orfs_fasta


def combine_substage(paths, params, hpc):
    """
    Gathers all per-query filtered domain CSVs (search substage) plus the
    literature domains into one combined pool (written out in full), then
    runs a cascade of MMseqs clustering rounds (e.g. 70% -> 50% -> 30%),
    each round operating on the PREVIOUS round's representative set only
    (not the full pool). This avoids connected-component clustering's
    tendency to transitively chain distant sequences together into a few
    giant clusters when run once on a huge, dense pool — pre-thinning at
    stricter cutoffs first removes most of the "bridge" sequences that
    make long chains possible. stratified_homology_filter (literature
    priority + category-label balancing) picks representatives at every
    round. The final round's representatives are written as
    expanded_dataset.csv.
    """
    literature_path = paths["updated_literature"]
    query_dir_base   = paths["expand_queries"]
    combine_dir      = paths["expand_combine_dir"]
    out_dir          = paths["expand_dir"]
    os.makedirs(combine_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    homology_cutoffs    = params["homology_cutoffs"]
    mmseqs_settings      = params["mmseqs_settings"]
    stratify_category    = params["stratify_category"]
    pivot_category        = params.get("label_distribution_pivot_category")
    categories            = list(config.CATEGORIES.keys())
    annotation_cols        = categories + ["validation_group"]

    lit_df = pd.read_csv(literature_path)

    print(f"\n  Generating literature domains...")
    lit_domains = seq_utils.literature_domains_with_categories(lit_df, annotation_cols)
    lit_domains['query'] = None
    lit_domains['orf_sequence'] = lit_domains['orf_id'].map(lit_df.set_index('orf_id')['orf_sequence'])
    print(f"  ✓ {len(lit_domains)} literature domains")

    print(f"\n  Gathering BLAST-derived domains from query directories...")
    blast_domains = gather_blast_domains(query_dir_base)

    # literature domains take priority on exact domain_id collisions
    combined_df = pd.concat([lit_domains, blast_domains], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=['domain_id'], keep='first')
    print(f"\n  Total combined domains (pre-clustering): {len(combined_df)} "
          f"({len(lit_domains)} literature + {len(blast_domains)} blast, after domain_id dedup)")

    combined_csv = os.path.join(out_dir, "combined_domains_all.csv")
    combined_df.to_csv(combined_csv, index=False)
    print(f"  ✓ {combined_csv} written: {len(combined_df)} domains (full combined pool, pre-clustering)")

    priority_ids   = set(lit_domains['domain_id'])
    label_map_full = dict(zip(combined_df['domain_id'], combined_df[stratify_category]))
    vg_map_full    = dict(zip(combined_df['domain_id'], combined_df['validation_group']))

    current_ids   = set(combined_df['domain_id'])
    current_fasta = os.path.join(combine_dir, "round0_input.fasta")
    io_utils.write_fasta(current_fasta, combined_df['domain_id'], combined_df['domain_sequence'])

    final_cutoff_id = None
    tier_ids = {}  # cutoff_id -> representative domain_ids at that tier (for label_distribution.csv)
    for cutoff in sorted(homology_cutoffs, reverse=True):
        cutoff_id = f"hc{int(cutoff * 100)}"
        print(f"\n  Clustering at {cutoff_id} ({cutoff}) — input: {len(current_ids)} sequences...")
        _, cluster_tsv, _ = subprocesses.mmseqs_ez_cluster(
            current_fasta,
            os.path.join(combine_dir, cutoff_id),
            cutoff,
            mmseqs_settings,
            hpc,
        )
        cluster_tsv_path = os.path.join(combine_dir, cluster_tsv)
        n_clusters = pd.read_csv(cluster_tsv_path, sep='\t', names=['representative', 'member'], header=None)['representative'].nunique()
        print(f"  ✓ {n_clusters} clusters")

        round_label_map    = {i: label_map_full[i] for i in current_ids}
        round_vg_map       = {i: vg_map_full[i] for i in current_ids}
        round_priority_ids = priority_ids & current_ids

        print(f"  Selecting representatives (stratified on {stratify_category}, "
              f"literature priority, validation_group tie-break)...")
        rep_df = stratified_homology_filter(cluster_tsv_path, round_label_map, stratify_category,
                                             priority_ids=round_priority_ids, vg_map=round_vg_map)

        current_ids = set(rep_df['domain_id'])
        final_cutoff_id = cutoff_id
        tier_ids[cutoff_id] = current_ids
        print(f"  {cutoff_id}: {len(current_ids)} representatives")

        current_fasta = os.path.join(combine_dir, f"{cutoff_id}_input_for_next.fasta")
        next_df = combined_df[combined_df['domain_id'].isin(current_ids)]
        io_utils.write_fasta(current_fasta, next_df['domain_id'], next_df['domain_sequence'])

    final_df = combined_df[combined_df['domain_id'].isin(current_ids)].copy()
    final_df['homology_group'] = final_cutoff_id

    output_csv = os.path.join(out_dir, "expanded_dataset.csv")
    final_df.to_csv(output_csv, index=False)

    lit_kept = final_df['domain_id'].isin(priority_ids).sum()
    print(f"\n  ✓ expanded_dataset.csv written: {len(final_df)} representative domains "
          f"({lit_kept}/{len(lit_domains)} literature domains preserved)")

    tier_dfs = {tid: combined_df[combined_df['domain_id'].isin(ids)] for tid, ids in tier_ids.items()}
    write_label_distribution_summary(lit_domains, tier_dfs, final_cutoff_id, categories, out_dir,
                                      pivot_category=pivot_category)

    print(f"\n  Writing expanded FASTA files ({final_cutoff_id} tier)...")
    domains_fasta, orfs_fasta = write_expanded_fastas(final_df, out_dir)

    if params.get("run_mafft", True):
        mafft_settings = params.get("mafft_settings", {"maxiterate": 1000, "retree": 3, "threads": 20})
        domains_aln = os.path.join(out_dir, "aligned_domains.aln")
        orfs_aln    = os.path.join(out_dir, "aligned_orfs.aln")
        print(f"\n  Aligning expanded domains (mafft --globalpair) and "
              f"ORFs (mafft --localpair) in parallel "
              f"({mafft_settings.get('threads', 1)} threads each)...")
        domains_proc = subprocesses.start_mafft(domains_fasta, domains_aln, "globalpair", hpc, **mafft_settings)
        orfs_proc    = subprocesses.start_mafft(orfs_fasta, orfs_aln, "localpair", hpc, **mafft_settings)
        subprocesses.finish_mafft(domains_proc, domains_aln)
        subprocesses.finish_mafft(orfs_proc, orfs_aln)
    else:
        print(f"\n  [combine] run_mafft=False, skipping alignment")


def search_substage(paths, params, hpc):
    """
    For each ORF in updated_literature.csv (written by the analyze stage —
    this is where validation_group comes from):
    - runs BLAST search (with resume check)
    - filters BLAST hits
    - runs HMMer scan
    - extracts and filters domains
    Then writes the filtering summary.
    """
    literature_path = paths["updated_literature"]
    msa_path        = paths["msa"]
    query_dir_base  = paths["expand_queries"]
    hmm_dir         = paths["expand_hmm_dir"]
    out_dir         = paths["expand_dir"]
    os.makedirs(query_dir_base, exist_ok=True)
    os.makedirs(hmm_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    blast_settings = params["blast_settings"]
    blast_filter   = params["blast_filter_params"]
    hmm_settings   = params["hmm_settings"]
    skip_hmmer     = params.get("skip_hmmer", False)
    categories     = list(config.CATEGORIES.keys())
    annotation_cols = categories + ["validation_group"]

    lit_df        = pd.read_csv(literature_path)
    total_queries = len(lit_df)
    print(f"\n  Found {total_queries} queries to process\n")

    hmm_profile = subprocesses.build_hmm_profile(msa_path, hmm_dir, hpc)

    all_stats = []

    for idx, query_row in lit_df.iterrows():
        orf_id    = query_row['orf_id']
        query_dir = os.path.join(query_dir_base, orf_id)
        os.makedirs(query_dir, exist_ok=True)

        print(f"\n  [{idx+1}/{total_queries}] {orf_id}")

        # --- BLAST ---
        if io_utils.complete_checker(query_dir, orf_id):
            print(f"    ✓ BLAST already complete, skipping...")
        else:
            fasta_path = os.path.join(query_dir, f"{orf_id}.fasta")
            io_utils.write_fasta(fasta_path, [orf_id], [query_row['orf_sequence']])
            success = subprocesses.run_blast(orf_id, query_dir, blast_settings, hpc)
            if not success:
                print(f"    ⚠️  BLAST failed, skipping {orf_id}")
                continue

        # --- filter BLAST hits ---
        blast_stats = filter_blast_hits(orf_id, query_dir, blast_filter)
        print(f"    BLAST: {blast_stats['blast_total']} hits → {blast_stats['final_sequences']} after filtering")

        # --- prepare HMMer input ---
        input_fasta = prepare_hmmer_input(orf_id, query_dir)

        # --- run HMMer ---
        domtblout_path = os.path.join(query_dir, f"{orf_id}.domtblout")
        if skip_hmmer and os.path.exists(domtblout_path):
            print(f"    ✓ HMMer skipped, reusing cached {orf_id}.domtblout")
        else:
            success = subprocesses.run_hmmer(orf_id, query_dir, input_fasta, hmm_profile, hmm_settings, hpc)
            if not success:
                print(f"    ⚠️  HMMer failed, skipping {orf_id}")
                continue

        # --- extract and filter domains ---
        extract_domains_from_hmm_hits(orf_id, query_dir, input_fasta, query_row, annotation_cols)
        domain_df, domain_stats = filter_domains(orf_id, query_dir, params)
        print(f"    Domains: {domain_stats['domains_start']} → {domain_stats['domains_final']} after filtering")

        all_stats.append({**blast_stats, **{k: v for k, v in domain_stats.items() if k != 'identifier'}})

    print(f"\n  Writing filtering summary...")
    write_filtering_summary(all_stats, out_dir)


# ===== visualize substage =====

def _plot_lda_3d_reference(pdf, X_ref, X_query, query_labels, color_palette, title):
    """
    3D LDA scatter with two populations: X_ref (literature — the data the
    projection was fit on) always plotted in solid black, and X_query
    (non-query expanded-dataset domains) colored by query_labels via
    color_palette, plotted on top.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    def _z(X):
        return X[:, 2] if X.shape[1] > 2 else np.zeros(X.shape[0])

    ax.scatter(X_ref[:, 0], X_ref[:, 1], _z(X_ref),
               label="literature", color="black", s=20, alpha=0.6)

    present = set(pd.unique(query_labels))
    ordered_labels = [l for l in color_palette if l in present]
    ordered_labels += [l for l in present if l not in color_palette]

    z_query = _z(X_query)
    for label in ordered_labels:
        mask = query_labels == label
        color = color_palette.get(label, color_palette.get("unknown", "gray"))
        ax.scatter(X_query[mask, 0], X_query[mask, 1], z_query[mask],
                   label=str(label), color=color, s=30, alpha=0.8)

    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)


def visualize_substage(paths, params, hpc):
    """
    Fits a PCA+LDA projection on literature domain embeddings for
    `lda_category` (default: mechanism), then applies that SAME fitted
    projection to the non-query (BLAST-derived) domains in
    expanded_dataset.csv. Each layer gets a paired pair of pages sharing
    the identical fitted axes: literature-only (colored by
    `color_category`), then literature (black, reference) + expanded
    (colored) — so the two are directly, visually comparable. A fixed
    list of layers, not a best-layer search — this is a qualitative check
    of whether homology-expanded domains land where their inherited label
    predicts, not a statistical trajectory analysis (that's analyze.py's
    job). Only `embedding_source` ("domain_sequence" for now) is used.

    `layer_ranges` (list of [start, end] inclusive pairs) adds extra pages
    per range, using pooled (concatenated) features across that range
    instead of a single layer — otherwise identical treatment to a single
    layer's page (same PCA fit-on-literature/apply-to-expanded + LDA).
    """
    esm_settings     = config.PARAMS["inference_settings"]
    categories       = list(config.CATEGORIES.keys())
    embedding_source = params.get("embedding_source", "domain_sequence")
    layers           = params.get("layers", [2, 12, 22, 32])
    layer_ranges     = params.get("layer_ranges", [])
    lda_category     = params.get("lda_category", "category_1")
    color_category   = params.get("color_category", "category_1")
    color_palette    = params.get("color_palette", {})

    out_dir = paths["expand_dir"]
    os.makedirs(out_dir, exist_ok=True)

    lit_df = pd.read_csv(paths["updated_literature"])
    lit_domains = seq_utils.literature_domains_with_categories(lit_df, categories)
    print(f"[expand.visualize] {len(lit_domains)} literature domains")

    expanded_df = pd.read_csv(os.path.join(out_dir, "expanded_dataset.csv"))
    non_query_df = expanded_df[expanded_df["query"].notna()].copy()
    print(f"[expand.visualize] {len(non_query_df)} non-query (BLAST-derived) expanded domains")

    embed_dir = os.path.join(paths["embeddings"], embedding_source)

    all_layers = sorted(set(layers) | {l for lo, hi in layer_ranges for l in range(lo, hi + 1)})

    # Literature embeddings: already cached under cache_key "analyze" (all
    # 33 layers) from the analyze stage — reuse that cache directly.
    lit_ids = lit_domains["domain_id"].tolist()
    lit_seq_dict = dict(zip(lit_domains["domain_id"], lit_domains[embedding_source]))
    inf.prepare_embeddings(
        lit_ids, lit_seq_dict, paths, esm_settings, hpc, all_layers,
        embed_dir=embed_dir, cache_key="analyze",
    )
    lit_cache = FeatureCache(os.path.join(embed_dir, "csv", "analyze"), lit_ids, all_layers)

    # Non-query expanded domains: new embeddings, own cache_key (.pt files
    # are still shared globally by domain_id, so the ~56 literature domains
    # already present in expanded_dataset.csv cost nothing extra here since
    # they're excluded from non_query_df anyway).
    exp_ids = non_query_df["domain_id"].tolist()
    exp_seq_dict = dict(zip(non_query_df["domain_id"], non_query_df[embedding_source]))
    inf.prepare_embeddings(
        exp_ids, exp_seq_dict, paths, esm_settings, hpc, all_layers,
        embed_dir=embed_dir, cache_key="expand",
    )
    exp_cache = FeatureCache(os.path.join(embed_dir, "csv", "expand"), exp_ids, all_layers)

    lit_lda_labels   = lit_domains[lda_category].values
    lit_color_labels = pd.Series(lit_domains[color_category].values).fillna("unknown").values
    exp_color_labels = pd.Series(non_query_df[color_category].values).fillna("unknown").values

    lda_name   = config.CATEGORIES.get(lda_category, {}).get("name", lda_category)
    color_name = config.CATEGORIES.get(color_category, {}).get("name", color_category)

    valid_lit       = pd.notna(lit_lda_labels)
    y_lit_valid     = lit_lda_labels[valid_lit]
    lit_color_valid = lit_color_labels[valid_lit]

    def _pooled(cache, group_layers):
        arrays = [cache.get_all(l) for l in group_layers]
        if any(a is None for a in arrays):
            return None
        return arrays[0] if len(arrays) == 1 else np.concatenate(arrays, axis=1)

    groups  = [([layer], f"layer {layer}") for layer in layers]
    groups += [(list(range(lo, hi + 1)), f"layers {lo}-{hi} (pooled)") for lo, hi in layer_ranges]

    pdf_path = os.path.join(out_dir, "expand_lda_visualizations.pdf")
    with PdfPages(pdf_path) as pdf:
        for group_layers, label in groups:
            X_lit = _pooled(lit_cache, group_layers)
            X_exp = _pooled(exp_cache, group_layers)
            if X_lit is None or X_exp is None:
                print(f"  [expand.visualize] Skipping {label}: missing cached embeddings")
                continue

            X_lit_valid = X_lit[valid_lit]

            # Fit ONCE per group — both pages below reuse this exact fitted
            # scaler/pca/lda, so their axes are identical and comparable.
            scaler, pca, X_lit_pca = lda_utils.pca_fit(X_lit_valid)
            X_lit_lda, lda_model = lda_utils.fit_lda_projection(X_lit_pca, y_lit_valid, n_components=3)

            if X_lit_lda.shape[1] < 2:
                print(f"  [expand.visualize] Skipping {label}: only {X_lit_lda.shape[1]} LD axis available")
                continue

            title_lit = f"{lda_name} LDA space (literature only), colored by {color_name} — {label}"
            lda_utils.plot_lda_3d(pdf, X_lit_lda, lit_color_valid, color_palette, title_lit)
            print(f"  [expand.visualize] Added PDF page: {title_lit}")

            X_exp_pca = lda_utils.pca_apply(scaler, pca, X_exp)
            X_exp_lda = lda_model.transform(X_exp_pca)

            title_exp = (f"{lda_name} LDA space (fit on literature), "
                         f"colored by {color_name} — {label}")
            _plot_lda_3d_reference(pdf, X_lit_lda, X_exp_lda, exp_color_labels, color_palette, title_exp)
            print(f"  [expand.visualize] Added PDF page: {title_exp}")

    print(f"[expand.visualize] Wrote {pdf_path}")


def run(paths, params, hpc, substages=None):
    if substages is None:
        substages = config.STAGES["expand"].get("substages", {})

    if substages.get("search", False):
        print("\n[expand] Running search substage...")
        search_substage(paths, params["search"], hpc)
        print("\n[expand] Search substage complete.")

    if substages.get("combine", False):
        print("\n[expand] Running combine substage...")
        combine_substage(paths, params["combine"], hpc)
        print("\n[expand] Combine substage complete.")

    if substages.get("visualize", False):
        print("\n[expand] Running visualize substage...")
        visualize_substage(paths, params["visualize"], hpc)
        print("\n[expand] Visualize substage complete.")
