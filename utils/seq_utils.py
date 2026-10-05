# seq_utils.py
"""
Sequence manipulation shared across stages. Domain sequences are never
persisted on their own (literature.csv only stores orf_sequence + bounds) —
generate_domain_sequences is the single place that extracts them, so
analyze.py and (later) expand.py don't duplicate the extraction logic.
"""
import pandas as pd


def split_bounds(bounds_string):
    """Split a comma-separated 'start-end,start-end' bounds string into a list of range strings."""
    return [b.strip() for b in bounds_string.split(',')]


def domain_id_for(orf_id, bounds):
    return f"{orf_id}_{bounds}"


def extract_domain(full_sequence, boundary_string):
    """Extract a domain from a sequence using a 'start-end' boundary string (1-based, inclusive)."""
    start, end = map(int, boundary_string.strip().split('-'))
    return full_sequence[start - 1:end]


def remove_overlapping_domains(domain_df):
    """
    Remove overlapping HMM domain hits from the same gene, keeping the
    hit with the better (lower) c-value. Only needed for whole-proteome
    scanning (survey.py) — a single gene can have multiple redundant/
    overlapping HMM hits there, which expand.py's per-query BLAST/HMMer
    flow essentially never sees.
    """
    domain_df = domain_df.sort_values(['target_name', 'env_from', 'c-value']).reset_index(drop=True)

    keep_mask = [True] * len(domain_df)

    for gene, gene_df in domain_df.groupby('target_name'):
        indices = gene_df.index.tolist()

        for i in range(len(indices)):
            if not keep_mask[indices[i]]:
                continue

            idx_i = indices[i]
            end_i = domain_df.loc[idx_i, 'env_to']

            for j in range(i + 1, len(indices)):
                idx_j = indices[j]
                if not keep_mask[idx_j]:
                    continue

                start_j = domain_df.loc[idx_j, 'env_from']

                # c-value already sorted, so idx_i has the better c-value
                if start_j <= end_i:
                    keep_mask[idx_j] = False

    return domain_df[keep_mask].reset_index(drop=True)


def score_walkerA(sequence):
    """Score Walker A motif. Returns highest score found."""
    max_score = 0

    for i, residue in enumerate(sequence):
        if residue != 'K':
            continue

        score = 0

        # Check positions relative to K
        if i > 0 and sequence[i - 1] == 'G':
            score += 10
        if i > 0 and sequence[i - 1] == 'A':
            score += 8
        if i < len(sequence) - 1 and sequence[i + 1] in ('T', 'S'):
            score += 5
        if i > 5 and sequence[i - 6] == 'G':
            score += 10
        if i > 2 and sequence[i - 3] == 'G':
            score += 3
        if i > 5 and sequence[i - 6] == 'S':
            score += 3
        if i > 3 and sequence[i - 4] == 'P':
            score += 3

        if score > max_score:
            max_score = score

    return max_score


def generate_domain_sequences(lit_df):
    """
    Build a domain_id -> domain_sequence dataframe from literature.csv's
    orf_sequence + domain_bounds columns. One row per bounds entry (an ORF
    with multiple domain_bounds ranges yields multiple rows).
    """
    rows = []
    for _, row in lit_df.iterrows():
        for bounds in split_bounds(row['domain_bounds']):
            rows.append({
                'domain_id':       domain_id_for(row['orf_id'], bounds),
                'domain_sequence': extract_domain(row['orf_sequence'], bounds),
            })
    return pd.DataFrame(rows)


def literature_domains_with_categories(lit_df, categories):
    """
    Per-domain rows from literature.csv: one row per domain_bounds entry,
    carrying orf_id + categories (inherited from the parent ORF row),
    joined against generate_domain_sequences for the actual
    domain_sequence values. Shared by analyze.py and expand.py's combine
    substage so this logic lives in exactly one place.
    """
    rows = []
    for _, row in lit_df.iterrows():
        for bounds in split_bounds(row['domain_bounds']):
            entry = {
                'domain_id': domain_id_for(row['orf_id'], bounds),
                'orf_id':    row['orf_id'],
            }
            for cat in categories:
                entry[cat] = row.get(cat)
            rows.append(entry)
    expanded_df = pd.DataFrame(rows)

    domain_seq_df = generate_domain_sequences(lit_df)
    return expanded_df.merge(domain_seq_df, on='domain_id', how='left')
