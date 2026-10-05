import os
import pandas as pd

#=========================================================
# i/o helpers
#=========================================================

def _sanitize_sequence(sequence):
    """Replace non-standard IUPAC residue codes with closest chemical equivalent."""
    _SUB_MAP = {
        'B': 'D',   # Asx (Asp/Asn) → Asp (more abundant)
        'Z': 'E',   # Glx (Glu/Gln) → Glu (more abundant)
        'J': 'L',   # Xle (Leu/Ile) → Leu (more abundant)
        'U': 'C',   # Selenocysteine → Cysteine (structurally closest)
        'O': 'K',   # Pyrrolysine → Lysine (structurally closest)
    }
    return ''.join(_SUB_MAP.get(ch, ch) for ch in sequence)


def write_fasta(filename, headers, sequences):
    """
    Write sequences to a FASTA file. Sanitizes non-standard residues.
    Args:
        filename: Path to output FASTA file
        headers: List or Series of sequence headers
        sequences: List or Series of sequences
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    bad_count = 0
    with open(filename, 'w') as fasta_file:
        for header, sequence in zip(headers, sequences):
            original = sequence
            sequence = _sanitize_sequence(sequence)
            if sequence != original:
                bad_count += 1
            fasta_file.write(f">{header}\n{sequence}\n")
    if bad_count:
        print(f"  [write_fasta] Replaced non-standard residues in {bad_count}/{len(sequences)} sequences")

def read_fasta(fasta_file, truncate_header=False):
    """
    convert a fasta file to a dataframe with 'header' and 'sequence' columns
    optionally, set truncate_header to True and it will put everything after
    the first space in the header into a third column called 'header_info'
    (truncate_header=True is designed for blast hits)
    """
    target_name = None
    headers = []
    header_info = []
    sequences = []

    with open(fasta_file, 'r') as fasta:
        for line in fasta:

            if line.startswith('>'):
                # Save previous entry if it exists
                if target_name is not None:
                    headers.append(target_name.strip())
                    sequences.append(sequence)

                sequence = ""

                if truncate_header:
                    header_parts = line.split()
                    target_name = header_parts[0][1:]
                    info = ' '.join(header_parts[1:])
                    header_info.append(info)
                else:
                    target_name = line[1:].strip()
            else:
                sequence += line.strip()  # Already stripped, no \n

        # Don't forget the last entry
        if target_name is not None:
            headers.append(target_name.strip())
            sequences.append(sequence.strip())

    # Create DataFrame with appropriate columns
    if truncate_header:
        fasta_df = pd.DataFrame({
            'header': headers,
            'header_info': header_info,
            'sequence': sequences
        })
    else:
        fasta_df = pd.DataFrame({
            'header': headers,
            'sequence': sequences
        })

    return fasta_df

def read_blast_tsv(blast_output_tsv_file):

    blast_columns = ['query','target_name','homology_score','alignment_length',
                     'mismatches','gap_openings','query_start','query_end',
                     'subject_start','subject_end','evalue','bitscore']

    blast_tsv_df = pd.read_csv(blast_output_tsv_file,
                                sep='\t',
                                header=None,
                                names= blast_columns)
    return blast_tsv_df

def read_hmmsearch_domtblout(hmmsearch_domtblout):
    rows = []
    with open(hmmsearch_domtblout, 'r') as domtblout:
        for line in domtblout:
            if line.startswith('#'):
                continue
            line_parts = line.strip().split()
            rows.append([line_parts[0],line_parts[11],line_parts[12],line_parts[13],line_parts[19],line_parts[20]])
    hmm_df = pd.DataFrame(rows, columns=['target_name','c-value', 'i_value', 'score','env_from', 'env_to'])

    return hmm_df

#=========================================================
# directory helpers
#=========================================================

def complete_checker(query_dir, identifier):
    """
    Check if a completeness marker exists for a query-based process.
    """
    complete_marker = os.path.join(query_dir, f"{identifier}_COMPLETE")

    if not os.path.exists(complete_marker):
        return False

    return True
