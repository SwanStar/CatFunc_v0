import os
import subprocess

def run_blast(identifier, query_dir, blast_settings, hpc):
    """Run BLAST search for a single query."""
    query_fasta  = os.path.join(query_dir, f"{identifier}.fasta")
    output_tsv   = os.path.join(query_dir, f"{identifier}_BLAST.tsv")
    output_fasta = os.path.join(query_dir, f"{identifier}_BLAST.fasta")

    database_path = blast_settings["database"]
    hit_limit     = blast_settings["hit_limit"]
    num_threads   = blast_settings["num_threads"]
    module_load   = "\n".join(hpc["module_cmd"])

    command = f'''
{module_load}
export BLASTDB={database_path}
blastp -query {query_fasta} -db nr -out {output_tsv} -outfmt 6 -num_threads {num_threads} -max_target_seqs {hit_limit}
cut -f 2 {output_tsv} | blastdbcmd -db nr -entry_batch - -out {output_fasta}
'''
    try:
        subprocess.run(command, shell=True, executable='/bin/bash')
        if os.path.exists(output_fasta) and os.path.getsize(output_fasta) > 0:
            completion_file = os.path.join(query_dir, f"{identifier}_COMPLETE")
            with open(completion_file, 'w') as f:
                f.write("BLAST completed successfully\n")
            return True
        else:
            print(f"    ⚠️  No sequences retrieved for {identifier}")
            return False
    except Exception as e:
        print(f"    ❌ BLAST failed for {identifier}: {e}")
        return False


def build_hmm_profile(msa_path, metadata_dir, hpc):
    """Build HMM profile from MSA. Skips if already exists."""
    msa_basename = os.path.splitext(os.path.basename(msa_path))[0]
    hmm_profile  = os.path.join(metadata_dir, f"{msa_basename}_profile.hmm")

    if os.path.exists(hmm_profile):
        print(f"  ✓ Using existing HMM profile: {msa_basename}_profile.hmm")
        return hmm_profile

    print(f"  Building HMM profile from {msa_basename}...")
    module_load = "\n".join(hpc["module_cmd"])
    command = f'''
{module_load}
hmmbuild {hmm_profile} {msa_path}
'''
    try:
        subprocess.run(command, shell=True, executable='/bin/bash', check=True)
        print(f"  ✓ HMM profile built successfully")
        return hmm_profile
    except subprocess.CalledProcessError as e:
        print(f"  ❌ hmmbuild failed: {e}")
        raise


def mmseqs_ez_cluster(fasta_file, output_id, homology_cutoff, mmseqs_settings, hpc, verbose=False, tmp_dir=None):
    """
    Runs the mmseqs easy-cluster function.
    Returns absolute paths to rep_seq, cluster tsv, and all_seqs output files.
    """
    parent_dir          = os.path.dirname(fasta_file)
    rep_seqs_fasta_path = f"{output_id}_rep_seq.fasta"
    cluster_tsv_path    = f"{output_id}_cluster.tsv"
    all_seqs_fasta_path = f"{output_id}_all_seqs.fasta"

    if tmp_dir is None:
        tmp_dir = f"{output_id}_tmp"

    cluster_mode = mmseqs_settings.get('cluster_mode', 1)
    threads      = mmseqs_settings.get('num_threads', 1)
    k            = mmseqs_settings.get('k', 5)
    module_load  = "\n".join(hpc["module_cmd"])

    command = f'''
{module_load}
mmseqs easy-cluster {fasta_file} {output_id} {tmp_dir} \
    --cluster-mode {cluster_mode} \
    --min-seq-id {homology_cutoff} \
    --threads {threads} \
    -k {k}
'''
    if verbose:
        subprocess.run(command, shell=True, executable='/bin/bash')
    else:
        log_file = os.path.join(parent_dir, f"{output_id}.log")
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, 'w') as log:
            subprocess.run(command, shell=True, executable='/bin/bash',
                           stdout=log, stderr=log)

    return rep_seqs_fasta_path, cluster_tsv_path, all_seqs_fasta_path


def start_mafft(input_fasta, output_aln, pair_mode, hpc, maxiterate=1000, retree=3, threads=1):
    """
    Launch MAFFT multiple sequence alignment as a background subprocess and
    return its Popen handle immediately (does not block) — lets the caller
    launch several mafft runs in parallel, then wait on all of them via
    finish_mafft. pair_mode: 'globalpair' (G-INS-i, for domain-length
    sequences with similar overall length) or 'localpair' (L-INS-i, for
    full-length ORFs where only a sub-region may align well).

    Returns None (nothing to wait on) if output_aln already exists and is
    non-empty.
    """
    if os.path.exists(output_aln) and os.path.getsize(output_aln) > 0:
        print(f"    ✓ Using existing alignment: {os.path.basename(output_aln)}")
        return None

    module_load = "\n".join(hpc["module_cmd"])
    log_file    = f"{output_aln}.log"

    command = f'''
{module_load}
mafft --thread {threads} --maxiterate {maxiterate} --{pair_mode} --retree {retree} {input_fasta} > {output_aln} 2> {log_file}
'''
    return subprocess.Popen(command, shell=True, executable='/bin/bash')


def finish_mafft(proc, output_aln):
    """Wait for a start_mafft() process (if any) and verify its output."""
    if proc is None:
        return True
    proc.wait()
    if os.path.exists(output_aln) and os.path.getsize(output_aln) > 0:
        print(f"    ✓ mafft alignment written: {output_aln}")
        return True
    else:
        print(f"    ⚠️  mafft produced no output: {output_aln}")
        return False


def run_hmmer(identifier, query_dir, input_fasta, hmm_profile, hmm_settings, hpc):
    """Run HMMer search for a single query."""
    output_domtblout = os.path.join(query_dir, f"{identifier}.domtblout")
    output_log       = os.path.join(query_dir, f"{identifier}_hmmsearch.log")
    num_threads      = hmm_settings["num_threads"]
    module_load      = "\n".join(hpc["module_cmd"])

    command = f'''
{module_load}
hmmsearch --cpu {num_threads} --domtblout {output_domtblout} {hmm_profile} {input_fasta} > {output_log} 2>&1
'''
    try:
        subprocess.run(command, shell=True, executable='/bin/bash', check=True)
        if os.path.exists(output_domtblout) and os.path.getsize(output_domtblout) > 0:
            return True
        else:
            print(f"    ⚠️  No HMM hits found for {identifier}")
            return False
    except subprocess.CalledProcessError as e:
        print(f"    ❌ hmmsearch failed for {identifier}: {e}")
        return False
