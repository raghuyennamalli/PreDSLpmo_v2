#!/usr/bin/env python3
"""
SCENARIO 1 - own family:
    AA9.fasta       -> LPMO_AA9.fasta
    AA10.fasta      -> LPMO_AA10.fasta
    ... (AA11, AA13, AA14, AA15, AA16, AA17 the same way)
    Non_LPMO.fasta  -> all NON-LPMO_*_training.fasta files together

SCENARIO 2 - other families (everything in training EXCEPT the scenario-1 files):
    AA9.fasta       -> all other LPMO_AA*.fasta + all NON-LPMO_*_training.fasta
    ...
    Non_LPMO.fasta  -> all LPMO_AA*.fasta files

A sequence FAILS a scenario if it has any training hit with
    identity >= --pid  AND  query coverage >= --cov

FINAL independent set = sequences that PASS BOTH scenarios.
  
    python check_independent_vs_training.py --pid 70 --cov 70 
"""

import argparse
import glob
import os
import shutil
import subprocess
import sys

import pandas as pd

INDEP_DIR = "/media/ragothaman/yennamalli/Sweety/LPMO_NEW/New_Version/independent_dataset"
TRAIN_DIR = ("/media/ragothaman/yennamalli/Sweety/LPMO_NEW/New_Version/"
             "Cluster_Aware_Spliting_Changed/Training_Validation_Final/Training")
OUT_DIR = os.path.join(INDEP_DIR, "familywise_similarity_check")

BLAST_COLS = ["qid", "sid", "pident", "length", "evalue", "bitscore", "qlen", "slen", "qcovhsp"]
SCENARIOS = [("S1", "scenario1_own_family"), ("S2", "scenario2_other_families")]

def read_fasta(path):
    records, header, seq = [], None, []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if header is not None:
                    records.append((header, "".join(seq)))
                header, seq = line[1:].strip(), []
            else:
                seq.append(line)
    if header is not None:
        records.append((header, "".join(seq)))
    return records


def clean_seq(s):
    return s.upper().replace("*", "").replace("-", "").replace(" ", "")


def write_fasta(records, path):
    with open(path, "w") as out:
        for header, seq in records:
            out.write(f">{header}\n")
            for i in range(0, len(seq), 60):
                out.write(seq[i:i + 60] + "\n")


def label(path):
    """LPMO_AA9.fasta -> LPMO_AA9 ; NON-LPMO_AA1_training.fasta -> NON-LPMO_AA1 ; AA9.fasta -> AA9"""
    base = os.path.splitext(os.path.basename(path))[0]
    if base.lower().endswith("_training"):
        base = base[: -len("_training")]
    return base


def norm(s):
    return s.upper().replace("-", "_")


def own_family_files(indep_file, train_files):
    fam = norm(label(indep_file))
    if fam.startswith("NON_LPMO"):
        return [f for f in train_files if norm(label(f)).startswith("NON_LPMO")]
    return [f for f in train_files if norm(label(f)) == f"LPMO_{fam}"]


def run(cmd):
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL)


def blast_check(fam, tag, q_fa, tfs, tmp_dir, args):
    """BLAST query FASTA against training files tfs. Return DataFrame: one row per FAILED query
    (its highest-identity hit meeting the cutoffs), with training file/header attached."""
    t_records, sid2info = [], {}
    for tf in tfs:
        for h, s in read_fasta(tf):
            s = clean_seq(s)
            if s:
                sid = f"T{len(t_records) + 1:07d}"
                t_records.append((sid, s))
                sid2info[sid] = (label(tf), h)
    t_fa = os.path.join(tmp_dir, f"{fam}_{tag}_train.fasta")
    write_fasta(t_records, t_fa)
    db = os.path.join(tmp_dir, f"{fam}_{tag}_db")
    run(["makeblastdb", "-in", t_fa, "-dbtype", "prot", "-out", db])

    out_tsv = os.path.join(tmp_dir, f"{fam}_{tag}_blast.tsv")
    run(["blastp", "-query", q_fa, "-db", db, "-out", out_tsv,
         "-outfmt", "6 qseqid sseqid pident length evalue bitscore qlen slen qcovhsp",
         "-evalue", str(args.evalue), "-max_target_seqs", "1000", "-max_hsps", "1",
         "-num_threads", str(args.threads)])
    hits = (pd.read_csv(out_tsv, sep="\t", names=BLAST_COLS)
            if os.path.getsize(out_tsv) > 0 else pd.DataFrame(columns=BLAST_COLS))

    bad = hits[(hits["pident"] >= args.pid) & (hits["qcovhsp"] >= args.cov)]
    bad = bad.sort_values(["qid", "pident"], ascending=[True, False]).drop_duplicates("qid").copy()
    bad["t_file"] = bad["sid"].map(lambda s: sid2info[s][0])
    bad["t_header"] = bad["sid"].map(lambda s: sid2info[s][1])
    return bad

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--indep_dir", default=INDEP_DIR)
    ap.add_argument("--train_dir", default=TRAIN_DIR)
    ap.add_argument("--out_dir", default=OUT_DIR)
    ap.add_argument("--pid", type=float, default=30.0, help="identity cutoff %% (default 30)")
    ap.add_argument("--cov", type=float, default=70.0, help="query coverage cutoff %% (default 70)")
    ap.add_argument("--evalue", type=float, default=1e-3)
    ap.add_argument("--threads", type=int, default=os.cpu_count() or 4)
    args = ap.parse_args()

    for prog in ("makeblastdb", "blastp"):
        if shutil.which(prog) is None:
            sys.exit(f"ERROR: '{prog}' not found in PATH.")

    indep_files = sorted(glob.glob(os.path.join(args.indep_dir, "*.fasta")))
    train_files = sorted(glob.glob(os.path.join(args.train_dir, "LPMO_*.fasta")) +
                         glob.glob(os.path.join(args.train_dir, "NON-LPMO_*.fasta")))
    if not indep_files:
        sys.exit(f"No .fasta files in {args.indep_dir}")
    if not train_files:
        sys.exit(f"No LPMO_*.fasta / NON-LPMO_*.fasta files in {args.train_dir}")

    tmp_dir = os.path.join(args.out_dir, "tmp")
    final_dir = os.path.join(args.out_dir, "final")
    os.makedirs(tmp_dir, exist_ok=True)
    os.makedirs(final_dir, exist_ok=True)
    for _, d in SCENARIOS:
        os.makedirs(os.path.join(args.out_dir, d, "kept"), exist_ok=True)
        os.makedirs(os.path.join(args.out_dir, d, "removed"), exist_ok=True)

    print(f"Remove if identity >= {args.pid}% AND coverage >= {args.cov}%\n")
    summary_rows, removed_rows = [], []

    for qf in indep_files:
        fam = label(qf)
        own = own_family_files(qf, train_files)
        if not own:
            print(f"[SKIP] {fam}: no matching own-family training file found")
            continue
        others = [f for f in train_files if f not in own]
        groups = {"S1": own, "S2": others}

        q_records = [(h, clean_seq(s)) for h, s in read_fasta(qf) if clean_seq(s)]
        id2rec = {f"Q{i:07d}": rec for i, rec in enumerate(q_records, 1)}
        q_fa = os.path.join(tmp_dir, f"{fam}_query.fasta")
        write_fasta([(qid, s) for qid, (h, s) in id2rec.items()], q_fa)

        failed = {}
        for tag, folder in SCENARIOS:
            tfs = groups[tag]
            print(f"[{tag}] {os.path.basename(qf)} vs {len(tfs)} file(s): "
                  f"{', '.join(label(t) for t in tfs)}")
            if not tfs:
                failed[tag] = set()
                bad = pd.DataFrame()
            else:
                bad = blast_check(fam, tag, q_fa, tfs, tmp_dir, args)
                failed[tag] = set(bad["qid"])
            for _, r in bad.iterrows():
                removed_rows.append({"family": fam, "scenario": folder,
                                     "independent_sequence": id2rec[r["qid"]][0],
                                     "closest_training_file": r["t_file"],
                                     "closest_training_sequence": r["t_header"],
                                     "identity_%": round(r["pident"], 2),
                                     "coverage_%": r["qcovhsp"], "evalue": r["evalue"]})
            base = os.path.join(args.out_dir, folder)
            write_fasta([rec for q, rec in id2rec.items() if q not in failed[tag]],
                        os.path.join(base, "kept", os.path.basename(qf)))
            write_fasta([rec for q, rec in id2rec.items() if q in failed[tag]],
                        os.path.join(base, "removed", os.path.basename(qf)))

        final_kept = [rec for q, rec in id2rec.items() if q not in failed["S1"] | failed["S2"]]
        write_fasta(final_kept, os.path.join(final_dir, os.path.basename(qf)))

        total = len(id2rec)
        summary_rows.append({
            "family": fam,
            "total": total,
            "S1_removed_own_family": len(failed["S1"]),
            "S2_removed_other_families": len(failed["S2"]),
            "removed_in_both": len(failed["S1"] & failed["S2"]),
            "total_removed": len(failed["S1"] | failed["S2"]),
            "FINAL_kept": len(final_kept),
            "FINAL_percent_kept": round(100 * len(final_kept) / total, 2) if total else 0,
        })
        print()

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(args.out_dir, "family_summary.csv"), index=False)
    pd.DataFrame(removed_rows).to_csv(os.path.join(args.out_dir, "removed_sequences.csv"), index=False)

    pd.set_option("display.width", 250)
    print(summary.to_string(index=False))
    print(f"\nFINAL independent set (passed both scenarios): {final_dir}")
    print(f"Tables: family_summary.csv, removed_sequences.csv in {args.out_dir}")


if __name__ == "__main__":
    main()
