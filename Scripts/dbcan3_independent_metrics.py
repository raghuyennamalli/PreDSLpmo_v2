#!/usr/bin/env python3
"""
Excel Workbook-DBCAN_INDEPENDENT.xlsx, every sheet has the columns: Gene ID | EC# | HMMER | dbCAN_sub | DIAMOND | Signalp | #ofTools

  python dbcan3_independent_metrics.py /path/to/DBCAN_INDEPENDENT.xlsx
 
"""

import argparse
import os
import re
import sys

import pandas as pd

DEFAULT_DIR = ("/media/ragothaman/yennamalli/Sweety/LPMO_NEW/New_Version/"
               "Cluster_Aware_Spliting_Changed")
DEFAULT_EXCEL = os.path.join(DEFAULT_DIR, "DBCAN_INDEPENDENT.xlsx")
DEFAULT_OUT = os.path.join(DEFAULT_DIR, "DBCAN_INDEPENDENT_metrics.xlsx")

TOOLS = ["HMMER", "dbCAN_sub", "DIAMOND"]
LPMO_FAMILIES = ["AA9", "AA10", "AA11", "AA13", "AA14", "AA15", "AA16", "AA17"]
LPMO_SET = set(LPMO_FAMILIES)
NON_LPMO_FAMILIES = {"AA1", "AA2", "AA3", "AA4", "AA5", "AA6", "AA7", "AA8", "AA12"}
CLASSES = LPMO_FAMILIES + ["Non-LPMO"]

INPUT_COUNTS = {
    "AA9": 327, "AA10": 93, "AA11": 59, "AA13": 8, "AA14": 17,
    "AA15": 18, "AA16": 49, "AA17": 12, "Non-LPMO": 583,
}

EMPTY = {"", "-", "N", "NA", "N/A", "NAN", "NONE"}
AA_RE = re.compile(r"AA(\d+)", re.IGNORECASE)
FAM_RE = re.compile(r"(GH|GT|PL|CE|AA)(\d+)", re.IGNORECASE)


def is_empty(cell):
    if cell is None or (isinstance(cell, float) and pd.isna(cell)):
        return True
    return str(cell).strip().upper() in EMPTY


def parse_families(cell):

    if is_empty(cell):
        return set()
    fams = set()
    for token in str(cell).split("+"):
        token = token.strip()
        if re.fullmatch(r"[\d.]+", token):      
            continue
        m = FAM_RE.match(token)                 
        if m:
            fams.add(f"{m.group(1).upper()}{int(m.group(2))}")
    return fams


def class_from_sheet(sheet):
    """'AA9' -> 'AA9', 'Non_LPMO' / 'Non-LPMO' / 'NonLPMO' -> 'Non-LPMO'."""
    if "non" in sheet.lower():
        return "Non-LPMO"
    m = AA_RE.search(sheet)
    if m:
        fam = f"AA{int(m.group(1))}"
        if fam in LPMO_SET:
            return fam
        if fam in NON_LPMO_FAMILIES:
            return "Non-LPMO"
    return None

def norm(col):
    return re.sub(r"[^a-z0-9#]", "", str(col).lower())


CANON = {"geneid": "Gene ID", "ec#": "EC#", "ec": "EC#",
         "hmmer": "HMMER", "dbcansub": "dbCAN_sub", "diamond": "DIAMOND",
         "signalp": "Signalp", "#oftools": "#ofTools", "oftools": "#ofTools"}


def load_workbook(path):
    xls = pd.ExcelFile(path)
    frames = []
    for sheet in xls.sheet_names:
        cls = class_from_sheet(sheet)
        if cls is None:
            print(f"[skip] '{sheet}': cannot tell the true class from the sheet name",
                  file=sys.stderr)
            continue
        raw = pd.read_excel(xls, sheet_name=sheet, header=None, dtype=str)
        hdr = next((i for i in range(min(15, len(raw)))
                    if any(norm(v) == "geneid" for v in raw.iloc[i])), None)
        if hdr is None:
            print(f"[skip] '{sheet}': no 'Gene ID' header", file=sys.stderr)
            continue
        df = raw.iloc[hdr + 1:].copy()
        df.columns = [CANON.get(norm(c), c) for c in raw.iloc[hdr]]
        df = df[df["Gene ID"].notna() & (df["Gene ID"].str.strip() != "")]
        df["Gene ID"] = df["Gene ID"].str.strip()
        missing = [t for t in TOOLS if t not in df.columns]
        if missing:
            print(f"[skip] '{sheet}': missing tool columns {missing}", file=sys.stderr)
            continue
        df["True_Class"] = cls
        df["Sheet"] = sheet
        frames.append(df)
        print(f"[read] '{sheet}' -> {cls}: {len(df)} rows", file=sys.stderr)
    if not frames:
        sys.exit("No usable sheets found.")
    return pd.concat(frames, ignore_index=True)


def add_unannotated(data, counts):

    dup = data.duplicated(subset=["True_Class", "Gene ID"])
    if dup.any():
        print(f"[warn] {dup.sum()} duplicate Gene IDs removed "
              f"(e.g. {data.loc[dup, 'Gene ID'].head(3).tolist()})", file=sys.stderr)
        data = data[~dup]
    data = data.assign(Annotated=True)

    coverage, extra = [], []
    for cls in CLASSES:
        n_excel = int((data["True_Class"] == cls).sum())
        if cls not in counts:
            print(f"[warn] no input count for {cls}; using the Excel rows only",
                  file=sys.stderr)
            n_input = n_excel
        else:
            n_input = counts[cls]
        n_missing = n_input - n_excel
        if n_missing < 0:
            print(f"[warn] {cls}: {n_excel} rows in Excel but only {n_input} input "
                  f"sequences; no missing rows added", file=sys.stderr)
            n_missing = 0
        coverage.append({"Class": cls, "Input sequences": n_input,
                         "In Excel": n_excel, "Not annotated (added)": n_missing})
        for i in range(n_missing):
            extra.append({"Sheet": "not in Excel", "Gene ID": f"unannotated_{cls}_{i+1}",
                          "True_Class": cls, "Annotated": False,
                          **{t: "-" for t in TOOLS}})

    cov = pd.DataFrame(coverage)
    tot = cov[["Input sequences", "In Excel", "Not annotated (added)"]].sum()
    cov = pd.concat([cov, pd.DataFrame([{"Class": "TOTAL", **tot}])], ignore_index=True)
    return pd.concat([data, pd.DataFrame(extra)], ignore_index=True), cov

def add_predictions(data, min_tools):
    for t in TOOLS:
        data[f"{t}_families"] = data[t].apply(parse_families)

    def consensus(row):
        votes = {}
        for t in TOOLS:
            for f in row[f"{t}_families"]:
                votes[f] = votes.get(f, 0) + 1
        return {f for f, n in votes.items() if n >= min_tools}

    data["Consensus_families"] = data.apply(consensus, axis=1)
    data["Tools_with_AA"] = data.apply(
        lambda r: sum(bool(r[f"{t}_families"]) for t in TOOLS), axis=1)  
    return data


def predicts(fams, annotated, cls, nonlpmo_mode):
    if cls != "Non-LPMO":
        return cls in fams                      
    if not annotated:                           
        return False
    if nonlpmo_mode == "annotated":
        return not (fams & LPMO_SET)            
    return bool(fams & NON_LPMO_FAMILIES)      

def div(a, b):
    return a / b if b else 0.0


def score(data, fam_col, nonlpmo_mode):
    rows = []
    for cls in CLASSES:
        truth = data["True_Class"] == cls
        pred = pd.Series([predicts(f, bool(a), cls, nonlpmo_mode)
                          for f, a in zip(data[fam_col], data["Annotated"])],
                         index=data.index)
        tp, fn = int((truth & pred).sum()), int((truth & ~pred).sum())
        fp, tn = int((~truth & pred).sum()), int((~truth & ~pred).sum())
        r, p = div(tp, tp + fn), div(tp, tp + fp)
        rows.append({"Class": cls, "N": int(truth.sum()),
                     "True Positive": tp, "False Positive": fp,
                     "False Negative": fn, "True Negative": tn,
                     "Recall": round(r, 4), "Precision": round(p, 4),
                     "F1-score": round(div(2 * p * r, p + r), 4)})
    df = pd.DataFrame(rows)

    tp, fp = df["True Positive"].sum(), df["False Positive"].sum()
    fn, tn = df["False Negative"].sum(), df["True Negative"].sum()
    r, p = div(tp, tp + fn), div(tp, tp + fp)
    present = df[df["N"] > 0]
    extra = pd.DataFrame([
        {"Class": "Micro-average", "N": df["N"].sum(),
         "True Positive": tp, "False Positive": fp,
         "False Negative": fn, "True Negative": tn,
         "Recall": round(r, 4), "Precision": round(p, 4),
         "F1-score": round(div(2 * p * r, p + r), 4)},
        {"Class": "Macro-average", "N": df["N"].sum(),
         "Recall": round(present["Recall"].mean(), 4),
         "Precision": round(present["Precision"].mean(), 4),
         "F1-score": round(present["F1-score"].mean(), 4)}])
    df = pd.concat([df, extra], ignore_index=True)
    for c in ["N", "True Positive", "False Positive", "False Negative", "True Negative"]:
        df[c] = df[c].astype("Int64")
    return df


def confusion(data, fam_col):
    """True class (rows) vs consensus call (columns)."""
    def call(fams, annotated):
        if not annotated:
            return "not annotated"
        lp = sorted(fams & LPMO_SET, key=lambda f: int(f[2:]))
        if len(lp) > 1:
            return "multiple LPMO"
        if lp:
            return lp[0]
        if fams & NON_LPMO_FAMILIES:
            return "AA1-8/AA12"
        if fams:
            return "other CAZy (GH/GT/PL/CE)"
        return "annotated, no 2-tool call"
    calls = pd.Series([call(f, bool(a)) for f, a in
                       zip(data[fam_col], data["Annotated"])], index=data.index)
    cols = LPMO_FAMILIES + ["multiple LPMO", "AA1-8/AA12",
                            "other CAZy (GH/GT/PL/CE)",
                            "annotated, no 2-tool call", "not annotated"]
    cm = pd.crosstab(data["True_Class"], calls).reindex(index=CLASSES, columns=cols,
                                                        fill_value=0)
    cm.index.name = "True class"
    cm.columns.name = "Predicted"
    return cm


def parse_counts(items):
    counts = dict(INPUT_COUNTS)
    for it in items or []:
        k, v = it.split("=")
        k = "Non-LPMO" if "non" in k.lower() else k.upper()
        counts[k] = int(v)
    return counts


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("excel", nargs="?", default=DEFAULT_EXCEL)
    ap.add_argument("--min-tools", type=int, default=2)
    ap.add_argument("--nonlpmo", choices=["annotated", "aa"], default="annotated")
    ap.add_argument("--counts", nargs="*", metavar="CLASS=N",
                    help="override input counts, e.g. AA9=327 Non-LPMO=583")
    ap.add_argument("-o", "--output", default=DEFAULT_OUT)
    a = ap.parse_args()

    counts = parse_counts(a.counts)
    data, coverage = add_unannotated(load_workbook(a.excel), counts)
    data = add_predictions(data, a.min_tools)
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)

    print("\n=== Input sequences vs. sequences annotated by dbCAN3 ===")
    print(coverage.to_string(index=False), "\n")

    consensus = score(data, "Consensus_families", a.nonlpmo)
    print(f"dbCAN3 consensus (>= {a.min_tools} tools agree), "
          f"Non-LPMO mode: {a.nonlpmo} ")
    print(consensus.to_string(index=False), "\n")

    cm = confusion(data, "Consensus_families")
    print("Confusion table (consensus)")
    print(cm.to_string(), "\n")

    per_tool = pd.concat([score(data, f"{t}_families", a.nonlpmo).assign(Method=t)
                          for t in TOOLS], ignore_index=True)

    detail = data[["Sheet", "Gene ID", "True_Class", "Annotated"] + TOOLS].copy()
    for t in TOOLS:
        detail[f"{t} -> family"] = data[f"{t}_families"].apply(lambda s: ";".join(sorted(s)) or "-")
    detail["Consensus -> family"] = data["Consensus_families"].apply(
        lambda s: ";".join(sorted(s)) or "-")
    detail["Tools with a call"] = data["Tools_with_AA"]

    with pd.ExcelWriter(a.output, engine="openpyxl") as xw:
        consensus.to_excel(xw, sheet_name="Consensus_>=2tools", index=False)
        cm.to_excel(xw, sheet_name="Confusion")
        coverage.to_excel(xw, sheet_name="Input_coverage", index=False)
        per_tool.to_excel(xw, sheet_name="Per_tool", index=False)
        detail.to_excel(xw, sheet_name="Per_sequence", index=False)
    print(f"Written to {a.output}")


if __name__ == "__main__":
    main()
