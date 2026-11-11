import argparse
import csv
from pathlib import Path

from config import CBC_FEATURES
from utils import first_float, read_csv, write_csv


NAME_MAP = {
    "RDW---CV": "RDW_CV",
    "#NEUT": "NEUT_abs",
    "#LYMP": "LYMP_abs",
    "#MONO": "MONO_abs",
    "#EOS": "EOS_abs",
    "#BASO": "BASO_abs",
}


def clean_name(name):
    name = (name or "").strip()
    return NAME_MAP.get(name, name)


def parse_report(path):
    values = {name: "" for name in CBC_FEATURES}
    with open(path, newline="", encoding="utf-8-sig") as f:
        reader = csv.reader(f)
        next(reader, None)
        for row in reader:
            if len(row) < 2:
                continue
            name = clean_name(row[0])
            if name in values:
                values[name] = first_float(row[1])
    return values


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="data/processed/manifest.csv")
    parser.add_argument("--output", default="data/processed/cbc_features.csv")
    args = parser.parse_args()
    rows = []
    for item in read_csv(args.manifest):
        report = item.get("cbc_report_path", "")
        features = parse_report(report) if report and Path(report).exists() else {name: "" for name in CBC_FEATURES}
        rows.append({"case_id": item["case_id"], **features, "label_id": item["label_id"]})
    write_csv(args.output, rows, ["case_id", *CBC_FEATURES, "label_id"])
    print(f"Wrote CBC features for {len(rows)} cases to {args.output}")


if __name__ == "__main__":
    main()
