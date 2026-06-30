#!/usr/bin/env python3
"""Organize a folder of run sheets into the Kalos barcode registry.

Scans a directory for CSV / TSV files, registers each as a dataset (auto-detects
the measured outputs, group, and id columns), mints a stable barcode per dataset
and per run (anonymized), and writes one registry JSON + prints the manifest.

Run:  python examples/organize_data.py <data_dir> [registry.json]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402

from kalos.data.barcode_registry import BarcodeRegistry  # noqa: E402

_OUTCOME = re.compile(r"titer|titre|yield|conc|purity|lipase|biomass|od\d|product|response|score|kda|activity", re.I)
_GROUP = re.compile(r"medium|strain|recipe|batch|campaign|group|lot", re.I)
_ID = re.compile(r"^(id|name|sample.*|well|index|run|experiment|round|date|time|notes?)$", re.I)


def _load(path: Path) -> pd.DataFrame:
    sep = "\t" if path.suffix.lower() in (".tsv", ".txt") else ","
    return pd.read_csv(path, sep=sep)


def main() -> int:
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data")
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("barcodes.json")
    if not data_dir.exists():
        print(f"no such directory: {data_dir}")
        return 1

    reg = BarcodeRegistry()
    files = sorted(p for p in data_dir.rglob("*") if p.suffix.lower() in (".csv", ".tsv", ".txt"))
    print(f"organizing {len(files)} file(s) from {data_dir}\n")
    for f in files:
        try:
            df = _load(f)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip {f.name}: {exc}")
            continue
        cols = [str(c) for c in df.columns]
        result_cols = [c for c in df.columns if _OUTCOME.search(str(c))]
        meta_cols = [c for c in df.columns if (_GROUP.search(str(c)) or _ID.match(str(c).strip())) and c not in result_cols]
        info = reg.register_dataset(f.stem, df, result_cols=result_cols, meta_cols=meta_cols)
        print(f"  {info['dataset_barcode']}  {f.stem[:34]:34s} {info['n_runs']:4d} runs  outputs={[str(c) for c in result_cols][:3]}")

    reg.save(out)
    print(f"\nregistry: {len(reg)} runs across {len(reg.datasets)} dataset(s) -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
