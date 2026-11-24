#!/usr/bin/env python3
"""Create or verify the 600-pair BBBC005 F1/w2 manifest (Python stdlib only)."""
import argparse
import csv
import hashlib
import re
from collections import Counter
from pathlib import Path

LEVELS = (1, 5, 10, 14, 18, 23, 27, 31, 35, 40, 44, 48,
          53, 57, 61, 66, 70, 74, 78, 83, 87, 91, 96, 100)
EXPECTED = {(n, 1, s, 2) for n in LEVELS for s in range(1, 26)}
PATTERN = re.compile(r"C(\d+)_F(\d+)_s(\d+)_w(\d+)", re.I)
FIELDS = ("n", "foco", "amostra", "canal", "densidade", "imagem",
          "mascara", "sha256_imagem", "sha256_mascara")


def scan(root):
    files = {}
    for path in root.iterdir():
        if path.suffix.lower() not in {".tif", ".tiff"} or not path.is_file():
            continue
        match = PATTERN.search(path.name)
        if not match:
            raise ValueError(f"Unrecognized TIFF filename: {path}")
        key = tuple(map(int, match.groups()))
        if key[1] != 1 or key[3] != 2:
            continue
        if key in files:
            raise ValueError(f"Duplicate key {key}: {files[key]} and {path}")
        files[key] = path
    missing, extra = EXPECTED - files.keys(), files.keys() - EXPECTED
    if missing or extra:
        raise ValueError(f"{root}: missing={sorted(missing)}; extra={sorted(extra)}")
    return files


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def rows_for(base):
    images = scan(base / "imagens" / "BBBC005_v1_images")
    masks = scan(base / "ground_truth" / "BBBC005_v1_ground_truth")
    for key in sorted(EXPECTED):
        if images[key].name.casefold() != masks[key].name.casefold():
            raise ValueError(f"Filename mismatch for {key}")
    rows = []
    for i, key in enumerate(sorted(EXPECTED), 1):
        n, focus, sample, channel = key
        img, mask = images[key], masks[key]
        rows.append(dict(zip(FIELDS, (
            str(n), str(focus), str(sample), str(channel),
            "Baixa" if n <= 31 else "Media" if n <= 66 else "Alta",
            img.relative_to(base).as_posix(), mask.relative_to(base).as_posix(),
            sha256(img), sha256(mask)))))
        if i % 50 == 0:
            print(f"Hashed {i}/600 pairs", flush=True)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify and args.out.exists():
        parser.error("Manifest already exists. Use --verify or a new output path.")
    print("Checking exact keys and filenames...", flush=True)
    rows = rows_for(args.base)
    if args.verify:
        with args.out.open("r", encoding="utf-8-sig", newline="") as stream:
            reader = csv.DictReader(stream)
            if tuple(reader.fieldnames or ()) != FIELDS or list(reader) != rows:
                raise ValueError("Manifest differs from current files or expected order.")
        print("MANIFEST VERIFIED")
    else:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS, lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)
        print("MANIFEST CREATED")
    print(f"Pairs: {len(rows)}")
    print("Density:", dict(Counter(r["densidade"] for r in rows)))
    for field in ("sha256_imagem", "sha256_mascara"):
        counts = Counter(r[field] for r in rows)
        print(f"{field}: {len(counts)} distinct hashes; "
              f"{sum(v > 1 for v in counts.values())} repeated hash groups")
    print(args.out)
    print("MANIFESTO OK")


if __name__ == "__main__":
    main()
