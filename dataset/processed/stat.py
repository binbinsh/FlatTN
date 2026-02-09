#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Get label statistics for BMESO files")
    parser.add_argument("--bmes_file", default="dataset/processed/shuffled_BMESO/test.char.bmes")
    parser.add_argument("--output_csv", default="dataset/processed/shuffled_BMESO/statistics.csv")
    parser.add_argument("--exclude", default="O,PUNC")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    exclude = {x.strip() for x in args.exclude.split(",") if x.strip()}

    tag_counter = Counter()
    coarse_counter = Counter()

    with Path(args.bmes_file).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            _, tag = line.split(maxsplit=1)
            tag_counter[tag] += 1
            if tag == "O":
                coarse = "O"
            elif "-" in tag:
                coarse = tag.split("-", 1)[1]
            else:
                coarse = tag
            coarse_counter[coarse] += 1

    filtered = {k: v for k, v in coarse_counter.items() if k not in exclude}
    total = sum(filtered.values())

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("category,count,ratio\n")
        for cat, count in sorted(filtered.items(), key=lambda x: (-x[1], x[0])):
            ratio = count / max(total, 1)
            f.write(f"{cat},{count},{ratio:.8f}\n")

    print(json.dumps({
        "bmes_file": args.bmes_file,
        "num_tags": len(tag_counter),
        "total_tokens": sum(tag_counter.values()),
        "filtered_total": total,
        "top10_filtered": sorted(filtered.items(), key=lambda x: (-x[1], x[0]))[:10],
        "output_csv": str(out_csv),
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
