"""
Build a stratified evaluation split from VSR (Visual Spatial Reasoning).

VSR provides binary yes/no spatial-relation captions over COCO images, which
maps cleanly onto our Planner/Executor/Critic pipeline (no awkward 4-way
multiple-choice adaptation).

VSR examples look like:
    {"image": "000000451431.jpg",
     "image_link": "http://images.cocodataset.org/train2017/...jpg",
     "caption": "The person is inside the refrigerator.",
     "label": "1",
     "relation": "inside"}

We:
1. Filter to relations our 8-relation pipeline can verify.
2. Stratify by (relation, label) so each bucket is balanced.
3. Parse the caption into (subj, obj) using the canonical VSR template.
4. Download the COCO image to data/vsr_images/.
5. Emit an evaluate.py-compatible JSON split.

Usage:
  python scripts/prep_vsr_split.py --n 200 \\
      --out data/vsr_strat200.json --image_dir data/vsr_images
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
from collections import Counter, defaultdict
from io import BytesIO
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s vsr | %(message)s")
logger = logging.getLogger("prep_vsr")

# VSR relation strings → our 8 canonical relations. `swap` means obj1/obj2 in
# the question should be flipped from the caption order (e.g. "X is in Y"
# implies our pipeline's claim "Y contains X").
RELATION_MAP: dict[str, tuple[str, bool]] = {
    "at the left side of": ("left_of", False),
    "to the left of": ("left_of", False),
    "left of": ("left_of", False),
    "at the right side of": ("right_of", False),
    "to the right of": ("right_of", False),
    "right of": ("right_of", False),
    "above": ("above", False),
    "over": ("above", False),
    "below": ("below", False),
    "under": ("below", False),
    "beneath": ("below", False),
    "underneath": ("below", False),
    "in front of": ("in_front", False),
    "ahead of": ("in_front", False),
    "behind": ("behind", False),
    "at the back of": ("behind", False),
    "on": ("on", False),
    "on top of": ("on", False),
    "contains": ("contains", False),
    "containing": ("contains", False),
    "inside": ("contains", True),
    "in": ("contains", True),
    "within": ("contains", True),
}


def canonicalize_relation(rel_text: str) -> tuple[str, bool] | None:
    rel_lower = rel_text.lower().strip()
    if rel_lower in RELATION_MAP:
        return RELATION_MAP[rel_lower]
    return None


# Captions follow "The {subj} (is)? {relation} the {obj}." — sometimes with
# trailing "." sometimes without, sometimes with adjective stacks.
_CAPTION_RE = re.compile(
    r"^\s*[Tt]he\s+(?P<subj>.+?)\s+(?:is\s+)?{rel}\s+the\s+(?P<obj>.+?)\s*\.?\s*$"
)


def parse_caption(caption: str, vsr_relation: str) -> tuple[str, str] | None:
    """Pull subj/obj out of a VSR caption using its known relation phrase."""
    escaped = re.escape(vsr_relation)
    pat = re.compile(_CAPTION_RE.pattern.format(rel=escaped))
    m = pat.match(caption)
    if not m:
        return None
    subj = m.group("subj").strip()
    obj = m.group("obj").strip()
    if not subj or not obj:
        return None
    return subj, obj


def load_vsr_rows() -> list[dict]:
    from datasets import load_dataset

    logger.info("Loading VSR random split (test) from HuggingFace…")
    ds = load_dataset("cambridgeltl/vsr_random", split="test")
    logger.info(f"VSR test: {len(ds)} items")
    rows: list[dict] = []
    skipped_relation = 0
    skipped_parse = 0
    for row in ds:
        canon = canonicalize_relation(row["relation"])
        if canon is None:
            skipped_relation += 1
            continue
        canon_rel, swap = canon
        parsed = parse_caption(row["caption"], row["relation"])
        if parsed is None:
            skipped_parse += 1
            continue
        subj, obj = parsed
        if swap:
            subj, obj = obj, subj
        rows.append(
            {
                "image": row["image"],
                "image_link": row["image_link"],
                "caption": row["caption"],
                "label": int(row["label"]) if isinstance(row["label"], str) else int(row["label"]),
                "raw_relation": row["relation"],
                "relation": canon_rel,
                "subj": subj,
                "obj": obj,
            }
        )
    logger.info(
        f"After filtering: kept={len(rows)} (dropped_relation={skipped_relation} "
        f"dropped_parse={skipped_parse})"
    )
    return rows


def stratify(rows: list[dict], n: int, seed: int = 0) -> list[dict]:
    rng = random.Random(seed)
    by_key: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        by_key[(r["relation"], r["label"])].append(r)

    keys = list(by_key.keys())
    target_per_key = max(1, n // max(len(keys), 1))

    picked: list[dict] = []
    leftovers: list[dict] = []
    for key in keys:
        bucket = by_key[key]
        rng.shuffle(bucket)
        picked.extend(bucket[:target_per_key])
        leftovers.extend(bucket[target_per_key:])

    rng.shuffle(leftovers)
    while len(picked) < n and leftovers:
        picked.append(leftovers.pop())
    rng.shuffle(picked)
    return picked[:n]


def download_image(url: str, dest: Path, timeout: float = 30.0) -> bool:
    if dest.exists() and dest.stat().st_size > 0:
        return True
    import requests
    from PIL import Image

    try:
        resp = requests.get(url, timeout=timeout)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(dest, "JPEG", quality=90)
        return True
    except Exception as exc:
        logger.warning(f"download failed for {url}: {exc}")
        return False


def materialize(items: list[dict], image_dir: Path) -> list[dict]:
    image_dir.mkdir(parents=True, exist_ok=True)
    out: list[dict] = []
    for idx, it in enumerate(items):
        fname = f"vsr_{idx:04d}.jpg"
        fpath = image_dir / fname
        if not download_image(it["image_link"], fpath):
            continue
        question = f"Is the {it['subj']} {it['raw_relation']} the {it['obj']}?"
        out.append(
            {
                "image_path": fname,
                "question": question,
                "answer": "yes" if it["label"] == 1 else "no",
                "relation": it["relation"],
                "raw_relation": it["raw_relation"],
                "subj": it["subj"],
                "obj": it["obj"],
                "image_link": it["image_link"],
            }
        )
        if (idx + 1) % 25 == 0:
            logger.info(f"  materialized {idx + 1}/{len(items)}")
    return out


def main():
    parser = argparse.ArgumentParser(description="Prep stratified VSR eval split")
    parser.add_argument("--n", type=int, default=200, help="Target number of items")
    parser.add_argument("--out", default="data/vsr_strat200.json")
    parser.add_argument("--image_dir", default="data/vsr_images")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    rows = load_vsr_rows()
    # Oversample slightly so failed downloads don't drop us below the target.
    over = int(args.n * 1.2)
    picked = stratify(rows, min(over, len(rows)), seed=args.seed)
    logger.info(f"Stratified to {len(picked)}; materializing images to {args.image_dir}/")

    items = materialize(picked, Path(args.image_dir))
    items = items[: args.n]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(items, indent=2))
    logger.info(f"Wrote split ({len(items)} items) -> {out_path}")

    rel_counts = Counter(r["relation"] for r in items)
    label_counts = Counter(r["answer"] for r in items)
    logger.info(f"Relation balance: {dict(rel_counts)}")
    logger.info(f"Label balance:    {dict(label_counts)}")


if __name__ == "__main__":
    main()
