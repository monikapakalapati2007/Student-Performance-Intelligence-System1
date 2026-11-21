"""Record retrieval: ground the model's answers in the actual data row.

A 2.5M-parameter model cannot memorise a million rows, so record-specific
questions are answered from a retrieved record card placed in the prompt. The
model was trained on exactly this layout (card + question -> answer), so it only
has to read and copy a value, not recall it.

    python src/retrieval.py "What is the total score for student 450579?"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from typing import Dict, List, Optional, Tuple

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import pandas as pd  # noqa: E402

import dataset_loader as dl  # noqa: E402
import preprocess_air_quality as prep  # noqa: E402
from profiles import get_profile  # noqa: E402


class RecordIndex:
    """Identifier -> record card lookup built from the project's data tables."""

    def __init__(self, cards: Dict[str, str], key_column: str, n_rows: int) -> None:
        self.cards = cards
        self.key_column = key_column
        self.n_rows = n_rows

    def __len__(self) -> int:
        return len(self.cards)

    def get(self, key: str) -> Optional[str]:
        return self.cards.get(str(key).strip())

    def retrieve(self, question: str) -> Tuple[Optional[str], Optional[str]]:
        """Find the record a question refers to: (key, card) or (None, None)."""
        for token in _id_candidates(question):
            card = self.get(token)
            if card is not None:
                return token, card
        return None, None

    def sample(self, n: int = 1) -> List[Tuple[str, str]]:
        items = list(self.cards.items())[:n]
        return items


_ID_TOKEN = re.compile(r"\b\d{3,}\b")


def _id_candidates(question: str) -> List[str]:
    return _ID_TOKEN.findall(question or "")


def build_index(project_root: str = PROJECT_ROOT, data_dir: Optional[str] = None,
                profile_key: str = "auto", max_rows: int = 0) -> RecordIndex:
    search_dirs = [os.path.abspath(data_dir)] if data_dir else dl.default_search_dirs(project_root)
    splits, discovery = dl.load_splits(search_dirs, profile_key=profile_key)
    profile = get_profile(discovery.get("profile") or "air_quality")
    prep.set_profile(profile)
    spec = dl.classify_fields(next(iter(splits.values())))
    frame = pd.concat(splits.values(), ignore_index=True)
    if max_rows and len(frame) > max_rows:
        frame = frame.head(max_rows)
    key_column = spec.identifier[0] if spec.identifier else ""
    cards: Dict[str, str] = {}
    for _, row in frame.iterrows():
        if key_column:
            key = str(row[key_column]).strip()
        else:
            key = str(row.name)
        if key in cards:
            continue
        cards[key] = prep.record_context(row, spec)
    return RecordIndex(cards, key_column, len(frame))


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Look up the record a question refers to.")
    parser.add_argument("question", nargs="?", default=None)
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--profile", default="auto")
    args = parser.parse_args(argv)

    index = build_index(data_dir=args.data_dir, profile_key=args.profile)
    print(f"Indexed {len(index):,} records (key column: `{index.key_column}`)")
    if not args.question:
        key, card = index.sample(1)[0]
        print(f"\nExample record [{key}]:\n{card}")
        return 0
    key, card = index.retrieve(args.question)
    if card is None:
        print("No record id found in the question.")
        return 1
    print(f"\nMatched record [{key}]:\n{card}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
