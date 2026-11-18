"""Dependency-free byte-level BPE tokenizer trained on the air quality corpus.

Why a custom tokenizer: the pipeline must run on a CPU-only laptop with no
network access, so we avoid downloading a pretrained vocabulary. A byte-level
BPE learned from *this* corpus keeps the vocabulary small (default 4096) and
never produces an unknown token, because every byte is representable.
"""

from __future__ import annotations

import json
import os
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

PAD_TOKEN = "<pad>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"
SPECIAL_TOKENS = (PAD_TOKEN, BOS_TOKEN, EOS_TOKEN)

# GPT-2 style pre-tokenisation: keep the leading space, split digits and punctuation.
PRE_TOKEN_PATTERN = re.compile(
    r"'s|'t|'re|'ve|'m|'ll|'d| ?[A-Za-z]+| ?\d+| ?[^\sA-Za-z\d]+|\s+(?!\S)|\s+"
)


def pretokenize(text: str) -> List[str]:
    return PRE_TOKEN_PATTERN.findall(text)


class ByteLevelBPETokenizer:
    """Minimal byte-level BPE with a merge-rank table."""

    def __init__(
        self,
        merges: Optional[List[Tuple[int, int]]] = None,
        special_tokens: Sequence[str] = SPECIAL_TOKENS,
    ) -> None:
        self.special_tokens = list(special_tokens)
        self.merges: List[Tuple[int, int]] = list(merges or [])
        self.merge_ranks: Dict[Tuple[int, int], int] = {
            pair: rank for rank, pair in enumerate(self.merges)
        }
        self._rebuild_vocab()
        self._cache: Dict[str, List[int]] = {}

    # ------------------------------------------------------------------ vocab
    def _rebuild_vocab(self) -> None:
        offset = len(self.special_tokens)
        self.token_to_id: Dict[str, int] = {tok: i for i, tok in enumerate(self.special_tokens)}
        for byte in range(256):
            self.token_to_id[bytes([byte]).decode("latin-1")] = offset + byte
        next_id = offset + 256
        self.id_to_token: Dict[int, str] = {i: t for t, i in self.token_to_id.items()}
        self.id_to_bytes: Dict[int, bytes] = {}
        for byte in range(256):
            self.id_to_bytes[offset + byte] = bytes([byte])
        for left, right in self.merges:
            token = self.id_to_bytes[left] + self.id_to_bytes[right]
            self.id_to_bytes[next_id] = token
            self.id_to_token[next_id] = token.decode("latin-1")
            next_id += 1
        for token_id in range(next_id):
            self.id_to_token.setdefault(token_id, "")
        self._next_id = next_id

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD_TOKEN]

    @property
    def bos_id(self) -> int:
        return self.token_to_id[BOS_TOKEN]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS_TOKEN]

    def __len__(self) -> int:
        return self._next_id

    @property
    def vocab_size(self) -> int:
        return self._next_id

    # ------------------------------------------------------------------ train
    @classmethod
    def train(
        cls,
        texts: Iterable[str],
        vocab_size: int = 4096,
        min_frequency: int = 2,
        max_unique_pretokens: int = 60000,
        progress: bool = True,
    ) -> "ByteLevelBPETokenizer":
        """Learn BPE merges from the corpus (unique pre-tokens only, for speed)."""
        frequencies: Counter = Counter()
        for text in texts:
            frequencies.update(pretokenize(text))
        if not frequencies:
            raise ValueError("Tokenizer training corpus is empty")

        # Keep the most frequent pre-tokens; BPE quality barely changes otherwise.
        items = frequencies.most_common(max_unique_pretokens)
        words: List[List[int]] = []
        counts: List[int] = []
        for token, count in items:
            words.append(list(token.encode("utf-8")))
            counts.append(count)

        mergeable = len(SPECIAL_TOKENS) + 256
        target_merges = max(0, vocab_size - mergeable)
        merges: List[Tuple[int, int]] = []
        ranks: Dict[Tuple[int, int], int] = {}
        checked = 0
        while len(merges) < target_merges:
            pair_counts: Counter = Counter()
            for word, count in zip(words, counts):
                for i in range(len(word) - 1):
                    pair_counts[(word[i], word[i + 1])] += count
            if not pair_counts:
                break
            (best_pair, best_count) = pair_counts.most_common(1)[0]
            if best_count < min_frequency:
                break
            new_id = mergeable + len(merges)
            ranks[best_pair] = len(merges)
            merges.append(best_pair)
            # apply the merge in place
            left, right = best_pair
            for idx, word in enumerate(words):
                if len(word) < 2:
                    continue
                merged: List[int] = []
                i = 0
                while i < len(word):
                    if i < len(word) - 1 and word[i] == left and word[i + 1] == right:
                        merged.append(new_id)
                        i += 2
                    else:
                        merged.append(word[i])
                        i += 1
                words[idx] = merged
            checked += 1
            if progress and checked % 500 == 0:
                print(f"  [tokenizer] merges learned: {len(merges)} (best pair count {best_count})", flush=True)
        if progress:
            print(f"  [tokenizer] finished with {len(merges)} merges -> vocab {mergeable + len(merges)}", flush=True)
        return cls(merges=merges)

    # ------------------------------------------------------------------ encode
    def _encode_pretoken(self, token: str) -> List[int]:
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        ids = [len(SPECIAL_TOKENS) + b for b in token.encode("utf-8")]
        if len(ids) > 1 and self.merges:
            while True:
                best_rank: Optional[int] = None
                best_index = -1
                for i in range(len(ids) - 1):
                    rank = self.merge_ranks.get((ids[i], ids[i + 1]))
                    if rank is not None and (best_rank is None or rank < best_rank):
                        best_rank, best_index = rank, i
                if best_index < 0:
                    break
                new_id = len(SPECIAL_TOKENS) + 256 + best_rank
                ids = ids[:best_index] + [new_id] + ids[best_index + 2:]
        if len(self._cache) < 200000:
            self._cache[token] = ids
        return ids

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        ids: List[int] = []
        if add_bos:
            ids.append(self.bos_id)
        for token in pretokenize(text):
            ids.extend(self._encode_pretoken(token))
        if add_eos:
            ids.append(self.eos_id)
        return ids

    def encode_batch(self, texts: Sequence[str], add_bos: bool = True, add_eos: bool = True) -> List[List[int]]:
        return [self.encode(t, add_bos=add_bos, add_eos=add_eos) for t in texts]

    def decode(self, ids: Sequence[int], skip_special: bool = True) -> str:
        chunks: List[bytes] = []
        for token_id in ids:
            token_id = int(token_id)
            if token_id < len(SPECIAL_TOKENS):
                if not skip_special:
                    chunks.append(self.special_tokens[token_id].encode("latin-1"))
                continue
            data = self.id_to_bytes.get(token_id)
            if data is not None:
                chunks.append(data)
        return b"".join(chunks).decode("utf-8", errors="replace")

    # ------------------------------------------------------------------ io
    def save(self, directory: str) -> None:
        os.makedirs(directory, exist_ok=True)
        vocab = {token: idx for token, idx in self.token_to_id.items()}
        for merge_id in range(len(self.merges)):
            token = self.id_to_bytes[len(SPECIAL_TOKENS) + 256 + merge_id].decode("latin-1")
            vocab[token] = len(SPECIAL_TOKENS) + 256 + merge_id
        with open(os.path.join(directory, "vocab.json"), "w", encoding="latin-1") as fh:
            json.dump(vocab, fh, ensure_ascii=False, indent=0)
        with open(os.path.join(directory, "merges.json"), "w", encoding="utf-8") as fh:
            json.dump([[a, b] for a, b in self.merges], fh)
        with open(os.path.join(directory, "tokenizer_config.json"), "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "type": "byte_level_bpe",
                    "vocab_size": self.vocab_size,
                    "special_tokens": self.special_tokens,
                    "num_merges": len(self.merges),
                    "pre_token_pattern": PRE_TOKEN_PATTERN.pattern,
                },
                fh, indent=2,
            )

    @classmethod
    def load(cls, directory: str) -> "ByteLevelBPETokenizer":
        merges_path = os.path.join(directory, "merges.json")
        config_path = os.path.join(directory, "tokenizer_config.json")
        specials: Sequence[str] = SPECIAL_TOKENS
        merges: List[Tuple[int, int]] = []
        if os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as fh:
                cfg = json.load(fh)
            specials = cfg.get("special_tokens", SPECIAL_TOKENS)
        if os.path.exists(merges_path):
            with open(merges_path, "r", encoding="utf-8") as fh:
                merges = [(int(a), int(b)) for a, b in json.load(fh)]
        else:
            raise FileNotFoundError(f"No tokenizer merges found in {directory}")
        return cls(merges=merges, special_tokens=specials)

    # ------------------------------------------------------------------ blocks
    def encode_documents(self, documents: Sequence[str]) -> List[int]:
        """Flatten documents into one token stream with an EOS between each."""
        stream: List[int] = []
        for doc in documents:
            stream.append(self.bos_id)
            stream.extend(self.encode(doc))
            stream.append(self.eos_id)
        return stream


def create_blocks(token_ids: Sequence[int], block_size: int) -> "list[list[int]]":
    """Chunk a flat token stream into fixed-size contiguous blocks."""
    total = (len(token_ids) // block_size) * block_size
    return [list(token_ids[i:i + block_size]) for i in range(0, total, block_size)]


def create_padded_batch(sequences: Sequence[Sequence[int]], pad_id: int) -> Tuple[List[List[int]], List[List[bool]]]:
    """Right-pad a batch of sequences; returns (ids, attention_mask)."""
    width = max(len(s) for s in sequences)
    ids = [list(s) + [pad_id] * (width - len(s)) for s in sequences]
    masks = [[True] * len(s) + [False] * (width - len(s)) for s in sequences]
    return ids, masks
