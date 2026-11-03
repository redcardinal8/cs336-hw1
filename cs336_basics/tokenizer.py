from __future__ import annotations

import pickle
from collections.abc import Iterable, Iterator

import regex as re

# GPT-2 pretokenization regex (same one used in BPE training)
PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
_PAT_RE = re.compile(PAT)


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.vocab: dict[int, bytes] = dict(vocab)
        self.merges = list(merges)
        self.special_tokens = list(special_tokens) if special_tokens else []

        # Make sure every special token has an id (append if missing from the vocab).
        self.byte_to_id: dict[bytes, int] = {b: i for i, b in self.vocab.items()}
        for spec in self.special_tokens:
            spec_bytes = spec.encode("utf-8")
            if spec_bytes not in self.byte_to_id:
                new_id = max(self.vocab) + 1 if self.vocab else 0
                self.vocab[new_id] = spec_bytes
                self.byte_to_id[spec_bytes] = new_id

        # pair -> priority (lower = merged earlier during training)
        self.merge_ranks: dict[tuple[bytes, bytes], int] = {pair: i for i, pair in enumerate(self.merges)}

        # Longest specials first so overlapping specials (e.g. "<|eot|><|eot|>") match greedily.
        if self.special_tokens:
            ordered = sorted(self.special_tokens, key=len, reverse=True)
            self._special_re = re.compile("(" + "|".join(re.escape(s) for s in ordered) + ")")
            self._max_special_len = max(len(s) for s in self.special_tokens)
        else:
            self._special_re = None
            self._max_special_len = 0
        self._special_set = set(self.special_tokens)

        self._cache: dict[str, list[int]] = {}

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> Tokenizer:
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)
        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    # ------------------------------------------------------------------ #
    # Encoding
    # ------------------------------------------------------------------ #
    def _bpe(self, pretoken: str) -> list[int]:
        """Apply merges (in training order) to a single pretoken and return its ids."""
        cached = self._cache.get(pretoken)
        if cached is not None:
            return cached

        parts = [bytes([b]) for b in pretoken.encode("utf-8")]
        ranks = self.merge_ranks
        while len(parts) > 1:
            # Find the adjacent pair with the lowest merge rank.
            best_rank = None
            best_pair = None
            for pair in zip(parts, parts[1:]):
                r = ranks.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best_rank, best_pair = r, pair
            if best_pair is None:
                break
            # Merge every (non-overlapping, left-to-right) occurrence of best_pair.
            merged = best_pair[0] + best_pair[1]
            new_parts = []
            i = 0
            while i < len(parts):
                if i < len(parts) - 1 and parts[i] == best_pair[0] and parts[i + 1] == best_pair[1]:
                    new_parts.append(merged)
                    i += 2
                else:
                    new_parts.append(parts[i])
                    i += 1
            parts = new_parts

        ids = [self.byte_to_id[p] for p in parts]
        self._cache[pretoken] = ids
        return ids

    def _encode_plain(self, text: str) -> list[int]:
        """Encode text known to contain no special tokens."""
        ids: list[int] = []
        for m in _PAT_RE.finditer(text):
            ids.extend(self._bpe(m.group()))
        return ids

    def _split_specials(self, text: str) -> list[str]:
        if self._special_re is None:
            return [text]
        return [c for c in self._special_re.split(text) if c]

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        for chunk in self._split_specials(text):
            if chunk in self._special_set:
                ids.append(self.byte_to_id[chunk.encode("utf-8")])
            else:
                ids.extend(self._encode_plain(chunk))
        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        Lazily encode an iterable of strings (e.g. a file handle). Produces exactly the
        same ids as encode("".join(iterable)), while only buffering a small tail of text
        whose tokenization could still change once more input arrives.
        """
        buffer = ""
        for piece in iterable:
            buffer += piece
            ids, buffer = self._encode_safe_prefix(buffer)
            yield from ids
        if buffer:
            yield from self.encode(buffer)

    def _encode_safe_prefix(self, text: str) -> tuple[list[int], str]:
        """
        Encode the longest prefix of `text` whose tokenization can no longer change no matter
        what text is appended, and return (ids, remainder). The final pretoken (whose extent
        may still grow) and any possible partial special token at the end are held back.
        """
        n = len(text)
        mx = self._max_special_len

        # A special-token match is final only if every (longer) special that could start at
        # the same position already fits in the buffer; otherwise more input could extend it.
        last_special_end = 0
        if self._special_re is not None:
            for m in self._special_re.finditer(text):
                if m.start() + mx <= n:
                    last_special_end = m.end()

        # Any special token after last_special_end starts within the final `mx` characters,
        # so text[last_special_end : n - mx] is a prefix of one plain (special-free) segment.
        plain = text[last_special_end : max(last_special_end, n - mx)]

        # Hold back the last pretoken of that prefix: its extent may still grow. The others
        # are emitted exactly as matched here (re-running the regex on the cut prefix alone
        # could group trailing whitespace differently).
        ids = self.encode(text[:last_special_end]) if last_special_end else []
        matches = [m.group() for m in _PAT_RE.finditer(plain)]
        for pretoken in matches[:-1]:
            ids.extend(self._bpe(pretoken))
        split_at = last_special_end + sum(len(p) for p in matches[:-1])
        return ids, text[split_at:]

    # ------------------------------------------------------------------ #
    # Decoding
    # ------------------------------------------------------------------ #
    def decode(self, ids: list[int]) -> str:
        return b"".join(self.vocab[i] for i in ids).decode("utf-8", errors="replace")
