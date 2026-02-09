from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .rules import Rule, Trie


@dataclass
class Lattice:
    tokens: list[str]
    pos_s: list[int]
    pos_e: list[int]
    lex_num: int


def load_lexicon_words(path: str | Path, min_len: int = 2) -> list[str]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    if lines and lines[0].strip().isdigit() and len(lines) > 1000:
        lines = lines[1:]

    words: list[str] = []
    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        word = parts[0]
        if len(word) < min_len:
            continue
        words.append(word)
    return words


def build_trie(words: list[str]) -> Trie:
    trie = Trie()
    for w in words:
        trie.insert(w)
    return trie


def build_lattice(
    chars: list[str],
    trie: Trie,
    rule: Rule,
    use_lexicon: bool = True,
    use_rule: bool = True,
    max_lattice_len: int | None = None,
) -> Lattice:
    sentence = "".join(chars)

    lexicon_matches: list[tuple[int, int, str]] = []
    rule_matches: list[tuple[int, int, str]] = []

    if use_lexicon:
        # Trie returns inclusive end index.
        seen = set()
        for start, end, token, *_ in trie.get_lexicon(sentence):
            key = (int(start), int(end), str(token))
            if key in seen:
                continue
            seen.add(key)
            lexicon_matches.append(key)
        lexicon_matches.sort(key=lambda x: (x[0], x[1], x[2]))

    if use_rule:
        # Rule returns exclusive end index; convert to inclusive end index.
        seen = set()
        for start, end_exclusive, token, *_ in rule.get_lexicon(sentence):
            end_inclusive = int(end_exclusive) - 1
            if end_inclusive < int(start):
                continue
            key = (int(start), end_inclusive, str(token))
            if key in seen:
                continue
            seen.add(key)
            rule_matches.append(key)
        rule_matches.sort(key=lambda x: (x[0], x[1], x[2]))

    extras = lexicon_matches + rule_matches
    if max_lattice_len is not None and max_lattice_len > 0:
        max_extra = max(max_lattice_len - len(chars), 0)
        extras = extras[:max_extra]

    tokens = chars + [token for _, _, token in extras]
    pos_s = list(range(len(chars))) + [s for s, _, _ in extras]
    pos_e = list(range(len(chars))) + [e for _, e, _ in extras]

    return Lattice(tokens=tokens, pos_s=pos_s, pos_e=pos_e, lex_num=len(extras))
