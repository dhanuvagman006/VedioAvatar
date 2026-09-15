"""Script normalisation and chunking for TTS.

Kept dependency-free on purpose: the TTS worker imports this module from inside
its own virtual environment.
"""
from __future__ import annotations

import re

_SENTENCE_END = re.compile(r"(?<=[.!?…。！？])\s+")
_CLAUSE_SPLIT = re.compile(r"(?<=[,;:،])\s+")
_WS = re.compile(r"[ \t\f\v]+")
_PARAGRAPH = re.compile(r"\n\s*\n+")


def normalize_script(text: str) -> str:
    """Normalise whitespace and quotes; keep paragraph breaks as blank lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")
    paragraphs = [_WS.sub(" ", p.replace("\n", " ")).strip() for p in _PARAGRAPH.split(text)]
    return "\n\n".join(p for p in paragraphs if p)


def split_sentences(text: str) -> list[str]:
    """Split normalised text into sentences; paragraph breaks always split."""
    sentences: list[str] = []
    for paragraph in normalize_script(text).split("\n\n"):
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if sentence:
                sentences.append(sentence)
    return sentences


def _split_long(sentence: str, max_chars: int) -> list[str]:
    """Break a sentence longer than max_chars on clause boundaries, then on words."""
    if len(sentence) <= max_chars:
        return [sentence]
    pieces: list[str] = []
    current = ""
    for clause in _CLAUSE_SPLIT.split(sentence):
        if len(clause) > max_chars:
            if current:
                pieces.append(current)
                current = ""
            words = clause.split(" ")
            buf = ""
            for word in words:
                candidate = f"{buf} {word}".strip()
                if len(candidate) > max_chars and buf:
                    pieces.append(buf)
                    buf = word
                else:
                    buf = candidate
            if buf:
                current = buf
            continue
        candidate = f"{current} {clause}".strip()
        if len(candidate) > max_chars and current:
            pieces.append(current)
            current = clause
        else:
            current = candidate
    if current:
        pieces.append(current)
    return pieces


def chunk_text(text: str, max_chars: int = 250) -> list[str]:
    """Group sentences into chunks of at most ``max_chars`` characters.

    Sentences are never merged across paragraph breaks. Over-long sentences are
    split on clause boundaries and, failing that, on words.
    """
    if max_chars < 20:
        raise ValueError("max_chars must be at least 20")
    chunks: list[str] = []
    for paragraph in normalize_script(text).split("\n\n"):
        current = ""
        for sentence in _SENTENCE_END.split(paragraph):
            sentence = sentence.strip()
            if not sentence:
                continue
            for piece in _split_long(sentence, max_chars):
                candidate = f"{current} {piece}".strip()
                if len(candidate) > max_chars and current:
                    chunks.append(current)
                    current = piece
                else:
                    current = candidate
        if current:
            chunks.append(current)
    return chunks


def estimate_speech_seconds(text: str, words_per_second: float = 2.5) -> float:
    words = len(normalize_script(text).split())
    return max(1.0, words / words_per_second)
