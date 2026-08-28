"""
Chunking Module, Multiple strategies for splitting text into chunks.
Each strategy has its own relevant parameters.
"""

import re
from typing import List

CHUNKING_STRATEGIES = [
    {
        "id": "fixed",
        "explainer": "Slides a fixed-width window across the text: every chunk_size characters becomes a chunk, and the last `overlap` characters of each chunk are repeated at the start of the next so a sentence sitting on a boundary isn't lost to both sides. Fast, predictable, and completely blind, it will happily cut mid-word or mid-idea. Use it as the baseline that shows why boundary-aware strategies exist.",
        "name": "Fixed-Size Chunking",
        "description": "Split text into fixed-size character windows with sliding overlap. Simplest approach, ignores sentence/paragraph boundaries.",
        "icon": "📏",
        "params": [
            {"key": "chunk_size", "label": "Chunk Size (chars)", "type": "slider", "min": 100, "max": 2000, "step": 50, "default": 500},
            {"key": "chunk_overlap", "label": "Overlap (chars)", "type": "slider", "min": 0, "max": 200, "step": 10, "default": 50},
        ],
    },
    {
        "id": "recursive",
        "explainer": "Tries the most natural split first, paragraphs (blank lines), then single line breaks, then sentence ends, then spaces, and only falls back to a finer separator when a piece still exceeds the size cap. Ideas stay intact far more often than with fixed-size cutting. This is the default strategy in LangChain and most production RAG stacks.",
        "name": "Recursive Character Splitting",
        "description": "Tries splitting by paragraphs → newlines → sentences → words, respecting natural boundaries while staying within size limits. Most popular in LangChain.",
        "icon": "🔄",
        "params": [
            {"key": "chunk_size", "label": "Max Chunk Size (chars)", "type": "slider", "min": 100, "max": 2000, "step": 50, "default": 500},
            {"key": "chunk_overlap", "label": "Overlap (chars)", "type": "slider", "min": 0, "max": 200, "step": 10, "default": 50},
        ],
    },
    {
        "id": "sentence",
        "explainer": "Splits the text at sentence endings (., !, ?), then packs consecutive whole sentences into a chunk until the size cap is reached. Min Sentences forces at least that many sentences per chunk even if it overshoots the cap. Never cuts mid-sentence, but long sentences make chunk sizes uneven, and there is no overlap between chunks.",
        "name": "Sentence-Based Chunking",
        "description": "Splits at sentence boundaries (., !, ?), then groups consecutive sentences until the target size is reached. Min Sentences forces at least that many sentences per chunk, even past the size cap. No overlap, each sentence appears in exactly one chunk.",
        "icon": "📝",
        "params": [
            {"key": "chunk_size", "label": "Max Chunk Size (chars)", "type": "slider", "min": 100, "max": 2000, "step": 50, "default": 500},
            {"key": "min_sentences", "label": "Min Sentences per Chunk", "type": "slider", "min": 1, "max": 10, "step": 1, "default": 1},
        ],
    },
    {
        "id": "semantic",
        "explainer": "Splits at paragraph breaks (blank lines), merges very short paragraphs (headers, page numbers) into their neighbors, then packs paragraphs into chunks up to the size cap. Works best on documents with real paragraph structure; on text extracted without line breaks it degrades to one giant piece and falls back to sentence splitting.",
        "name": "Semantic Paragraph Chunking",
        "description": "Splits at paragraph boundaries (double newlines), merging short paragraphs together. Best for well-structured documents with clear paragraph breaks.",
        "icon": "🧠",
        "params": [
            {"key": "chunk_size", "label": "Max Chunk Size (chars)", "type": "slider", "min": 200, "max": 3000, "step": 100, "default": 800},
            {"key": "min_paragraph_length", "label": "Min Paragraph Length (chars)", "type": "slider", "min": 20, "max": 300, "step": 10, "default": 50},
        ],
    },
]


def chunk_text(
    text: str,
    strategy: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    min_sentences: int = 1,
    min_paragraph_length: int = 50,
) -> List[str]:
    """
    Chunk text using the specified strategy.

    Args:
        text: The full document text.
        strategy: One of 'fixed', 'recursive', 'sentence', 'semantic'.
        chunk_size: Target/max chunk size in characters.
        chunk_overlap: Overlap between chunks (fixed & recursive only).
        min_sentences: Min sentences per chunk (sentence strategy only).
        min_paragraph_length: Min paragraph chars to keep (semantic only).

    Returns:
        List of text chunks.
    """
    if strategy == "fixed":
        return _fixed_chunking(text, chunk_size, chunk_overlap)
    elif strategy == "recursive":
        return _recursive_chunking(text, chunk_size, chunk_overlap)
    elif strategy == "sentence":
        return _sentence_chunking(text, chunk_size, min_sentences)
    elif strategy == "semantic":
        return _semantic_chunking(text, chunk_size, min_paragraph_length)
    else:
        raise ValueError(f"Unknown chunking strategy: {strategy}")


def _fixed_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Simple fixed-size character chunking with sliding window overlap."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        step = chunk_size - overlap if overlap < chunk_size else chunk_size
        start += max(step, 1)  # Prevent infinite loop
    return chunks


def _recursive_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Recursively split using a hierarchy of separators: \\n\\n → \\n → . → space."""
    separators = ["\n\n", "\n", ". ", " "]
    chunks = _recursive_split(text, separators, chunk_size)

    # Apply overlap by repeating trailing text from the previous chunk. The overlap is
    # budgeted so the combined chunk never exceeds chunk_size, and the tail is snapped
    # to a word boundary so we never prepend a partial word.
    if overlap > 0 and len(chunks) > 1:
        overlapped = [chunks[0]]
        for i in range(1, len(chunks)):
            prev = chunks[i - 1]
            budget = min(overlap, max(chunk_size - len(chunks[i]) - 1, 0))
            tail = prev[-budget:] if budget > 0 else ""
            if tail and budget < len(prev) and not prev[-budget - 1].isspace() and not tail[0].isspace():
                # The cut landed mid-word, drop the partial leading word.
                tail = tail.split(" ", 1)[1] if " " in tail else ""
            tail = tail.strip()
            overlapped.append(f"{tail} {chunks[i]}" if tail else chunks[i])
        return overlapped

    return chunks


def _recursive_split(
    text: str, separators: List[str], chunk_size: int
) -> List[str]:
    """Helper for recursive splitting."""
    if len(text) <= chunk_size:
        return [text.strip()] if text.strip() else []

    # Try each separator
    for sep in separators:
        parts = text.split(sep)
        if len(parts) > 1:
            chunks = []
            current = ""
            for part in parts:
                candidate = current + sep + part if current else part
                if len(candidate) <= chunk_size:
                    current = candidate
                else:
                    if current.strip():
                        chunks.append(current.strip())
                    # If a single part exceeds chunk_size, recurse with next separator
                    if len(part) > chunk_size:
                        remaining_seps = separators[separators.index(sep) + 1:]
                        if remaining_seps:
                            sub = _recursive_split(part, remaining_seps, chunk_size)
                            chunks.extend(sub)
                            current = ""
                        else:
                            # Final fallback, hard split
                            chunks.extend(_fixed_chunking(part, chunk_size, 0))
                            current = ""
                    else:
                        current = part
            if current.strip():
                chunks.append(current.strip())
            return chunks

    # Fallback to fixed chunking
    return _fixed_chunking(text, chunk_size, 0)


def _sentence_chunking(text: str, chunk_size: int, min_sentences: int = 1) -> List[str]:
    """Split at sentence boundaries, grouping sentences until chunk size is reached."""
    # Split into sentences using punctuation followed by whitespace
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [s.strip() for s in sentences if s.strip()]

    chunks = []
    current_chunk = ""
    sentence_count = 0

    for sentence in sentences:
        candidate = current_chunk + " " + sentence if current_chunk else sentence

        # The min-sentence floor takes precedence over the size cap: a chunk is never
        # flushed before it holds min_sentences sentences, even if that overshoots
        # chunk_size. (Otherwise min_sentences would have no observable effect.)
        if len(candidate) <= chunk_size or (current_chunk and sentence_count < min_sentences):
            current_chunk = candidate
            sentence_count += 1
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk = sentence
                sentence_count = 1
            else:
                # Single sentence exceeds chunk_size, keep it whole
                chunks.append(sentence.strip())
                current_chunk = ""
                sentence_count = 0

    if current_chunk:
        chunks.append(current_chunk.strip())

    return chunks


def _semantic_chunking(text: str, chunk_size: int, min_paragraph_length: int = 50) -> List[str]:
    """
    Split by paragraph boundaries (double newlines), merging short paragraphs.
    Filters out paragraphs shorter than min_paragraph_length.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]

    # Filter very short paragraphs (headers, page numbers, etc.) by merging with next
    merged_paragraphs = []
    buffer = ""
    for para in paragraphs:
        if len(para) < min_paragraph_length and buffer:
            buffer += "\n\n" + para
        elif len(para) < min_paragraph_length:
            buffer = para
        else:
            if buffer:
                merged_paragraphs.append(buffer + "\n\n" + para)
                buffer = ""
            else:
                merged_paragraphs.append(para)
    if buffer:
        if merged_paragraphs:
            merged_paragraphs[-1] += "\n\n" + buffer
        else:
            merged_paragraphs.append(buffer)

    # Now group into chunks within size limit
    chunks = []
    current = ""

    for para in merged_paragraphs:
        candidate = current + "\n\n" + para if current else para
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current.strip())
            # If paragraph itself is too long, split it at sentences
            if len(para) > chunk_size:
                sub_chunks = _sentence_chunking(para, chunk_size, 1)
                chunks.extend(sub_chunks)
                current = ""
            else:
                current = para

    if current:
        chunks.append(current.strip())

    return chunks
