from collections import defaultdict
from os import PathLike
import sys

import regex

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
compiled_PAT = regex.compile(GPT2_PAT)


def load_file(file_name: str):
    with open(file_name, encoding="utf-8") as f:
        return f.read()


def word_to_byte_tuple(token):
    word = []

    for value in token.encode("utf-8"):
        word.append(bytes([value]))

    return tuple(word)


def count_pair(word_counts):
    pair_counts = defaultdict(int)

    for word, frequency in word_counts.items():
        for pair in zip(word, word[1:]):
            pair_counts[pair] += frequency

    return pair_counts


def _merge_word(words, best_pair):
    merged_tokens = best_pair[0] + best_pair[1]
    new_word = []
    i = 0
    while i < len(words):
        if i + 1 < len(words) and (words[i], words[i + 1]) == best_pair:
            new_word.append(merged_tokens)
            i += 2
        else:
            new_word.append(words[i])
            i += 1

    return tuple(new_word)


def apply_merge_token(word_counts, best_pair):
    merged_word_count = defaultdict(int)

    for words, freq in word_counts.items():
        if best_pair[0] not in words or best_pair[1] not in words:
            merged_word_count[words] += freq
            continue

        merged_word_count[_merge_word(words, best_pair)] += freq

    return merged_word_count


def train_bpe(
    input_path: str | PathLike[str],
    vocab_size: int,
    special_tokens: list[str],
    *,
    progress: bool = False,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    process = None
    if progress:
        import psutil
        from tqdm import tqdm

        process = psutil.Process()

    # Initialize vocabulary
    vocab = {}
    for i in range(256):
        vocab[i] = bytes([i])

    specials = []
    for token in special_tokens:
        if not token:
            raise ValueError("Special tokens cannot be empty!")

        if token not in specials:
            specials.append(token)
            token_bytes = token.encode("utf-8")

            if token_bytes not in vocab.values():
                vocab[len(vocab)] = token_bytes

    if vocab_size < len(vocab):
        raise ValueError("vocab_size is smaller than initial vocabulary")

    # Read text
    if progress:
        print(f"Reading training text: {input_path}", file=sys.stderr, flush=True)
    text = load_file(input_path)
    segments = [text]

    if progress:
        print("Preparing special-token boundaries...", file=sys.stderr, flush=True)
    if specials:
        patterns = []

        for token in sorted(specials, key=len, reverse=True):
            patterns.append(regex.escape(token))

        segments = regex.split("|".join(patterns), text)

    # Pre-tokenization
    word_counts = defaultdict(int)

    pretoken_progress = segments
    if progress:
        pretoken_progress = tqdm(segments, desc="Pre-tokenization", unit="segment", mininterval=5, ascii=True)
    for index, segment in enumerate(pretoken_progress, start=1):
        for match in compiled_PAT.finditer(segment):
            word = word_to_byte_tuple(match.group())
            word_counts[word] += 1

        if progress and index % 10000 == 0:
            pretoken_progress.set_postfix(rss_GiB=f"{process.memory_info().rss / 1024**3:.2f}", refresh=False)

    if progress:
        print(
            f"Pre-tokenization complete: {len(word_counts):,} unique pre-tokens; "
            f"RSS {process.memory_info().rss / 1024**3:.2f} GiB",
            file=sys.stderr,
            flush=True,
        )

    # Merge
    merges = []
    words = list(word_counts)
    frequencies = list(word_counts.values())
    pair_counts = count_pair(word_counts)
    pair_to_words = defaultdict(set)

    if progress:
        print("Indexing adjacent pairs...", file=sys.stderr, flush=True)
    for word_id, word in enumerate(words):
        for pair in zip(word, word[1:]):
            pair_to_words[pair].add(word_id)
    del word_counts

    merge_progress = range(vocab_size - len(vocab))
    if progress:
        merge_progress = tqdm(merge_progress, desc="BPE merges", unit="merge", mininterval=5, ascii=True)
    for _ in merge_progress:
        if not pair_counts:
            break

        best_pair = None
        best_count = -1

        for pair, count in pair_counts.items():
            if count > best_count:
                best_pair = pair
                best_count = count
            elif count == best_count and pair > best_pair:
                best_pair = pair

        merges.append(best_pair)
        vocab[len(vocab)] = best_pair[0] + best_pair[1]

        affected_words = pair_to_words.pop(best_pair)
        for word_id in affected_words:
            word = words[word_id]
            frequency = frequencies[word_id]
            old_pairs = list(zip(word, word[1:]))

            for pair in old_pairs:
                pair_counts[pair] -= frequency
                if pair_counts[pair] == 0:
                    del pair_counts[pair]

            for pair in set(old_pairs):
                pair_to_words[pair].discard(word_id)
                if not pair_to_words[pair]:
                    del pair_to_words[pair]

            word = _merge_word(word, best_pair)
            words[word_id] = word
            for pair in zip(word, word[1:]):
                pair_counts[pair] += frequency
                pair_to_words[pair].add(word_id)

        if progress and len(merges) % 100 == 0:
            merge_progress.set_postfix(rss_GiB=f"{process.memory_info().rss / 1024**3:.2f}", refresh=False)

    return vocab, merges
