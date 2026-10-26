from collections import defaultdict

import pytest

from cs336_basics.bpe import apply_merge_token, compiled_PAT, count_pair, train_bpe, word_to_byte_tuple


@pytest.mark.parametrize(
    "text",
    [
        "aaaa aaaa aaa aa a",
        "abab abab baba ab abac abcabc bcabc",
        "猪🙂猪🙂 猪 猪🙂<|endoftext|>ababaaaa",
        "<|endoftext|><|endoftext|>",
    ],
)
def test_incremental_counts_match_full_recount(tmp_path, text):
    source = tmp_path / "corpus.txt"
    source.write_text(text, encoding="utf-8")
    expected_vocab = {index: bytes([index]) for index in range(256)}
    expected_vocab[256] = b"<|endoftext|>"
    word_counts = defaultdict(int)
    for segment in text.split("<|endoftext|>"):
        for match in compiled_PAT.finditer(segment):
            word_counts[word_to_byte_tuple(match.group())] += 1

    expected_merges = []
    while len(expected_vocab) < 280:
        pair_counts = count_pair(word_counts)
        if not pair_counts:
            break
        best_pair = max(pair_counts, key=lambda pair: (pair_counts[pair], pair))
        expected_merges.append(best_pair)
        expected_vocab[len(expected_vocab)] = best_pair[0] + best_pair[1]
        word_counts = apply_merge_token(word_counts, best_pair)

    vocab, merges = train_bpe(source, 280, ["<|endoftext|>"])
    assert vocab == expected_vocab
    assert merges == expected_merges
