import pickle
import regex
from cs336_basics.bpe import compiled_PAT, word_to_byte_tuple


class Tokenizer:
    def __init__(
        self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None
    ):
        self.vocab = vocab.copy()
        self.merges = merges.copy()
        self.special_tokens = []
        if special_tokens is not None:
            self.special_tokens.extend(special_tokens)

        tokens_byte = [token.encode("utf-8") for token in self.special_tokens]
        for token_bytes in tokens_byte:
            if token_bytes not in self.vocab.values():
                self.vocab[max(self.vocab, default=-1) + 1] = token_bytes

        self.token_to_id = {token_bytes: token_id for token_id, token_bytes in self.vocab.items()}

        self.merge_rank = {pair: rank for rank, pair in enumerate(merges)}

        self.special_pattern = None

        if self.special_tokens:
            escaped_tokens = []

            for token in sorted(self.special_tokens, key=len, reverse=True):
                escaped_tokens.append(regex.escape(token))

            pattern = "(" + "|".join(escaped_tokens) + ")"
            self.special_pattern = regex.compile(pattern)

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        with open(vocab_filepath, "rb") as f:
            vocab = pickle.load(f)

        with open(merges_filepath, "rb") as f:
            merges = pickle.load(f)
        return cls(vocab, merges, special_tokens)

    def encode(self, text):
        if self.special_pattern is None:
            chunks = [text]
        else:
            chunks = self.special_pattern.split(text)

        final_id = []
        for chunk in chunks:
            if not chunk:
                continue
            if chunk in self.special_tokens:
                final_id.append(self.token_to_id[chunk.encode("utf-8")])
            else:
                for match in compiled_PAT.finditer(chunk):
                    tokens = list(word_to_byte_tuple(match.group()))

                    while len(tokens) > 1:
                        best_pair = None
                        best_rank = float("inf")

                        for i in range(len(tokens) - 1):
                            pair = (tokens[i], tokens[i + 1])

                            if pair in self.merge_rank:
                                rank = self.merge_rank[pair]

                                if rank < best_rank:
                                    best_pair = pair
                                    best_rank = rank

                        if best_pair is None:
                            break

                        merged_token = best_pair[0] + best_pair[1]
                        new_tokens = []
                        i = 0

                        while i < len(tokens):
                            if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best_pair:
                                new_tokens.append(merged_token)
                                i += 2
                            else:
                                new_tokens.append(tokens[i])
                                i += 1

                        tokens = new_tokens

                    for token_bytes in tokens:
                        final_id.append(self.token_to_id[token_bytes])

        return final_id

    def encode_iterable(self, iterable):
        buffer = ""

        for text in iterable:
            if not text:
                continue

            buffer += text

            safe_end = len(buffer)

            for special in self.special_tokens:
                max_length = min(len(special) - 1, len(buffer))

                for length in range(max_length, 0, -1):
                    if buffer.endswith(special[:length]):
                        safe_end = min(
                            safe_end,
                            len(buffer) - length,
                        )
                        break

            if self.special_pattern is None:
                chunks = [buffer]
            else:
                chunks = self.special_pattern.split(buffer)

            cut = 0
            previous_start = None
            offset = 0

            for chunk in chunks:
                if chunk in self.special_tokens:
                    start = offset

                    if previous_start is not None and previous_start <= safe_end:
                        cut = previous_start

                    previous_start = start

                else:
                    for match in compiled_PAT.finditer(chunk):
                        start = offset + match.start()

                        if previous_start is not None and previous_start <= safe_end:
                            cut = previous_start

                        previous_start = start

                offset += len(chunk)

            if cut > 0:
                for token_id in self.encode(buffer[:cut]):
                    yield token_id

                buffer = buffer[cut:]

        for token_id in self.encode(buffer):
            yield token_id

    def decode(self, ids: list[int]) -> str:
        byte_parts = []

        for token_id in ids:
            byte_parts.append(self.vocab[token_id])

        all_bytes = b"".join(byte_parts)

        return all_bytes.decode("utf-8", errors="replace")
