"""A small byte-level BPE tokenizer shared by all three domains.

Byte-level means every string can be encoded from the 256 byte symbols, so an
unknown token can never appear. BPE merges then make common words shorter.
"""

from pathlib import Path

from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from tnms.config import TOKENIZER_PATH

VOCAB_SIZE = 2048

# The order fixes the ids: <pad> is 0, <bos> is 1, and so on. The tests check this.
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<img>", "</img>", "<aud>", "</aud>"]
PAD, BOS, EOS, IMG, IMG_END, AUD, AUD_END = range(len(SPECIAL_TOKENS))


def train_tokenizer(texts, path: Path = TOKENIZER_PATH) -> Tokenizer:
    """Train BPE on an iterable of strings and save it to `path`."""
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=False,
    )
    tok.train_from_iterator(texts, trainer=trainer)
    tok.save(str(path))
    return tok


def load_tokenizer(path: Path = TOKENIZER_PATH) -> Tokenizer:
    return Tokenizer.from_file(str(path))
