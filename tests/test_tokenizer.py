from tnms.data import cifar100, speechcommands
from tnms.data.tokenizer import SPECIAL_TOKENS, VOCAB_SIZE


def test_special_tokens_have_stable_ids(tokenizer):
    for i, tok in enumerate(SPECIAL_TOKENS):
        assert tokenizer.token_to_id(tok) == i
    assert tokenizer.get_vocab_size() == VOCAB_SIZE


def test_labels_and_words_encode_cleanly(tokenizer):
    """Byte-level BPE has no <unk>, so a clean round trip proves nothing was lost."""
    words = [f.replace("_", " ") for f in cifar100.FINE_TO_COARSE]
    words += [c.replace("_", " ") for c in cifar100.COARSE_TO_FINE]
    words += speechcommands.WORDS
    for w in words:
        ids = tokenizer.encode(w).ids
        assert len(ids) >= 1
        assert all(i >= len(SPECIAL_TOKENS) for i in ids)
        assert tokenizer.decode(ids) == w
