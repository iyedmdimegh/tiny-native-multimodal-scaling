"""Download all datasets, train the tokenizer and build the caches.

Safe to run again: every step is skipped when its output already exists.
"""

from itertools import chain

from tnms.config import DATA_DIR, TOKENIZER_PATH
from tnms.data import cifar100, speechcommands, tinystories
from tnms.data.cifar100 import Cifar100
from tnms.data.tokenizer import load_tokenizer, train_tokenizer


def label_texts():
    """Every caption the image and audio domains can produce, so the tokenizer
    sees all label words and template words at least once."""
    for fine in cifar100.FINE_TO_COARSE:
        for t in range(len(cifar100.TEMPLATES)):
            yield cifar100.caption(fine, t)
    for word in speechcommands.WORDS:
        for template in speechcommands.TEMPLATES:
            yield template.format(word=word)


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"data directory: {DATA_DIR}")

    print("CIFAR-100")
    Cifar100("train")

    print("TinyStories")
    raw = tinystories.load_raw()
    if not TOKENIZER_PATH.exists():
        print("training tokenizer")
        train_tokenizer(chain(tinystories.tokenizer_corpus(raw), label_texts()))
    tinystories.prepare(raw, load_tokenizer())

    print("Speech Commands")
    speechcommands.prepare()
    print("done")


if __name__ == "__main__":
    main()
