# tiny-native-multimodal-scaling

A small study of scaling laws and data mixtures for an encoder-free multimodal transformer. One decoder-only model is trained from scratch on three kinds of data at once: CIFAR-100 images to captions, Speech Commands audio to the spoken word, and TinyStories text. There is no pretrained or separate vision or audio encoder. Image and spectrogram patches go through a single linear layer straight into the shared transformer. This is a toy-scale study inspired by two Apple papers. It does not replicate or compare against their results.

**Status: work in progress. Results pending.** No findings are reported until the runs exist.

## Setup

Requires Python 3.11 and an NVIDIA GPU for the real experiments. The smoke run works on CPU.

```
python -m venv .venv
.venv/Scripts/activate            # Linux/macOS: source .venv/bin/activate
make setup                        # installs pinned packages and this repo
make data                         # about 3.5 GB download, about 9 GB on disk
make test
make smoke
```

Without `make` (for example on Windows), run the commands from the `Makefile` directly, for example `python scripts/prepare_data.py`.

Downloaded data and run outputs go to `./data` and `./runs` by default. Set `TNMS_DATA_DIR` and `TNMS_RUNS_DIR` to put them somewhere else.

## License

MIT. Iyed Mdimegh, INSAT.
