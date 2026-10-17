# Design choices

Each entry says what was chosen and why, in a few sentences.

## Data

**CIFAR-100 source.** The images come from the copy that the University of Toronto publishes on the Hugging Face Hub (`uoft-cs/cifar100`, pinned revision), not from `torchvision.datasets.CIFAR100`. The original server sent about 1 kB/s from my network. It is the same 50,000 train and 10,000 test images with the same labels, and a test checks the superclass map against the coarse label stored with every test image.

**Fixed CIFAR-100 split.** The 50,000 training images are split into 45,000 train and 5,000 validation with a fixed seed (0), the same for every run. The test set is only used for reported numbers, never for choosing anything.

**CIFAR normalisation from the train split only.** The per-channel mean and std come from the 45,000 training images. Using validation or test images for them would leak a little information about the held-out sets.

**Coarse class names as captions.** The 20 superclass names are the standard ones (for example "vehicles 1", "medium mammals") with underscores replaced by spaces. Some read awkwardly in a caption, but they are the official labels and they are fixed by a test.

**Audio time shift on the cached spectrogram.** Computing spectrograms on the fly would make audio the slowest domain by far. So each clip is cached once with 100 ms of silence on both sides (121 frames), and the random shift picks which 96 frames to read. One frame is 10 ms, so this is the same as shifting the waveform, rounded to 10 ms.

**Audio crop.** A 1 second clip gives 101 frames. We keep the first 96 so the frame count divides by the 8-frame patch. The dropped 5 frames are the last 50 ms of the clip.

**Audio normalisation.** One global mean and std over all train log-mel values (unshifted). Per-band statistics would also work, but a single pair is simpler and enough for a toy study.

**Reading WAV files with scipy.** torchaudio's file loading needs an extra audio backend on Windows. `scipy.io.wavfile` reads 16-bit WAV directly, and torchaudio is still used for the download and the mel filterbank.

**TinyStories splits.** TinyStories only has train and validation. Our text validation set is the last 20,000 stories of the HF train split, which the training stream never reaches. The HF validation split is our text test set, so all three domains have a val set for choices and a test set for reports.

**40M text training tokens.** The largest text need is 0.8 of a 40M-token mixture run (32M) or a third of an 80M grid run (27M). 40M covers both, so text is never repeated. Image and audio do repeat, see the epoch counts in the README.

**Text example ends with `<eos>`.** The spec puts `<eos>` at the end of every 128-token text window, even when the window cuts a story in the middle. The model learns that position 127 is always `<eos>`, which makes that one prediction nearly free. It is one token out of 127 and the same for every run, so it shifts all text losses by a small constant and does not change the scaling trends.

**Tokenizer.** Byte-level BPE with 2048 entries. Byte-level means any string can be encoded, so `<unk>` can never appear. It is trained on about 20 MB of TinyStories train text plus every possible image and audio caption, so label words get their own merges.

## Sequences and loss

**Loss only on text.** Next-token loss is computed on caption tokens (and their `<eos>`) for image and audio, and on every token after `<bos>` for text. Patches are continuous vectors with no token to predict, so they are treated as input context and language is the target.

**Training loss weighting.** The training loss is the mean over all loss tokens in the batch. A text example has 127 loss tokens and a caption about 10, so text gives most of the gradient even at equal token share. Evaluation reports each domain separately, so this does not hide anything.

**Prefix-LM mask.** The modality open token, the patches and the close token attend to each other in both directions. An image has no natural left-to-right order, so letting patches see each other is the natural choice. Everything else is causal.

**Mixture by token share.** h is the share of tokens, not of examples. Domain i is drawn with probability proportional to h_i divided by its mean sequence length, which makes the expected token share equal to h_i. A test checks this within 5%.

## Model

**N excludes embeddings.** N counts the transformer blocks and the final norm. The token embedding (tied with the output), position embedding and the two patch projections are left out. This follows the common convention in scaling-law work. The patch projections are tiny (48 x d and 64 x d) anyway.

**No weight decay on embedding-like layers.** Weight decay applies to the block matrices only. The token and position embeddings and the patch projections, which are the parameters outside N, get no decay, and neither do the norms.

## Training

**Warmup length.** Warmup is 5% of the steps with a minimum of 100, as in the plan. For very short runs (the smoke preset) 100 steps would be longer than the run, so warmup is capped at 10% of the steps. This cap never triggers for runs of 1,000 steps or more, which covers every grid run.

**Schedule from the first 200 batches.** Batches differ in token count, so the number of steps for a budget D is not known in advance. We draw the first 200 batches, take their mean token count, and fix the cosine schedule from that. Those same batches are then used for training, so the data order is unchanged.

**bf16 autocast.** The local GPU (RTX 3050, compute 8.6) supports bf16, which needs no loss scaling. On GPUs without bf16 the code falls back to fp16 with a GradScaler.

**No torch.compile.** It needs Triton, which has no official Windows build, and the brief only allows the listed packages. Everything runs in eager mode. The benchmark in `results/benchmark.json` is measured that way.

**LR selection.** For each size, LR is chosen by the lowest validation loss averaged over the three domains, at D = 10M tokens with one seed. If the best LR is the smallest or largest tried, the sweep adds half or double that value and reruns, up to two times. The test split plays no part.

**The grid reads the swept LR.** The full preset says `lr: sweep` and the grid runner reads the best LR per size from `results/lr_sweep.csv`. Copying numbers by hand into a config is an easy place for a mistake to hide.

**Only the full preset writes to results/.** Smoke and small presets write their tables next to their runs, so a quick check can never overwrite the committed numbers.

**Noise seeds and the LR check share the grid runner.** One resumable queue runs the grid, the extra seeds and the 80M LR neighbours, ordered by budget. It can be left alone for hours and restarted after a crash without losing finished runs.

## Fitting

**Fit on seed 0 only.** The three noise cells have extra seeds, but the fit uses the seed-0 run of every cell, so each cell has the same weight. The extra seeds only give the error bars and the noise table.

**Fit on the logged D, not the budget.** A run stops once it passes its budget, so the real D is a little above it. The fit uses the real value.

**Bootstrap starts.** The full fit tries all 3,125 starting points, which takes about two minutes. Doing that 200 times per target is too slow, so each bootstrap refit starts from the full-data optimum plus 30 random grid points. A refit could in principle miss a far-away optimum that the full grid would find, which would make the intervals a bit too narrow.

**Baseline for the holdouts.** The brief asks for a naive baseline. For the largest-N holdout it is a separate law L = E + A / N^alpha for each D, fitted on S1 to S4. For the largest-D holdout it is a separate L = E + B / D^beta for each size, fitted on the four smaller budgets. Both use the same Huber fit as the main law.

**Empirical frontier.** Runs are sorted by compute 6ND. A run is on the frontier if its loss is lower than every run with less compute.

## Controls

**Shuffled labels are permuted once per run.** Every training image gets the label of another image, fixed for the whole run. The caption distribution stays the same, so the model can still learn the label frequencies but nothing about which image has which label.

**Sample prompts.** The five prompts in `scripts/run_controls.py` were written before generating anything, and all five outputs go into `results/samples.txt`.

**"Below the floor" needs a margin.** `grid.csv` flags a strict comparison, test loss under the no-input floor. Several small cells are under it by less than 0.001, which is noise, not reading the input. In the write-up a domain only counts as reading the input when its loss is below the floor by more than twice the largest seed standard deviation of that domain over the three noise cells. This rule was fixed before the 40M and 80M runs had finished.
