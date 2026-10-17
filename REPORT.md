# Report: scaling a tiny encoder-free multimodal transformer

Iyed Mdimegh, INSAT

## Question

I wanted to see, at a scale I can run on a laptop GPU, whether a single decoder-only transformer that takes raw image and audio patches can be described by the usual scaling laws, and whether a fitted law can pick a better data mixture than the uniform one. This is a toy-scale study inspired by two Apple papers. It does not replicate or compare against their results.

## What I did

I trained one model family on three datasets at once: CIFAR-100 images with templated captions, Speech Commands words with templated captions, and TinyStories. Patches enter the transformer through a single linear layer, attend to each other in both directions, and the loss is on text only. I swept the learning rate for five sizes (98k to 14M non-embedding parameters), trained a 5x5 grid of sizes and token budgets (5M to 80M) with a fresh cosine schedule per run, and added two extra seeds on three cells to measure noise. I fitted L(N, D) = E + A/N^alpha + B/D^beta in log space with a Huber loss, put bootstrap intervals on the parameters, and tested the law on held-out runs against a simple per-row power law. For mixtures I trained 13 runs at one size and budget, fitted a small per-domain law on validation loss, checked it with leave-three-out cross-validation, and trained the predicted optima again with a new seed. Finally I compared early fusion with late fusion at exactly the same parameter count.

## What I found

Text scaled cleanly. Its loss fell from 3.38 to 1.86 across the grid and the law fitted it within 1.2% on average.

Image and audio did not, and this was the main lesson. With templated captions the model can get a fair loss by learning the caption templates and the label frequencies without looking at the input. That puts the loss on a "no-input floor" that I can compute exactly. Small models and small budgets sit on that floor and only later start to read the patches. The smallest model never did, for either modality. Audio shows it most clearly: flat at the floor, then a sharp drop once the model has seen enough clips. A smooth power law cannot follow that kink, so the audio fit was off by 8.6% on average.

The extrapolation test was the part that did not work. Fitting all sizes but the largest and predicting the largest, or all budgets but the largest and predicting that, the joint law lost to a simple one-term power law fitted per row in 7 of 8 comparisons. The bootstrap intervals on the exponents were wide, and the compute-optimal tokens-per-parameter ratio the law suggests lies far outside the grid. With 25 points and two regimes in the data, I do not think the fitted exponents mean much beyond this setup.

The mixture law worked better than I expected. It beat a linear model in cross-validation for all three domains, and its predicted best mixture for the mean loss, (0.09, 0.34, 0.57), came within 0.003 of its predicted validation loss on a fresh seed. On test it beat the uniform mixture by about five seed standard deviations. The mixture that is best for image alone was predicted just as well, but it costs a lot of text loss and repeats the image set more than nine times.

Late fusion, with a small encoder per modality and a shorter decoder at the same parameter count, was clearly worse than early fusion at both sizes I tried, by far more than the seed noise, and most of all on audio (mean test loss 0.915 early against 0.997 late for the larger model). I do not read this as "early fusion wins". Matching parameters took four layers away from the text path, and late fusion also used less compute per token. A fair comparison at equal compute needs more runs.

## What I checked

Before trusting any number I ran controls. The model can memorise a single batch down to the exact entropy of its first token. Swapping in another example's patches raises the caption loss a lot, so the model reads its input. A model trained on shuffled image labels stays at the floor. Train and validation loss stay within 0.012 of each other. One control caught a bug in itself: the first patch-swap control paired neighbouring clips, and Speech Commands files are sorted by word, so the swap barely changed anything. Learning rates and mixtures were chosen on validation loss only. Every number in the README is checked against the result tables by a script.

## What I would change

Captions with real variety, so the no-input floor is low and the model has to read the input from the start. More seeds per cell and more sizes and budgets, so the fit has enough points to tell the regimes apart. A learning rate swept per budget, since the 80M check showed drift for the largest model. A joint mixture law in N, D and the shares, which needs many more runs than I could afford here.
