---
title: "Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models"
year: 2024
authors: Louis Sharrock, Jack Simons, Song Liu, Mark Beaumont
citekey: sharrockSequentialNeuralScore2024
tags:
  - zotero
Optional: store related links for Dataview queries too
related:
  - 
---

[1]

L. Sharrock, J. Simons, S. Liu, and M. Beaumont, “Sequential Neural Score Estimation: Likelihood-Free Inference with Conditional Score Based Diffusion Models,” June 03, 2024, _arXiv_: arXiv:2210.04872. doi: [10.48550/arXiv.2210.04872](https://doi.org/10.48550/arXiv.2210.04872).

- zotero: [zotero://select/library/items/886HXA5T](zotero://select/library/items/886HXA5T)
- url: http://arxiv.org/abs/2210.04872
- pdf: [Preprint PDF](file://C:\Users\aritr\Zotero\storage\BJ2QDPYX\Sharrock%20et%20al.%20-%202024%20-%20Sequential%20Neural%20Score%20Estimation%20Likelihood-Free%20Inference%20with%20Conditional%20Score%20Based%20Diffusion.pdf)
# Abstract
We introduce Sequential Neural Posterior Score Estimation (SNPSE), a score-based method for Bayesian inference in simulator-based models. Our method, inspired by the remarkable success of score-based methods in generative modelling, leverages conditional score-based diffusion models to generate samples from the posterior distribution of interest. The model is trained using an objective function which directly estimates the score of the posterior. We embed the model into a sequential training procedure, which guides simulations using the current approximation of the posterior at the observation of interest, thereby reducing the simulation cost. We also introduce several alternative sequential approaches, and discuss their relative merits. We then validate our method, as well as its amortised, non-sequential, variant on several numerical examples, demonstrating comparable or superior performance to existing state-of-the-art methods such as Sequential Neural Posterior Estimation (SNPE).


# Highlights


# Related
%% begin related %%
- [[Paste another paper title or citekey note here]]
%% end related %%

# Notes
%% begin notes %%
 - Score is the gradient of the log density with respect to the data
 - Uses the concept of diffusion to sample from the posterior
	 - Sample from prior
	 - Add random noise
	 - Try to denoise and make them similar to posterior samples (diffusion)
	 - The diffusion is conditioned 
%% end notes %%

## Suggested (by overlapping tags)
```dataview
LIST
FROM "02_Literature_Notes"
WHERE file.path != this.file.path
AND length(intersection(file.tags, this.file.tags)) > 0
SORT file.mtime desc
LIMIT 10

%% Import Date: 2025-11-06T15:34:50.572+01:00 %%
