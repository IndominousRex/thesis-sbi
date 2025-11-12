---
title: "Compositional simulation-based inference for time series"
year: 2025
authors: Manuel Gloeckler, Shoji Toyota, Kenji Fukumizu, Jakob H. Macke
citekey: gloecklerCompositionalSimulationbasedInference2025
tags:
  - zotero
Optional: store related links for Dataview queries too
related:
  - 
---

[1]

M. Gloeckler, S. Toyota, K. Fukumizu, and J. H. Macke, “Compositional simulation-based inference for time series,” Mar. 03, 2025, _arXiv_: arXiv:2411.02728. doi: [10.48550/arXiv.2411.02728](https://doi.org/10.48550/arXiv.2411.02728).

- zotero: [zotero://select/library/items/PGYT4REA](zotero://select/library/items/PGYT4REA)
- url: http://arxiv.org/abs/2411.02728
- pdf: [Preprint PDF](file://C:\Users\aritr\Zotero\storage\2E7JAKT9\Gloeckler%20et%20al.%20-%202025%20-%20Compositional%20simulation-based%20inference%20for%20time%20series.pdf)
# Abstract
Amortized simulation-based inference (SBI) methods train neural networks on simulated data to perform Bayesian inference. While this strategy avoids the need for tractable likelihoods, it often requires a large number of simulations and has been challenging to scale to time series data. Scientific simulators frequently emulate real-world dynamics through thousands of single-state transitions over time. We propose an SBI approach that can exploit such Markovian simulators by locally identifying parameters consistent with individual state transitions. We then compose these local results to obtain a posterior over parameters that align with the entire time series observation. We focus on applying this approach to neural posterior score estimation but also show how it can be applied, e.g., to neural likelihood (ratio) estimation. We demonstrate that our approach is more simulation-efficient than directly estimating the global posterior on several synthetic benchmark tasks and simulators used in ecology and epidemiology. Finally, we validate scalability and simulation efficiency of our approach by applying it to a high-dimensional Kolmogorov flow simulator with around one million data dimensions.


# Highlights


> [!cite]
> Amortized simulation-based inference (SBI) methods train neural networks on simulated data to perform Bayesian inference.
> [Page ](zotero://open-pdf/library/items/2E7JAKT9?page=)


> [!cite]
> As a consequence, such simulators have an inherently Markovian structure, which can be leveraged for efficient inference!
> [Page 2](zotero://open-pdf/library/items/2E7JAKT9?page=2)


# Related
%% begin related %%
- [[Paste another paper title or citekey note here]]
%% end related %%

# Notes
%% begin notes %%
 - Most SBI models assume iid data
 - Uses Markovian property of time series simulators
 - Instead of full trajectories, learn from consecutive states
 - Train local models on short transitions and the aggregate them to form the global posterior
 - Try to understand the factorization equation
 - How it works with noise (using GAUSS or JAC)
 - The proposal needs to be independent of theta and cover all states
 - More efficient and better scalability
 - Standard inference models suffer from slow sampling and potential failure modes such as robustly handling multimodality
 - 

NFSE idea:
  1. Most simulators for time series data are based on differential equations that model dynamics of the system through transitions over time or model processes which are iteratively updated at each time step.
  2. These simulators have an inherently Markovian structure which can be used for efficient inference.
  3. NPE and other global target distribution estimators require a large number of simulations which is very expensive.
  4. NFSE mitigates the problem using Markov Factorization of the forward model
	  1. Since the simulator is specified by the state transition probabilities, they contain information about the parameters.
	  2. From a dataset of single step transitions, it is therefore possible to recover the global target distribution.
  5. 
%% end notes %%

## Suggested (by overlapping tags)
```dataview
LIST
FROM "02_Literature_Notes"
WHERE file.path != this.file.path
AND length(intersection(file.tags, this.file.tags)) > 0
SORT file.mtime desc
LIMIT 10

%% Import Date: 2025-11-06T12:48:51.033+01:00 %%
