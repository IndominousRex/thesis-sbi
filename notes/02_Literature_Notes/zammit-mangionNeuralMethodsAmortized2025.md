---
title: "Neural Methods for Amortized Inference"
year: 2025
authors: Andrew Zammit-Mangion, Matthew Sainsbury-Dale, Raphaël Huser
citekey: zammit-mangionNeuralMethodsAmortized2025
tags:
  - zotero
Optional: store related links for Dataview queries too
related:
  - 
---

[1]

A. Zammit-Mangion, M. Sainsbury-Dale, and R. Huser, “Neural Methods for Amortized Inference,” _Annual Review of Statistics and Its Application_, vol. 12, no. 1, pp. 311–335, Mar. 2025, doi: [10.1146/annurev-statistics-112723-034123](https://doi.org/10.1146/annurev-statistics-112723-034123).

- zotero: [zotero://select/groups/6286989/items/CV7BCYFL](zotero://select/groups/6286989/items/CV7BCYFL)
- url: https://www.annualreviews.org/content/journals/10.1146/annurev-statistics-112723-034123
- pdf: [PDF](file://C:\Users\aritr\Zotero\storage\5KT4WQEB\Zammit-Mangion%20et%20al.%20-%202025%20-%20Neural%20Methods%20for%20Amortized%20Inference.pdf)
# Abstract
Simulation-based methods for statistical inference have evolved dramatically over the past 50 years, keeping pace with technological advancements. The field is undergoing a new revolution as it embraces the representational capacity of neural networks, optimization libraries, and graphics processing units for learning complex mappings between data and inferential targets. The resulting tools are amortized, in the sense that, after an initial setup cost, they allow rapid inference through fast feed-forward operations. In this article we review recent progress in the context of point estimation, approximate Bayesian inference, summary-statistic construction, and likelihood approximation. We also cover software and include a simple illustration to showcase the wide array of tools available for amortized inference and the benefits they offer over Markov chain Monte Carlo methods. The article concludes with an overview of relevant topics and an outlook on future research directions.


# Highlights


# Related
%% begin related %%
- [[Paste another paper title or citekey note here]]
- [[Paste another paper title or citekey note here]]
%% end related %%

# Notes
%% begin notes %%
 - Lots of mathematics and equations in the paper. 
 - Didn't understand reverse KL (variational Bayes)
 - 
%% end notes %%

## Suggested (by overlapping tags)
```dataview
LIST
FROM "02_Literature_Notes"
WHERE file.path != this.file.path
AND length(intersection(file.tags, this.file.tags)) > 0
SORT file.mtime desc
LIMIT 10

%% Import Date: 2025-11-05T13:41:42.862+01:00 %%
