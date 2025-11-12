<%*
// Ask for experiment name (asynchronously)
const name = await tp.system.prompt("Experiment name or ID?");
%>
---
type: experiment
title: "<% name %>"
date: <% tp.date.now("YYYY-MM-DD") %>
tags: [experiment, code, results]
---

# Experiment – <% name %>

## 🧰 Config
- Simulation type:
- Dataset:
- Hyperparams:
- Seed:
- Notes:

## ⚙️ Procedure
1. 
2. 

## 📈 Results
- Training Loss:
- Validation Loss:
- Metrics:
- Plots: `![[results_plot.png]]`

## 🧠 Observations
- 

## 🔄 Next Steps
- 
