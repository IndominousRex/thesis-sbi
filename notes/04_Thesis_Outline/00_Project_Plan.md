---
type: project_plan
title: "Thesis Roadmap – Simulation-Based Inference for Parameter Density Estimation"
start: 2025-11-04
status: in-progress
tags: [thesis, planning]
---

# 🎓 Thesis Roadmap

**Goal:** Implement and compare amortized SBI methods (NPE & NPSE) for parameter inference in dynamical systems.

---

## 🗓️ Timeline

| Phase                           | Duration        | Status | Goal                                           |
| ------------------------------- | --------------- | ------ | ---------------------------------------------- |
| 1️⃣ Research Foundation         | Nov 04 – Nov 25 | 🔄     | Summarize 4 key papers, identify gaps          |
| 2️⃣ Baseline NPE Implementation | Nov 26 – Dec 31 | ⬜      | Working NPE pipeline & posterior evaluation    |
| 3️⃣ NPSE / Compositional SBI    | Jan 01 – Jan 31 | ⬜      | Implement & compare NPSE                       |
| 4️⃣ Extensions & Ablations      | Feb 01 – Feb 28 | ⬜      | Hidden-state handling & hyperparameter studies |
| 5️⃣ Evaluation & Documentation  | Mar 01 – Mar 31 | ⬜      | Final plots, report writing                    |

---

## ✅ Key Deliverables

- [ ] Literature summary (`04_Thesis_Outline/01_Literature_Summary.md`)
- [ ] NPE baseline notebook
- [ ] NPSE implementation notebook
- [ ] Ablation study results
- [ ] Figures & plots for thesis text
- [ ] Final LaTeX document with bibliography

---

## 📚 Reading Checklist

- [ ] Zammit-Mangion et al. 2025 – Neural Methods for Amortized Inference  
- [ ] Sharrock et al. 2024 – Sequential Neural Score Estimation  
- [ ] Glöckler et al. 2025 – Compositional SBI for Time Series  
- [ ] Rozet & Louppe 2023 – Score-Based Data Assimilation  

---

## 🧠 Weekly Focus (Rolling Log)

| Week   | Focus                           | Notes |
| ------ | ------------------------------- | ----- |
| Week 1 | Literature & conceptual mapping |       |
| Week 2 | NPE baseline setup              |       |
| Week 3 | Run baseline experiments        |       |
| Week 4 | Implement NPSE                  |       |
| Week 5 | NPSE vs NPE comparison          |       |
| Week 6 | Hidden-state extension          |       |
| Week 7 | Ablations & metrics             |       |
| Week 8 | Writeup & plots                 |       |

---

## 📊 Dataview Section

```dataview
TABLE date, file.link AS "Experiment", status
FROM "03_Experiments"
SORT date DESC
LIMIT 10
