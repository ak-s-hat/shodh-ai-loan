# shodh-ai-loan

## Lending Club Loan Approval: Deep Learning & Offline Reinforcement Learning Project

This repository provides the complete codebase, data preprocessing workflows, and model implementations for predicting loan defaults and designing optimal loan approval strategies using Deep Learning (DL) and Offline Reinforcement Learning (RL).

---

## Project Summary

This project aims to:
1. Forecast the likelihood of loan defaults through supervised deep learning models.
2. Develop an offline RL agent (CQL) to discover optimal loan approval strategies.
3. Assess how DL and RL policy approaches differ by comparing financial returns and associated risks.

---

## Repository Structure & Insights

- **DL Model:** Predicts loan defaults using a supervised deep learning model, now with approval threshold selected by grid search to maximize financial return (not just accuracy). Reports both AUC and F1-score, plus confusion matrix visualization.
- **RL Model:** Implements a batch-constrained offline RL policy with the CQL algorithm, with a redesigned reward function that penalizes defaults much more harshly (double principal loss to enforce safety). Policy returns are benchmarked versus DL and other strategies.
- **Feature Engineering:** Includes advanced feature creation such as time-since-delinquency, moving averages for income, frequency encoding for grade, categorical encoding, and robust handling of missing values. Helps capture borrower risk with more granularity.
- **Policy Comparison:** RL agents may approve loans with higher risk if anticipated rewards justify potential losses. All policies (always approve, always deny, random, grade-bucket, tuned DL, RL CQL) are compared to show ROI and risk tradeoff.

---

## How to Run

1. **Preprocessing**
   - Run `data_preprocessing.py` to generate the fully engineered dataset, including all custom features.
2. **Train DL Model**
   - Run `dl_model.py'. The DL model conducts a grid search for the best approval threshold, maximizing ROI on the validation/test set.
3. **Train RL Agent**
   - Run `rl_model.py` to train the CQL offline RL agent. Automatically compares learned RL policy against all baselines including DL-threshold, random, and grade-based.
4. **Visualization**
   - Notebook demo (`shodh-ai.ipynb`) visualizes metrics, confusion matrix, and RL/DL disagreement regions.

---

## Results Interpretation

- Compare output metrics such as average expected return, policy value, AUC, and F1-score.
- RL policy will generally be more conservative after the reward changes (higher penalty on defaults).
- DL threshold is no longer 0.5; it is chosen for max expected profit.
- Policy comparison table and plots in notebook help show key tradeoffs.

---

## Limitations & Future Directions

- RL training uses single-step episodes and does not model full repayment behavior or time-dependent losses.
- Feature set likely still omits some borrower-specific nonlinear effects; future work can expand with bureau data, behavioral predictors, and temporal modeling.
- The full hybrid of supervised learning and RL for policy optimization is an ongoing research direction.

---
