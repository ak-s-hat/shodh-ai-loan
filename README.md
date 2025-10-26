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

- **DL Model:** Focuses on prediction accuracy for identifying defaults (reports both AUC and F1-score).
- **RL Model:** Maximizes projected financial gains (evaluates policy value).
- **Policy Comparison:** RL agents may approve loans with higher risk if anticipated rewards justify potential losses.

---

## Limitations & Future Directions

- The current RL setup uses a one-step episode framework, which does not account for extended repayment behaviors.
- Feature set may be insufficient to capture all borrower behavior nuances.
- Future enhancements can include hybrid RL+DL models, additional offline RL algorithms (e.g., SAC, IQL), threshold calibration for DL models, and expanding feature space.
