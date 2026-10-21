# Goal recognition on simulated grid worlds

Question: can a gradient-boosted goal scorer with engineered per-candidate features match or beat a UVFA-style MLP value regressor at recognising the intended goal from a partial grid-world trajectory?

Data: seeded 12x12 grid worlds with random obstacles, 4 candidate goals per map, and noisy shortest-path trajectories to the intended goal. Only the first 50% of each trajectory is observed. Episodes are split 70/30 (train/test) with unseen maps in the test set.

Baseline: scikit-learn `MLPRegressor` (standardised inputs) predicting remaining steps to each goal; goal = argmin predicted cost. Tuned with 3-fold GroupKFold on the training set only.

Improved: LightGBM binary classifier that predicts whether each candidate is the intended goal, using engineered progress/heading features and cross-candidate relative features. Goal = candidate with the highest predicted probability. Tuned with the same 4-combination, 3-fold GroupKFold budget on the training set only (neg log-loss).

Metric: goal-recognition accuracy on the test episodes, averaged over fits with seeds 42–44. The test set is used once.

Run: `pip install -r requirements.txt && python analysis.py` (CPU only, a few minutes). Outputs `results.json` and `figures/accuracy_comparison.png`.

Caveats: simulated data only, so results are not comparable to the GRUF paper. The sample is modest.

## Reference

This project is motivated by the following paper:

> F Meneguzzi, B Nageris, L Amado, R Mirsky (2026). *Online Goal Recognition: When Inference-Time Computation is Not The (Full) Answer*. assets-eu.researchsquare.com. https://assets-eu.researchsquare.com/files/rs-8982676/v1_covered_fd9ab69a-1d28-4a35-bff5-1feb9b98aef9.pdf?c=1779434899

Paper author: Prof. Felipe Meneguzzi, University of Aberdeen (felipe.meneguzzi@abdn.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
