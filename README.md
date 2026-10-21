# Goal recognition on simulated grid worlds

Compares a UVFA-style MLP value regressor (baseline) with a LightGBM regressor (improved) for recognising the intended goal from the first half of a noisy grid-world trajectory.

## Method
- 1500 seeded 12x12 grid worlds (25% obstacles), 4 candidate goals each, one noisy shortest-path trajectory per map.
- 70/30 episode split (stratified by true goal, unseen maps in test), seed 42.
- Goal prediction = argmin of predicted remaining steps over the 4 candidates at the end of the observed prefix.
- Both models are tuned with 3-fold GroupKFold on the training set only, then refit for seeds 42–44; the test set is used once.

## Changes in this version
- The improved model adds cross-candidate features: each goal's distance, heading and progress is compared with the other candidate goals on the same map at the same time step.
- The improved LightGBM uses 400 trees and reg_lambda=1.0.
- The baseline, data generation, split and evaluation are unchanged.

## Run
```
pip install -r requirements.txt
python analysis.py
```
Outputs `results.json` and `figures/accuracy_comparison.png`.

## Caveat
Simulated data only; results are not comparable to the GRUF paper.

## Reference

This project is motivated by the following paper:

> F Meneguzzi, B Nageris, L Amado, R Mirsky (2026). *Online Goal Recognition: When Inference-Time Computation is Not The (Full) Answer*. assets-eu.researchsquare.com. https://assets-eu.researchsquare.com/files/rs-8982676/v1_covered_fd9ab69a-1d28-4a35-bff5-1feb9b98aef9.pdf?c=1779434899

Paper author: Prof. Felipe Meneguzzi, University of Aberdeen (felipe.meneguzzi@abdn.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
