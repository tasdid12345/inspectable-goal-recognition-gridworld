# Goal recognition on grid worlds: MLP value regressor vs. LightGBM

Online goal recognition has to infer which goal an agent is heading for from a partial trajectory, and recent work such as "Online Goal Recognition: When Inference-Time Computation is Not The (Full) Answer" (2026) proposes goal-agnostic recognisers built on universal value function approximators. This experiment asks whether a simpler, gradient-boosted scorer with engineered progress features can match a UVFA-style MLP at the same task, using a controlled simulated setting.

**Question:** Can a gradient-boosted goal scorer with per-observation contributions match or beat a UVFA-style MLP value regressor at recognising the intended goal from a partial grid-world trajectory?

**Dataset:** Simulated 2D grid-world trajectories generated in code with numpy (no download). Each episode has its own random obstacle map, four candidate goals and one noisy shortest-path trajectory, of which the first 50% is observed. This is not the paper's own dataset.

**Baseline vs. improved:** The baseline is a scikit-learn MLP regressor that predicts remaining steps to each goal from observed position, goal, recent motion and step index. The improved model is a LightGBM regressor on the same inputs plus engineered progress and heading-toward-goal features. Both are tuned with 3-fold grouped cross-validation on the training episodes only, and the goal with the lowest predicted cost is chosen.

## How to run

```
pip install -r requirements.txt
python analysis.py
```

Results: written to `results.json` (accuracy chart saved to `figures/`).

## Reference

This project is motivated by the following paper:

> F Meneguzzi, B Nageris, L Amado, R Mirsky (2026). *Online Goal Recognition: When Inference-Time Computation is Not The (Full) Answer*. assets-eu.researchsquare.com. https://assets-eu.researchsquare.com/files/rs-8982676/v1_covered_fd9ab69a-1d28-4a35-bff5-1feb9b98aef9.pdf?c=1779434899

Paper author: Prof. Felipe Meneguzzi, University of Aberdeen (felipe.meneguzzi@abdn.ac.uk)


---
_The code in this repository was drafted with AI assistance and runs automatically on GitHub Actions; results come from the latest run._
