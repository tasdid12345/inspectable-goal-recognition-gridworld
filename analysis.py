"""Goal recognition on simulated grid worlds.
Baseline: MLP value regressor (UVFA-style) predicting remaining steps to each goal.
Improved: LightGBM regressor with engineered progress/heading features plus
cross-candidate (relative) features, so each goal is scored against the other goals
on the same map at the same time step. Both pick the goal with the lowest predicted
cost-to-go at the end of the observed prefix."""
import collections, json, os, sys, time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMRegressor
from sklearn.model_selection import GridSearchCV, GroupKFold, train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED, SEEDS = 42, [42, 43, 44]
N_EP, S, K = 1500, 12, 4                 # episodes, grid side length, candidate goals per map
OBST, NOISE, MAX_STEPS = 0.25, 0.10, 60  # obstacle density, random-action probability, step cap
OBS_FRAC, TAU = 0.5, 2.0                 # observed fraction of trajectory; softmax temperature for log-loss
MOVES = [(1, 0), (-1, 0), (0, 1), (0, -1)]
# column indices inside row_feats() output (improved variant)
DE_COL, COS_COL, RATIO_COL = 9, 14, 13
rng = np.random.default_rng(SEED)
t0 = time.time()


def log(msg):
    print(f"[{time.time() - t0:6.1f}s] {msg}", flush=True)


def bfs(free, src):
    """Shortest-path (steps) distance from src to every cell; inf if unreachable."""
    dist = np.full((S, S), np.inf)
    dist[src] = 0
    q = collections.deque([src])
    while q:
        r, c = q.popleft()
        for dr, dc in MOVES:
            nr, nc = r + dr, c + dc
            if 0 <= nr < S and 0 <= nc < S and free[nr, nc] and dist[nr, nc] == np.inf:
                dist[nr, nc] = dist[r, c] + 1
                q.append((nr, nc))
    return dist


def make_episode():
    """One random map with K goals and one noisy shortest-path trajectory to the true goal."""
    while True:  # resample maps with too few reachable cells
        free = rng.random((S, S)) >= OBST
        cells = np.argwhere(free)
        start = tuple(cells[rng.integers(len(cells))])
        d0 = bfs(free, start)
        cand = np.argwhere(np.isfinite(d0) & (d0 >= 6))
        if len(cand) >= K + 20:
            break
    goals = [tuple(x) for x in cand[rng.choice(len(cand), K, replace=False)]]
    D = [bfs(free, g) for g in goals]          # remaining-steps lookup for each goal
    g = int(rng.integers(K))                   # true (intended) goal
    pos, path = start, [start]
    for _ in range(MAX_STEPS):
        if pos == goals[g]:
            break
        nbrs = [(pos[0] + dr, pos[1] + dc) for dr, dc in MOVES]
        nbrs = [p for p in nbrs if 0 <= p[0] < S and 0 <= p[1] < S and free[p]]
        if rng.random() < NOISE:               # random action noise
            pos = nbrs[rng.integers(len(nbrs))]
        else:                                  # greedy shortest-path step, random tie-break
            best = min(D[g][p] for p in nbrs)
            opts = [p for p in nbrs if D[g][p] == best]
            pos = opts[rng.integers(len(opts))]
        path.append(pos)
    return dict(start=start, goals=goals, true=g, D=D, path=path)


def row_feats(pos, start, goal, vel, t, improved):
    """Baseline: position, goal, recent motion, step index. Improved adds engineered progress/heading."""
    p, gl, s = np.array(pos, float), np.array(goal, float), np.array(start, float)
    base = [*p, *gl, *vel, t]
    if not improved:
        return base
    dg = gl - p
    de, dsg = np.linalg.norm(dg), np.linalg.norm(gl - s)
    cos = vel @ dg / (np.linalg.norm(vel) * de + 1e-9)   # heading toward goal
    return base + [*s, de, np.abs(dg).sum(), dsg, dsg - de, de / (dsg + 1e-9), cos, *dg]


def block_rows(ep, pos, vel, t, improved):
    """K rows for one (episode, prefix end). Improved adds relative features: each goal's
    distance / heading / progress compared with the other candidate goals on the same map."""
    F = np.array([row_feats(pos, ep["start"], ep["goals"][k], vel, t, improved)
                  for k in range(K)], float)
    if not improved:
        return F
    de, cos, ratio = F[:, DE_COL], F[:, COS_COL], F[:, RATIO_COL]
    rel = np.column_stack([
        de - de.min(),                 # distance excess over the closest candidate
        de - de.mean(),                # distance relative to the average candidate
        np.argsort(np.argsort(de)),    # rank of distance among candidates (0 = closest)
        cos - cos.max(),               # heading excess over the best-aligned candidate
        cos - cos.mean(),
        ratio - ratio.max(),           # progress excess over the best candidate
    ])
    return np.hstack([F, rel])


def build(eps, improved):
    """Rows = (prefix end t, candidate goal k) per episode; target = true remaining steps to goal k."""
    X, y, grp, fin = [], [], [], []
    for i, ep in enumerate(eps):
        path = ep["path"]
        n_obs = max(3, int(OBS_FRAC * len(path)))   # only the first 50% is observed
        for t in range(1, n_obs):
            pos = path[t]
            vel = np.subtract(pos, path[max(0, t - 3)]) / min(3, t)
            X.extend(block_rows(ep, pos, vel, t, improved))
            y.extend([ep["D"][k][pos] for k in range(K)])
            grp.extend([i] * K)
            fin.extend([t == n_obs - 1] * K)
    return np.array(X), np.array(y), np.array(grp), np.array(fin)


def make_model(improved, seed):
    if improved:
        return LGBMRegressor(n_estimators=400, subsample=0.8, subsample_freq=1,
                             colsample_bytree=0.8, reg_lambda=1.0,
                             random_state=seed, verbose=-1)
    return make_pipeline(StandardScaler(), MLPRegressor(max_iter=500, early_stopping=True, random_state=seed))


def evaluate(cost, true):
    """cost: (n_ep, K) predicted remaining steps. Returns accuracy, top-2 accuracy, mean log-loss."""
    n = len(true)
    acc = np.mean(cost.argmin(1) == true)
    top2 = np.mean((np.argsort(cost, 1)[:, :2] == true[:, None]).any(1))
    logits = -cost / TAU
    mx = logits.max(1, keepdims=True)
    logp = logits - (mx + np.log(np.exp(logits - mx).sum(1, keepdims=True)))
    return acc, top2, -logp[np.arange(n), true].mean()


def main():
    eps = [make_episode() for _ in range(N_EP)]
    log(f"generated {N_EP} grid-world episodes ({K} goals each, one unseen map per episode)")
    tr, te = train_test_split(eps, test_size=0.3, random_state=SEED,
                              stratify=[e["true"] for e in eps])
    true_te = np.array([e["true"] for e in te])
    res = {}
    for name, improved in [("baseline", False), ("improved", True)]:
        Xtr, ytr, gtr, _ = build(tr, improved)
        Xte, _, _, fte = build(te, improved)
        Xte = Xte[fte]                          # final observed prefix only: K rows per test episode
        log(f"{name}: {len(Xtr)} training rows; 3-fold GroupKFold tuning on training set only")
        grid = ({"num_leaves": [15, 31], "learning_rate": [0.05, 0.1]} if improved else
                {"mlpregressor__alpha": [1e-4, 1e-2],
                 "mlpregressor__hidden_layer_sizes": [(64, 64), (128, 128)]})
        gs = GridSearchCV(make_model(improved, SEED), grid, cv=GroupKFold(3),
                          scoring="neg_mean_absolute_error")
        gs.fit(Xtr, ytr, groups=gtr)
        log(f"{name}: best CV params {gs.best_params_}")
        runs = []
        for s in SEEDS:                         # refit with best params; test set used once
            m = make_model(improved, s).set_params(**gs.best_params_).fit(Xtr, ytr)
            runs.append(evaluate(m.predict(Xte).reshape(-1, K), true_te))
        res[name] = np.array(runs)              # columns: acc, top2, logloss
        log(f"{name}: test accuracy per seed {res[name][:, 0].round(3).tolist()}")

    b, i = res["baseline"], res["improved"]
    met = {"baseline_score": b[:, 0].mean(), "baseline_score_std": b[:, 0].std(),
           "improved_score": i[:, 0].mean(), "improved_score_std": i[:, 0].std(),
           "baseline_top2_accuracy": b[:, 1].mean(), "improved_top2_accuracy": i[:, 1].mean(),
           "baseline_logloss": b[:, 2].mean(), "improved_logloss": i[:, 2].mean()}
    met["improvement"] = met["improved_score"] - met["baseline_score"]   # higher accuracy is better
    met = {k: float(v) for k, v in met.items()}
    if not all(np.isfinite(list(met.values()))):
        log("non-finite metrics; exiting")
        sys.exit(1)

    os.makedirs("figures", exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 3.5))
    ax.bar(["baseline\nMLP", "improved\nLightGBM"], [met["baseline_score"], met["improved_score"]],
           yerr=[met["baseline_score_std"], met["improved_score_std"]], capsize=5,
           color=["#999999", "#33aa77"])
    ax.set_ylabel("goal-recognition accuracy (test)")
    ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig("figures/accuracy_comparison.png", dpi=120)

    out = {
        "question": "Can a gradient-boosted goal scorer with per-observation contributions match or beat a UVFA-style MLP value regressor at recognising the intended goal from a partial grid-world trajectory?",
        "method": "Seeded 12x12 grid worlds with random obstacles and 4 candidate goals; noisy shortest-path trajectories observed for the first 50%; split by episode (70/30) with unseen maps in test; goal = argmin predicted remaining steps.",
        "dataset": {
            "name": "Simulated 2D grid-world goal-recognition trajectories (generated in code)",
            "source_url": "https://numpy.org/doc/stable/reference/random/index.html",
            "n_rows": N_EP, "n_train": len(tr), "n_test": len(te),
            "same_as_paper": False, "baseline_is_papers_method": False,
        },
        "metric_name": "goal-recognition accuracy (test episodes)",
        "higher_is_better": True,
        "baseline": "scikit-learn MLPRegressor (standardised inputs) as a UVFA-style value regressor predicting remaining steps from position, goal, recent motion and step index; tuned by 3-fold GroupKFold on training data.",
        "improved": "LightGBM regressor on the same inputs plus engineered start position, Euclidean/Manhattan distance to goal, start-to-goal progress, distance ratio and heading-toward-goal cosine, plus cross-candidate relative features (distance excess/rank, heading and progress relative to the other goals on the same map); same 4-combo CV tuning budget.",
        "metrics": met,
        "notes": ("Simulated data only; the paper's GRUF model and its numbers are not reproduced, so results are not comparable to the paper. "
                  "n_rows counts episodes, not observation rows. Log-loss is the multiclass softmax (temperature 2, fixed, not tuned) negative log-probability of the true goal, averaged over test episodes. "
                  "Tuning used seed 42 only; seeds 42-44 vary the final fit. Test set used once. Baseline and split are unchanged; only the improved model's features and LightGBM settings changed. Dataset is small, so the sample size is modest; no subsampling was needed."),
    }
    with open("results.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    log(f"baseline acc {met['baseline_score']:.3f} +/- {met['baseline_score_std']:.3f}; "
        f"improved acc {met['improved_score']:.3f} +/- {met['improved_score_std']:.3f}")
    log("wrote results.json and figures/accuracy_comparison.png")


if __name__ == "__main__":
    main()
