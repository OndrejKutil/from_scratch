# Machine Learning From Scratch

Building fundamental machine learning algorithms from scratch in Python — using only NumPy for the core math, no ML frameworks — to understand how they actually work under the hood.

Each algorithm lives in its own notebook under `src/<algorithm>/model.ipynb`, with the implementation written step by step and visualized.

## Implemented

| Algorithm | Type | Location | Core ideas implemented |
|---|---|---|---|
| Neural Network | Supervised | `src/neural_net/` | Forward & backward propagation, activations, gradient descent |
| K-Means | Unsupervised | `src/kmeans/` | Centroid initialization, assign/update loop, convergence threshold |
| Decision Tree (CART) | Supervised | `src/decision_tree/` | Gini impurity, greedy split search, recursive tree growth, prediction by traversal, overfitting analysis |
| Random Forest (Bagging) | Supervised | `src/random_forest/` | Bootstrap resampling, per-split feature subsampling, majority-vote aggregation, variance reduction vs. a single tree |

## What I learned & implemented

- **Neural network** — how forward and backward propagation fit together, and how gradient descent nudges weights down a loss surface.
- **K-Means** — clustering as an iterative refinement loop: assign points to the nearest centroid, move centroids to the mean, repeat until movement falls below a threshold.
- **Decision tree** — that the whole model is greedy minimization of *impurity*: pick the single best axis-aligned split, then recurse. Prediction is just walking a sample down the tree to a leaf. Depth is the key knob — on noisy XOR data, test accuracy peaked at `max_depth ≈ 4`, then deeper trees overfit (perfect training score, worse generalization).
- **Random forest** — that an ensemble adds no new math. It's the same CART tree (refactored into a reusable `tree.py` estimator) plus two independent sources of randomness — a bootstrap resample of the **rows** per tree, and a fresh random subset of the **columns** at every split — followed by a majority vote. What makes it work is *decorrelation*: averaging only helps when the trees are wrong in **different** ways, which is why *where* the randomness is drawn matters more than how much of it there is. Freezing the feature subset per tree instead of per split cripples each tree (0.792 → 0.902 on the same data) because it goes permanently blind to the other features.
- **Ensembles are built from deliberately handicapped learners.** Depth-matched on noisy two-moons, the forest's individual trees averaged **0.912** test accuracy — *worse* than the lone tree's **0.933**, since bootstrapping starves each of ~37% of the rows and feature subsampling blinds each split. Yet the vote scored **0.944**, beating not just the average member but the single **best** tree in the forest (0.937). Averaging recovers far more than the handicap costs.

## What it gave me

- Intuition for the math behind each method — impurity, gradients, distances — instead of treating them as black boxes.
- Fluency translating equations into vectorized NumPy — and a clearer sense of what "vectorized" actually means (`np.apply_along_axis` lives in NumPy but still loops in Python; it benchmarked *slower* than a hand-written loop, while pushing the loop over samples into C ran ~56× faster).
- A practical feel for the trade-offs that matter in real ML: convergence, the bias/variance trade-off, and *why* you never trust training accuracy alone — a lesson the forest sharpened, since bagging pushed *training* accuracy up to 0.998 while genuinely generalizing better.

## Moving forward

- **Gradient Boosting** — the other ensemble on top of the tree, and a bigger build than the forest was: it needs regression trees (variance splits, mean predictions) rather than the classifier, plus residuals, an additive model, and a learning rate.
- **Out-of-bag scoring** for the forest — each tree ignores ~37% of the rows, so scoring every sample with only the trees that never saw it gives an honest validation estimate with no test set at all.
- **Fill the classic gaps** — logistic regression, PCA, and k-nearest neighbors.
- **Branch into classic algorithms** — graph search (Dijkstra / A\*) as a change of pace from the ML track.

## Running

Notebooks use `numpy`, `pandas`, and `matplotlib` (with `scikit-learn` for a few preprocessing helpers). Open any `src/<algorithm>/model.ipynb` and run the cells top to bottom.
