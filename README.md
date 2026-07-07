# Machine Learning From Scratch

Building fundamental machine learning algorithms from scratch in Python — using only NumPy for the core math, no ML frameworks — to understand how they actually work under the hood.

Each algorithm lives in its own notebook under `src/<algorithm>/model.ipynb`, with the implementation written step by step and visualized.

## Implemented

| Algorithm | Type | Location | Core ideas implemented |
|---|---|---|---|
| Neural Network | Supervised | `src/neural_net/` | Forward & backward propagation, activations, gradient descent |
| K-Means | Unsupervised | `src/kmeans/` | Centroid initialization, assign/update loop, convergence threshold |
| Decision Tree (CART) | Supervised | `src/decision_tree/` | Gini impurity, greedy split search, recursive tree growth, prediction by traversal, overfitting analysis |

## What I learned & implemented

- **Neural network** — how forward and backward propagation fit together, and how gradient descent nudges weights down a loss surface.
- **K-Means** — clustering as an iterative refinement loop: assign points to the nearest centroid, move centroids to the mean, repeat until movement falls below a threshold.
- **Decision tree** — that the whole model is greedy minimization of *impurity*: pick the single best axis-aligned split, then recurse. Prediction is just walking a sample down the tree to a leaf. Depth is the key knob — on noisy XOR data, test accuracy peaked at `max_depth ≈ 4`, then deeper trees overfit (perfect training score, worse generalization).

## What it gave me

- Intuition for the math behind each method — impurity, gradients, distances — instead of treating them as black boxes.
- Fluency translating equations into vectorized NumPy.
- A practical feel for the trade-offs that matter in real ML: convergence, the bias/variance trade-off, and *why* you never trust training accuracy alone.

## Moving forward

- **Ensembles on top of the decision tree** — Random Forest (bagging) and Gradient Boosting, both of which reuse the tree as their base learner.
- **Fill the classic gaps** — logistic regression, PCA, and k-nearest neighbors.
- **Branch into classic algorithms** — graph search (Dijkstra / A\*) as a change of pace from the ML track.

## Running

Notebooks use `numpy`, `pandas`, and `matplotlib` (with `scikit-learn` for a few preprocessing helpers). Open any `src/<algorithm>/model.ipynb` and run the cells top to bottom.
