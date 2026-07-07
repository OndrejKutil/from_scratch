import numpy as np
import pandas as pd


def load_data(n_samples: int = 200, noise: float = 0.08, seed: int = 42) -> pd.DataFrame:
    """
    Generate a 2D, 2-class synthetic dataset with an XOR-like structure.

    Each class occupies two *opposite* corners of the unit square, so the two
    classes are NOT linearly separable — no single straight line can divide
    them. A decision tree, however, can carve the space with several
    axis-aligned splits, which makes this a nice showcase for why tree depth
    matters.

    Args:
        n_samples: Approximate total number of points (split evenly over 4 blobs).
        noise: Standard deviation of the Gaussian spread around each blob center.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: feature_1, feature_2, label (0 or 1).
    """
    rng = np.random.default_rng(seed)

    # Four blob centers arranged as a 2x2 checkerboard.
    # Class 0 -> bottom-left & top-right; class 1 -> top-left & bottom-right.
    centers = {
        0: [(0.25, 0.25), (0.75, 0.75)],
        1: [(0.25, 0.75), (0.75, 0.25)],
    }

    features: list[tuple[float, float]] = []
    labels: list[int] = []
    per_blob = n_samples // 4

    for label, blob_centers in centers.items():
        for cx, cy in blob_centers:
            xs = rng.normal(cx, noise, per_blob)
            ys = rng.normal(cy, noise, per_blob)
            features.extend(zip(xs, ys))
            labels.extend([label] * per_blob)

    df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
    df["label"] = labels

    # Shuffle so rows aren't grouped by class.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
