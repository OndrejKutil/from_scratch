import numpy as np
import pandas as pd


def load_data(n_samples: int = 400, noise: float = 0.30, seed: int = 42) -> pd.DataFrame:
    """
    Generate the classic "two moons" dataset: two interleaving half-circles.

    Each class is a crescent; the two crescents interlock, so the boundary
    between them is a smooth *curve*, not a straight line. Add enough Gaussian
    noise and the crescents smear into each other — a single deep decision tree
    will chase that noise and carve jagged little islands (overfitting), while a
    random forest averages many trees into a smooth, stable boundary.

    That contrast is the whole point of this notebook, which is why we use noisy
    moons rather than the easier XOR blobs from the decision-tree notebook.

    Args:
        n_samples: Total number of points (split evenly across the two moons).
        noise: Standard deviation of the Gaussian noise added to every point.
        seed: Random seed for reproducibility.

    Returns:
        DataFrame with columns: feature_1, feature_2, label (0 or 1).
    """
    rng = np.random.default_rng(seed)
    n_per_moon = n_samples // 2

    # Angles sweep a half-circle for each moon.
    theta = np.linspace(0, np.pi, n_per_moon)

    # Moon 0: upper crescent, centered near the origin.
    x0 = np.cos(theta)
    y0 = np.sin(theta)

    # Moon 1: lower crescent, shifted right and down so the two interlock.
    x1 = 1 - np.cos(theta)
    y1 = 0.5 - np.sin(theta)

    features = np.vstack([
        np.column_stack([x0, y0]),
        np.column_stack([x1, y1]),
    ])
    labels = np.array([0] * n_per_moon + [1] * n_per_moon)

    # Gaussian noise is what forces the overfitting-vs-smoothing story.
    features = features + rng.normal(0, noise, features.shape)

    df = pd.DataFrame(features, columns=["feature_1", "feature_2"])
    df["label"] = labels

    # Shuffle so rows aren't grouped by class.
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return df
