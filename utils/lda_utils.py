# lda_utils.py
"""
LDA / Wilks' Lambda / PCA-dimensionality helpers for stages/analyze.py.
No equivalent exists in domain_inference — this is new.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


def pca_dims_at_95(X):
    """Number of PCA components needed to explain 95% of variance in X."""
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=0.95)
    pca.fit(Xs)
    return pca.n_components_


def pca_reduce(X, n_components):
    """Standardize + PCA-reduce X to n_components dimensions."""
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components)
    return pca.fit_transform(Xs)


def pca_reduce_at_95(X):
    """
    Standardize + PCA-reduce X to the number of components explaining 95%
    of variance, in a single PCA fit. Returns (X_pca, n_components).
    """
    Xs = StandardScaler().fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(Xs)
    return X_pca, pca.n_components_


def pca_fit(X):
    """
    Fit StandardScaler + PCA(0.95) on X. Returns (scaler, pca, X_pca) so
    the same fitted transform can later be applied to other data via
    pca_apply — e.g. fit on literature embeddings, apply to a larger
    expanded dataset so both land in the same PCA space.
    """
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    pca = PCA(n_components=0.95)
    X_pca = pca.fit_transform(Xs)
    return scaler, pca, X_pca


def pca_apply(scaler, pca, X):
    """Apply an already-fitted scaler+PCA (from pca_fit) to new data."""
    return pca.transform(scaler.transform(X))


def wilks_lambda(X, y):
    """
    Wilks' Lambda for one-way MANOVA: Lambda = det(Sw) / det(Sw + Sb),
    where Sw is the within-class scatter matrix and Sb the between-class
    scatter matrix. Ranges from 0 (classes perfectly separated) to 1 (no
    separation). Computed via log-determinants (np.linalg.slogdet) to
    avoid over/underflow on the p x p scatter matrices.

    X should already be PCA-reduced to well below n_samples (raw 1280-dim
    ESM2 features vs. a few hundred sequences would make Sw singular).
    """
    X = np.asarray(X)
    y = np.asarray(y)
    classes = np.unique(y)
    n_features = X.shape[1]

    overall_mean = X.mean(axis=0)
    Sw = np.zeros((n_features, n_features))
    Sb = np.zeros((n_features, n_features))

    for c in classes:
        Xc = X[y == c]
        nc = Xc.shape[0]
        mean_c = Xc.mean(axis=0)
        diff = Xc - mean_c
        Sw += diff.T @ diff
        mean_diff = (mean_c - overall_mean).reshape(-1, 1)
        Sb += nc * (mean_diff @ mean_diff.T)

    St = Sw + Sb
    sign_w, logdet_w = np.linalg.slogdet(Sw)
    sign_t, logdet_t = np.linalg.slogdet(St)

    if sign_w <= 0 or sign_t <= 0:
        return float("nan")

    return float(np.exp(logdet_w - logdet_t))


def bartlett_chi_square(wilks_lambda_val, n_samples, n_features, n_classes):
    """
    Bartlett's chi-square approximation to the sampling distribution of
    Wilks' Lambda (Bartlett 1938): chi2 = -(N - 1 - (p+g)/2) * ln(Lambda),
    approximately chi2-distributed with p*(g-1) degrees of freedom under
    the null hypothesis of no group separation. p = n_features (the PCA
    dimensionality Lambda was computed in), g = n_classes, N = n_samples.
    """
    if wilks_lambda_val is None or np.isnan(wilks_lambda_val) or wilks_lambda_val <= 0:
        return float("nan")
    return -(n_samples - 1 - (n_features + n_classes) / 2) * np.log(wilks_lambda_val)


def partial_eta_squared(wilks_lambda_val, n_features, n_classes):
    """
    Partial eta-squared derived from Wilks' Lambda: eta^2_partial =
    1 - Lambda^(1/s), where s = min(n_features, n_classes - 1).
    """
    if wilks_lambda_val is None or np.isnan(wilks_lambda_val):
        return float("nan")
    s = min(n_features, n_classes - 1)
    if s <= 0:
        return float("nan")
    return 1 - wilks_lambda_val ** (1.0 / s)


def fit_lda_projection(X, y, n_components):
    """
    Fit LinearDiscriminantAnalysis and return (X_lda, lda_model).
    n_components is clamped to min(n_classes - 1, n_features).
    """
    n_classes = len(np.unique(y))
    max_components = min(n_classes - 1, X.shape[1])
    nc = max(1, min(n_components, max_components))

    lda = LinearDiscriminantAnalysis(n_components=nc, solver="eigen")
    X_lda = lda.fit_transform(X, y)
    return X_lda, lda


def plot_lda_3d(pdf, X, color_labels, color_palette, title):
    """Single-population 3D LDA scatter, colored by color_labels."""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")

    z = X[:, 2] if X.shape[1] > 2 else np.zeros(X.shape[0])
    present = set(pd.unique(color_labels))
    ordered_labels = [l for l in color_palette if l in present]
    ordered_labels += [l for l in present if l not in color_palette]

    for label in ordered_labels:
        mask = color_labels == label
        color = color_palette.get(label, color_palette.get("unknown", "gray"))
        ax.scatter(X[mask, 0], X[mask, 1], z[mask],
                   label=str(label), color=color, s=30, alpha=0.8)

    ax.set_xlabel("LD1")
    ax.set_ylabel("LD2")
    ax.set_zlabel("LD3")
    ax.set_title(title)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    pdf.savefig(fig)
    plt.close(fig)
