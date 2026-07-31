from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import RobustScaler


@dataclass(frozen=True)
class ClusterResult:
    labels: np.ndarray
    embedding: np.ndarray
    stability: np.ndarray
    largest_cluster: int | None
    representatives: dict[int, list[int]]
    cluster_sizes: dict[int, int]


def _representatives(
    embedding: np.ndarray, labels: np.ndarray, *, per_cluster: int = 8
) -> dict[int, list[int]]:
    result: dict[int, list[int]] = {}
    for label in sorted(int(value) for value in np.unique(labels) if value >= 0):
        indices = np.flatnonzero(labels == label)
        center = np.median(embedding[indices], axis=0)
        distances = np.linalg.norm(embedding[indices] - center, axis=1)
        order = indices[np.argsort(distances)]
        medoids = order[: max(1, per_cluster // 2)].tolist()
        boundary = order[-max(1, per_cluster - len(medoids)) :].tolist()
        result[label] = list(dict.fromkeys(medoids + boundary))
    return result


def cluster_descriptors(
    descriptors: np.ndarray,
    *,
    min_cluster_size: int,
    min_samples: int,
    seed: int,
    pca_variance: float = 0.95,
    pca_max_components: int = 32,
    stability_runs: int = 0,
    stability_fraction: float = 0.9,
    cluster_selection_epsilon_quantile: float = 0.75,
) -> ClusterResult:
    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or len(values) < min_cluster_size:
        raise ValueError("not enough finite descriptors for clustering")
    scaled = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(values)
    maximum = min(pca_max_components, scaled.shape[0] - 1, scaled.shape[1])
    pca = PCA(n_components=pca_variance, svd_solver="full", random_state=seed)
    embedding = pca.fit_transform(scaled)
    if embedding.shape[1] > maximum:
        embedding = embedding[:, :maximum]
    def labels_for(sample: np.ndarray) -> np.ndarray:
        # With allow_single_cluster=True and epsilon=0, sklearn's HDBSCAN can
        # return exactly min_cluster_size points from a smooth, genuinely
        # unimodal trajectory manifold and label every other point as noise.
        # Calibrate epsilon from the robust upper quartile of the core-distance
        # distribution. Distinct, well-separated procedures remain separate,
        # while camera pose / operator amplitude variation inside one procedure
        # is not mistaken for 90% noise.
        neighbors = min(len(sample), min_samples + 1)
        distances = NearestNeighbors(n_neighbors=neighbors).fit(sample).kneighbors(
            return_distance=True
        )[0]
        epsilon = float(
            np.quantile(
                distances[:, -1],
                cluster_selection_epsilon_quantile,
            )
        )
        return HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method="eom",
            cluster_selection_epsilon=epsilon,
            allow_single_cluster=True,
        ).fit_predict(sample)

    base = labels_for(embedding)
    sizes = {
        int(label): int(np.sum(base == label))
        for label in np.unique(base)
        if int(label) >= 0
    }
    largest = max(sizes, key=lambda label: (sizes[label], -label)) if sizes else None
    stability = np.ones(len(values), dtype=np.float64)
    if stability_runs and largest is not None:
        rng = np.random.default_rng(seed)
        selected_counts = np.zeros(len(values), dtype=np.int64)
        membership_counts = np.zeros(len(values), dtype=np.int64)
        base_members = base == largest
        sample_size = max(min_cluster_size, int(np.ceil(len(values) * stability_fraction)))
        for _ in range(stability_runs):
            selected = np.sort(rng.choice(len(values), size=sample_size, replace=False))
            labels = labels_for(embedding[selected])
            selected_counts[selected] += 1
            best_label = None
            best_jaccard = -1.0
            for label in np.unique(labels):
                if label < 0:
                    continue
                members = np.zeros(len(values), dtype=bool)
                members[selected[labels == label]] = True
                union = np.sum(members | base_members)
                jaccard = float(np.sum(members & base_members) / union) if union else 0.0
                if jaccard > best_jaccard:
                    best_jaccard = jaccard
                    best_label = int(label)
            if best_label is not None:
                membership_counts[selected[labels == best_label]] += 1
        stability = np.divide(
            membership_counts,
            selected_counts,
            out=np.zeros(len(values), dtype=np.float64),
            where=selected_counts > 0,
        )
    return ClusterResult(
        labels=base,
        embedding=embedding,
        stability=stability,
        largest_cluster=largest,
        representatives=_representatives(embedding, base),
        cluster_sizes=sizes,
    )


def cluster_orientation_descriptors(
    descriptors: np.ndarray,
    *,
    seed: int,
    maximum_clusters: int = 6,
) -> ClusterResult:
    """BIC-selected mixture for a compact, potentially single orientation mode.

    HDBSCAN is deliberately not used here: a continuous camera-pose variation
    around one physical table orientation must not become mostly ``noise``.
    """

    values = np.asarray(descriptors, dtype=np.float64)
    if values.ndim != 2 or len(values) < 2:
        raise ValueError("at least two orientation descriptors are required")
    scaled = RobustScaler(quantile_range=(10.0, 90.0)).fit_transform(values)
    components = min(12, scaled.shape[0] - 1, scaled.shape[1])
    embedding = PCA(
        n_components=components, svd_solver="full", random_state=seed
    ).fit_transform(scaled)
    candidates: list[tuple[float, GaussianMixture]] = []
    for count in range(1, min(maximum_clusters, len(values) // 10) + 1):
        model = GaussianMixture(
            n_components=count,
            covariance_type="diag",
            n_init=10,
            random_state=seed,
            reg_covar=1e-5,
        ).fit(embedding)
        candidates.append((float(model.bic(embedding)), model))
    _, selected = min(candidates, key=lambda item: item[0])
    raw = selected.predict(embedding)
    # Relabel by descending population, making cluster 0 the deterministic
    # dominant proposal shown to the reviewer.
    old_labels = sorted(
        np.unique(raw), key=lambda label: (-int(np.sum(raw == label)), int(label))
    )
    relabel = {int(old): new for new, old in enumerate(old_labels)}
    labels = np.asarray([relabel[int(value)] for value in raw], dtype=np.int64)
    sizes = {
        int(label): int(np.sum(labels == label)) for label in np.unique(labels)
    }
    return ClusterResult(
        labels=labels,
        embedding=embedding,
        stability=np.ones(len(values), dtype=np.float64),
        largest_cluster=0,
        representatives=_representatives(embedding, labels),
        cluster_sizes=sizes,
    )
