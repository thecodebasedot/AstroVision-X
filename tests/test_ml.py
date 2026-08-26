"""Machine-learning components: scaling, anomaly detection, clustering, models."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision.core.exceptions import NotFittedError
from astrovision.ml.autoencoder import LinearAutoencoder
from astrovision.ml.clustering import DBSCAN, HDBSCANLite, KMeans, cluster, silhouette_score
from astrovision.ml.features import (
    FEATURE_NAMES,
    catalog_embeddings,
    catalog_features,
    combine_features,
    cutout_descriptor,
)
from astrovision.ml.gbdt import GradientBoostedClassifier
from astrovision.ml.isolation_forest import IsolationForest
from astrovision.ml.metric import SimilaritySearch
from astrovision.ml.scaler import PCA, RobustScaler


@pytest.fixture()
def blobs():
    rng = np.random.default_rng(0)
    return np.vstack([rng.normal(centre, 0.4, (60, 2))
                      for centre in ([0, 0], [6, 0], [3, 6])])


class TestScaler:
    def test_centres_and_scales(self):
        rng = np.random.default_rng(0)
        data = rng.normal(50.0, 10.0, (300, 3))
        scaled = RobustScaler().fit_transform(data)
        assert np.allclose(np.median(scaled, axis=0), 0.0, atol=0.1)
        assert np.allclose(np.std(scaled, axis=0), 1.0, atol=0.2)

    def test_imputes_missing_values(self):
        data = np.random.default_rng(1).normal(size=(100, 3))
        data[5, 1] = np.nan
        scaler = RobustScaler().fit(data)
        assert np.isfinite(scaler.transform(data)).all()
        assert scaler.missing_fraction_[1] > 0

    def test_resists_outliers(self):
        data = np.random.default_rng(2).normal(0.0, 1.0, (500, 1))
        data[:5] = 1e6
        scaled = RobustScaler().fit_transform(data)
        assert abs(float(np.median(scaled))) < 0.2

    def test_requires_fit(self):
        with pytest.raises(NotFittedError):
            RobustScaler().transform(np.zeros((3, 2)))


class TestPCA:
    def test_reduces_dimension_and_inverts(self):
        rng = np.random.default_rng(3)
        latent = rng.normal(size=(200, 2))
        data = latent @ rng.normal(size=(2, 6))
        pca = PCA(2)
        reduced = pca.fit_transform(data)
        assert reduced.shape == (200, 2)
        assert np.allclose(pca.inverse_transform(reduced), data, atol=1e-6)

    def test_explained_variance_sums_toward_one(self):
        data = np.random.default_rng(4).normal(size=(100, 5))
        pca = PCA(5).fit(data)
        assert float(pca.explained_variance_ratio_.sum()) == pytest.approx(1.0, abs=0.02)


class TestIsolationForest:
    def test_ranks_injected_outliers_first(self):
        rng = np.random.default_rng(5)
        normal = rng.normal(0.0, 1.0, (400, 4))
        outliers = np.array([[8.0, 8.0, 8.0, 8.0], [-9.0, 7.0, -8.0, 6.0]])
        data = np.vstack([normal, outliers])
        scores = IsolationForest(n_estimators=100, random_state=1).fit_score(data)
        assert set(np.argsort(scores)[-2:]) == {400, 401}

    def test_scores_are_bounded(self):
        data = np.random.default_rng(6).normal(size=(100, 3))
        scores = IsolationForest(n_estimators=32, random_state=0).fit_score(data)
        assert np.all((scores >= 0.0) & (scores <= 1.0))

    def test_contamination_sets_the_flag_rate(self):
        data = np.random.default_rng(7).normal(size=(500, 3))
        forest = IsolationForest(n_estimators=64, contamination=0.05,
                                 random_state=0).fit(data)
        assert 0.02 <= float(forest.predict(data).mean()) <= 0.10

    def test_requires_fit(self):
        with pytest.raises(NotFittedError):
            IsolationForest().score(np.zeros((3, 2)))


class TestAutoencoder:
    def test_flags_points_off_the_manifold(self):
        rng = np.random.default_rng(8)
        latent = rng.normal(size=(400, 2))
        projection = rng.normal(size=(2, 6))
        on_manifold = latent @ projection + rng.normal(0.0, 0.05, (400, 6))
        off_manifold = rng.normal(0.0, 3.0, (4, 6))
        model = LinearAutoencoder(latent_dim=2).fit(on_manifold)
        normal_score = float(np.median(model.score(on_manifold)))
        odd_score = float(np.min(model.score(off_manifold)))
        assert odd_score > 5 * normal_score

    def test_encode_gives_the_latent_dimension(self):
        data = np.random.default_rng(9).normal(size=(50, 8))
        assert LinearAutoencoder(latent_dim=3).fit(data).encode(data).shape == (50, 3)


class TestClustering:
    def test_kmeans_finds_the_blobs(self, blobs):
        labels = KMeans(n_clusters=3, random_state=0).fit_predict(blobs)
        assert len(set(labels)) == 3
        assert silhouette_score(blobs, labels) > 0.6

    def test_dbscan_labels_noise(self, blobs):
        noisy = np.vstack([blobs, np.random.default_rng(10).uniform(-6, 12, (10, 2))])
        labels = DBSCAN(eps=1.0, min_samples=5).fit_predict(noisy)
        assert (labels == -1).sum() > 0
        assert len(set(labels) - {-1}) == 3

    def test_hdbscan_needs_no_eps(self, blobs):
        labels = HDBSCANLite(min_cluster_size=8).fit_predict(blobs)
        assert len(set(labels) - {-1}) >= 2

    def test_dispatch_by_name(self, blobs):
        for method in ("kmeans", "dbscan", "hdbscan"):
            result = cluster(blobs, method, n_clusters=3, eps=1.0, min_cluster_size=8)
            assert len(result["labels"]) == len(blobs)

    def test_unknown_method_raises(self, blobs):
        with pytest.raises(ValueError):
            cluster(blobs, "not-a-method")


class TestSimilarity:
    def test_finds_itself_first(self):
        data = np.eye(5)
        ids, _ = SimilaritySearch().fit(data).query(data[2], k=1)
        assert ids[0] == 2

    def test_knn_distance_flags_the_isolated_point(self):
        rng = np.random.default_rng(11)
        data = np.vstack([rng.normal(size=(50, 3)), [[20.0, 20.0, 20.0]]])
        distances = SimilaritySearch(metric="euclidean").fit(data).knn_distance(5)
        assert int(np.argmax(distances)) == 50

    def test_neighbours_exclude_self(self):
        data = np.random.default_rng(12).normal(size=(20, 4))
        neighbours = SimilaritySearch().fit(data).neighbours(3)
        assert neighbours.shape == (20, 3)
        assert all(i not in neighbours[i] for i in range(20))


class TestGradientBoosting:
    @pytest.mark.parametrize("backend", ["numpy", "auto"])
    def test_classifies_separable_data(self, backend):
        rng = np.random.default_rng(13)
        data = np.vstack([rng.normal(0.0, 1.0, (100, 3)),
                          rng.normal(3.0, 1.0, (100, 3))])
        labels = np.array([0] * 100 + [1] * 100)
        model = GradientBoostedClassifier(n_estimators=25, backend=backend).fit(data, labels)
        assert float((model.predict(data) == labels).mean()) > 0.9

    def test_probabilities_sum_to_one(self):
        rng = np.random.default_rng(14)
        data = rng.normal(size=(60, 3))
        labels = (data[:, 0] > 0).astype(int)
        probabilities = GradientBoostedClassifier(
            n_estimators=10, backend="numpy").fit(data, labels).predict_proba(data)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-6)

    def test_identifies_the_useless_feature(self):
        rng = np.random.default_rng(15)
        signal = rng.normal(size=(200, 1))
        noise = rng.normal(size=(200, 1))
        data = np.hstack([signal, noise])
        labels = (signal[:, 0] > 0).astype(int)
        importance = GradientBoostedClassifier(
            n_estimators=20, backend="numpy").fit(data, labels).feature_importance(
                ["signal", "noise"])
        assert importance["signal"] > importance["noise"]


class TestFeatures:
    def test_catalog_features_shape(self, measured):
        catalog, _ = measured
        matrix, names = catalog_features(catalog)
        assert matrix.shape == (len(catalog), len(FEATURE_NAMES))
        assert names == FEATURE_NAMES

    def test_features_have_no_infinities(self, measured):
        catalog, _ = measured
        matrix, _ = catalog_features(catalog)
        assert not np.isinf(matrix).any()

    def test_descriptor_is_finite_and_fixed_length(self):
        from astrovision.simulate.profiles import gaussian_psf
        stamp = gaussian_psf((32, 32), (16, 16), 3.0, 100.0)
        descriptor = cutout_descriptor(stamp)
        assert descriptor.ndim == 1 and np.isfinite(descriptor).all()
        assert cutout_descriptor(np.zeros((32, 32))).shape == descriptor.shape

    def test_embeddings_are_unit_norm(self, clean_image, measured):
        catalog, _ = measured
        embeddings = catalog_embeddings(catalog, clean_image, 48)
        assert embeddings.shape[0] == len(catalog)
        assert np.allclose(np.linalg.norm(embeddings, axis=1), 1.0, atol=1e-6)

    def test_combine_features_concatenates(self, clean_image, measured):
        catalog, _ = measured
        matrix, _ = catalog_features(catalog)
        embeddings = catalog_embeddings(catalog, clean_image, 48)
        combined = combine_features(matrix, embeddings)
        assert combined.shape[1] == matrix.shape[1] + embeddings.shape[1]
