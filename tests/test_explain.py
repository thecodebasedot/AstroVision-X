"""Explanations, and whether they describe the model or merely look like it."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision import SkyConfig
from astrovision.core.backend import has
from astrovision.core.types import ObjectClass
from astrovision.ml import GradientBoostedClassifier
from astrovision.ml.datasets import stamps_from_fields
from astrovision.ml.explain import (
    Attribution,
    cam_matches_head_weights,
    deletion_curve,
    explain_prediction,
    explain_stamp,
    grad_cam,
    occlusion_map,
    retrieval_purity,
    retrieve_similar,
    shapley_values,
)

torch_only = pytest.mark.skipif(not has("torch"), reason="PyTorch is not installed")

CLASSES = [ObjectClass.STAR, ObjectClass.GALAXY, ObjectClass.NEBULA,
           ObjectClass.STAR_CLUSTER]


def field_config(seed):
    return SkyConfig(shape=(200, 200), n_stars=14, n_galaxies=8, n_nebulae=3,
                     n_clusters=3, n_lenses=0, n_anomalies=0, seed=seed,
                     seeing_fwhm=3.0, psf="moffat", background=120.0)


@pytest.fixture(scope="module")
def trained():
    from astrovision.ml import StampClassifier

    train = stamps_from_fields(field_config, range(400, 412), source="train")
    test = stamps_from_fields(field_config, range(700, 704), source="test")
    classifier = StampClassifier(backbone="cnn", classes=CLASSES, cutout=48,
                                 width=16, random_state=1)
    classifier.fit(train.stamps, train.labels, epochs=25, batch_size=32,
                   verbose=False)
    return classifier, test


@torch_only
class TestSaliency:
    def test_the_gradient_weights_equal_the_head_weights(self, trained):
        """With global average pooling into one linear head this is an
        identity. A mismatch means the gradient is not flowing where the map
        assumes it is."""
        classifier, test = trained
        check = cam_matches_head_weights(classifier, test.stamps[0])
        assert check["agrees"]
        assert check["max_difference"] < 1e-6

    def test_grad_cam_returns_a_map_the_size_of_the_stamp(self, trained):
        classifier, test = trained
        cam = grad_cam(classifier, test.stamps[0])
        assert cam.heatmap.shape == np.asarray(test.stamps[0]).shape
        assert 0.0 <= cam.heatmap.min() <= cam.heatmap.max() <= 1.0
        assert cam.native_shape == (12, 12)
        assert cam.predicted_class in {c.value for c in CLASSES}

    def test_grad_cam_says_what_resolution_it_actually_has(self, trained):
        """Twelve cells across a 48-pixel stamp. The heatmap is upsampled and
        the record says so, because reading structure into the interpolation
        is reading the interpolation."""
        classifier, test = trained
        cam = grad_cam(classifier, test.stamps[0])
        assert cam.to_dict()["native_shape"] == [12, 12]

    def test_occlusion_beats_grad_cam_on_faithfulness(self, trained):
        """Measured, not assumed: on this architecture the famous method is
        the worse one, and the default follows the measurement."""
        classifier, test = trained
        cam_scores, occlusion_scores = [], []
        for i in range(6):
            stamp = test.stamps[i]
            cam_scores.append(deletion_curve(
                classifier, stamp, grad_cam(classifier, stamp).heatmap,
                seed=i)["advantage"])
            occlusion_scores.append(deletion_curve(
                classifier, stamp, occlusion_map(classifier, stamp).heatmap,
                seed=i)["advantage"])
        assert float(np.mean(occlusion_scores)) > float(np.mean(cam_scores))

    def test_occlusion_puts_its_mass_on_the_object(self, trained):
        """The object is at the centre of the stamp by construction, so a map
        that is no more concentrated there than a uniform map is not pointing
        at anything."""
        classifier, test = trained
        uniform = (16 * 16) / (48 * 48)
        concentrations = []
        for i in range(6):
            heatmap = occlusion_map(classifier, test.stamps[i]).heatmap
            concentrations.append(
                float(heatmap[16:32, 16:32].sum() / max(heatmap.sum(), 1e-9)))
        assert float(np.mean(concentrations)) > 2.0 * uniform

    def test_the_deletion_test_defaults_to_a_noise_preserving_fill(self, trained):
        """Filling with a constant narrows the stamp's noise distribution, and
        the classifier's stretch is computed from that distribution -- so the
        constant fill flatters every map."""
        classifier, test = trained
        heatmap = occlusion_map(classifier, test.stamps[0]).heatmap
        assert deletion_curve(classifier, test.stamps[0], heatmap)["fill"] == "noise"
        constant = deletion_curve(classifier, test.stamps[0], heatmap,
                                  fill="constant")
        assert constant["fill"] == "constant"

    def test_deletion_reports_both_curves_and_their_difference(self, trained):
        classifier, test = trained
        heatmap = occlusion_map(classifier, test.stamps[0]).heatmap
        result = deletion_curve(classifier, test.stamps[0], heatmap)
        assert len(result["guided"]) == len(result["fractions"])
        assert len(result["random"]) == len(result["fractions"])
        assert np.isfinite(result["advantage"])

    def test_an_unknown_method_is_refused(self, trained):
        classifier, test = trained
        with pytest.raises(ValueError):
            explain_stamp(classifier, test.stamps[0], method="lime")

    def test_the_default_method_is_the_one_that_measured_better(self, trained):
        classifier, test = trained
        assert explain_stamp(classifier, test.stamps[0]).method == "occlusion"


class TestAttributions:
    @pytest.fixture(scope="class")
    def model(self):
        """Two informative features among six that carry nothing."""
        rng = np.random.default_rng(0)
        X = rng.normal(0, 1, (400, 8))
        y = ((1.5 * X[:, 2] - 1.0 * X[:, 5]
              + 0.3 * rng.normal(0, 1, 400)) > 0).astype(int)
        return GradientBoostedClassifier(n_estimators=60,
                                         backend="numpy").fit(X, y), X

    def test_it_finds_the_features_that_matter(self, model):
        fitted, X = model
        names = [f"f{i}" for i in range(8)]
        found = 0
        for i in range(6):
            attribution = explain_prediction(fitted, X[i], X[:80],
                                             feature_names=names,
                                             n_samples=120, seed=i)
            found += set(k for k, _ in attribution.top(2)) == {"f2", "f5"}
        assert found >= 5

    def test_noise_features_get_attributions_near_zero(self, model):
        fitted, X = model
        names = [f"f{i}" for i in range(8)]
        totals = {name: 0.0 for name in names}
        for i in range(8):
            attribution = explain_prediction(fitted, X[i], X[:80],
                                             feature_names=names,
                                             n_samples=100, seed=i)
            for name, value in attribution.values.items():
                totals[name] += abs(value) / 8
        informative = max(totals["f2"], totals["f5"])
        noise = max(totals[n] for n in names if n not in {"f2", "f5"})
        assert informative > 10 * noise

    def test_the_attributions_add_up_to_the_prediction(self, model):
        """Shapley values sum with the base rate to the model's own output.
        A residual beyond the sampling error means something other than the
        prediction is being attributed."""
        fitted, X = model
        attribution = explain_prediction(fitted, X[0], X[:80], n_samples=200,
                                         seed=1)
        assert attribution.additivity_error() < 0.05

    def test_every_estimate_carries_its_error(self, model):
        """A feature the model never splits on has a contribution of exactly
        zero in every permutation, so its error is exactly zero too -- that is
        the right answer, not a missing one."""
        fitted, X = model
        names = [f"f{i}" for i in range(8)]
        attribution = explain_prediction(fitted, X[0], X[:80],
                                         feature_names=names, n_samples=60,
                                         seed=1)
        assert set(attribution.errors) == set(attribution.values)
        assert all(e >= 0 and np.isfinite(e) for e in attribution.errors.values())
        assert attribution.errors["f2"] > 0
        for name in names:
            if attribution.errors[name] == 0.0:
                assert attribution.values[name] == 0.0

    def test_more_samples_narrow_the_error(self, model):
        """The estimate improves as one over the square root of the draws, and
        a run that has not converged says so rather than looking finished."""
        fitted, X = model
        small = explain_prediction(fitted, X[0], X[:80], n_samples=50, seed=1)
        large = explain_prediction(fitted, X[0], X[:80], n_samples=400, seed=1)
        assert max(large.errors.values()) < max(small.errors.values())

    def test_the_explanation_reads_as_a_sentence(self, model):
        fitted, X = model
        text = explain_prediction(fitted, X[0], X[:80],
                                  feature_names=[f"f{i}" for i in range(8)],
                                  n_samples=80).explain()
        assert "against a base rate of" in text
        assert "raised it by" in text or "lowered it by" in text

    def test_a_mismatched_background_is_an_error(self):
        with pytest.raises(ValueError):
            shapley_values(lambda x: np.zeros((len(x), 2)), np.zeros(4),
                           np.zeros((3, 5)))

    def test_an_unfitted_model_cannot_be_explained(self):
        from astrovision.core.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            explain_prediction(GradientBoostedClassifier(), np.zeros(3),
                               np.zeros((2, 3)))

    def test_an_empty_attribution_explains_itself(self):
        assert Attribution().explain() == "no attribution available"


@torch_only
class TestRetrieval:
    @pytest.fixture(scope="module")
    def embedded(self, trained):
        classifier, test = trained
        return classifier.embed(test.stamps), [l.value for l in test.labels]

    def test_neighbours_share_the_query_class_more_often_than_chance(self, embedded):
        """The check that makes retrieval an explanation rather than a list.
        Chance is the probability two random objects share a class, which for
        imbalanced classes is not one over their number."""
        embeddings, labels = embedded
        report = retrieval_purity(embeddings, labels, n=3)
        assert report["beats_chance"]
        assert report["lift"] > 1.5

    def test_the_learned_embedding_beats_comparing_pixels(self, embedded, trained):
        """If raw pixels retrieved as well, the embedding would not be adding
        anything and the explanation could be built without a model."""
        embeddings, labels = embedded
        _, test = trained
        pixels = np.array([np.asarray(s, dtype=float).ravel() for s in test.stamps])
        pixels = ((pixels - pixels.mean(axis=1, keepdims=True))
                  / (pixels.std(axis=1, keepdims=True) + 1e-9))
        assert (retrieval_purity(embeddings, labels, n=3)["purity"]
                > retrieval_purity(pixels, labels, n=3)["purity"])

    def test_retrieval_reports_distance_against_the_typical_separation(self, embedded):
        """A distance alone means nothing; the same number is close in one
        embedding and remote in another."""
        embeddings, labels = embedded
        found = retrieve_similar(embeddings, 0, n=3, labels=labels)
        assert len(found.indices) == 3
        assert found.typical_distance > 0
        assert np.isfinite(found.isolation)
        assert found.distances == sorted(found.distances)

    def test_the_query_is_not_its_own_neighbour(self, embedded):
        embeddings, labels = embedded
        assert 5 not in retrieve_similar(embeddings, 5, n=3).indices

    def test_retrieval_explains_itself_in_words(self, embedded):
        embeddings, labels = embedded
        text = retrieve_similar(embeddings, 0, n=3, labels=labels).explain()
        assert "nearest in the embedding" in text
        assert "typical separation" in text

    def test_too_few_objects_returns_nothing_rather_than_guessing(self):
        assert not retrieve_similar(np.zeros((1, 4)), 0).indices
        assert np.isnan(retrieval_purity(np.zeros((2, 4)), ["a", "b"])["purity"])
