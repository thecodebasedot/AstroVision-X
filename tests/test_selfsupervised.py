"""Learning a representation without labels, and what the augmentations decide."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision import SkyConfig
from astrovision.core.backend import has
from astrovision.core.types import ObjectClass
from astrovision.ml.datasets import stamps_from_fields
from astrovision.ml.selfsupervised import (
    AugmentationPolicy,
    ContrastiveEncoder,
    anomaly_ranking_quality,
    augment,
    linear_probe,
)

torch_only = pytest.mark.skipif(not has("torch"), reason="PyTorch is not installed")

CLASSES = [ObjectClass.STAR, ObjectClass.GALAXY, ObjectClass.NEBULA,
           ObjectClass.STAR_CLUSTER]


def field_config(seed):
    return SkyConfig(shape=(200, 200), n_stars=14, n_galaxies=8, n_nebulae=3,
                     n_clusters=3, n_lenses=0, n_anomalies=0, seed=seed,
                     seeing_fwhm=3.0, psf="moffat", background=120.0)


def compactness(stamp):
    """Second moment of the light: small for a star, large for a galaxy."""
    data = np.asarray(stamp, dtype=float)
    weights = np.clip(data - np.median(data), 0, None)
    if weights.sum() <= 0:
        return float("nan")
    yy, xx = np.mgrid[:data.shape[0], :data.shape[1]]
    cy = (weights * yy).sum() / weights.sum()
    cx = (weights * xx).sum() / weights.sum()
    return float(np.sqrt((weights * ((yy - cy) ** 2 + (xx - cx) ** 2)).sum()
                         / weights.sum()))


class TestAugmentations:
    @pytest.fixture(scope="class")
    def star(self):
        rng = np.random.default_rng(0)
        yy, xx = np.mgrid[:48, :48]
        stamp = 4000.0 * np.exp(-0.5 * (((yy - 23.5) ** 2 + (xx - 23.5) ** 2) / 1.8 ** 2))
        return stamp + rng.normal(100.0, 4.0, (48, 48))

    def test_rotation_and_reflection_preserve_the_object(self, star):
        """The sky has no preferred orientation, so these are exact symmetries
        rather than approximations, and the object's size must not move."""
        policy = AugmentationPolicy(translate=0, noise=0.0, blur=0.0,
                                    brightness=0.0)
        rng = np.random.default_rng(1)
        sizes = [compactness(augment(star, policy, rng)) for _ in range(8)]
        assert np.std(sizes) < 0.05 * np.mean(sizes)

    def test_a_resized_crop_changes_the_apparent_size(self, star):
        """The mechanism behind the default: a crop that rescales makes an
        unresolved source look resolved. Whether that *costs* anything is a
        separate question, and the answer measured here was no -- see the
        module docstring. This test asserts only the mechanism."""
        plain = AugmentationPolicy(translate=0, noise=0.0, blur=0.0,
                                   brightness=0.0)
        cropped = AugmentationPolicy(translate=0, noise=0.0, blur=0.0,
                                     brightness=0.0, resized_crop=True,
                                     crop_range=(0.4, 0.6))
        rng = np.random.default_rng(2)
        base = np.mean([compactness(augment(star, plain, rng)) for _ in range(6)])
        rng = np.random.default_rng(2)
        scaled = [compactness(augment(star, cropped, rng)) for _ in range(6)]
        assert np.nanmax(scaled) > 1.3 * base

    def test_the_default_policy_leaves_scale_alone(self):
        assert AugmentationPolicy().resized_crop is False

    def test_augmentation_keeps_the_stamp_shape_and_stays_finite(self, star):
        rng = np.random.default_rng(3)
        for policy in (AugmentationPolicy(), AugmentationPolicy(resized_crop=True)):
            out = augment(star, policy, rng)
            assert out.shape == star.shape
            assert np.isfinite(out).all()

    def test_noise_augmentation_scales_with_the_stamp(self, star):
        quiet = augment(star, AugmentationPolicy(noise=0.0, blur=0.0,
                                                 translate=0, brightness=0.0,
                                                 rotate=False, flip=False),
                        np.random.default_rng(4))
        loud = augment(star, AugmentationPolicy(noise=2.0, blur=0.0,
                                                translate=0, brightness=0.0,
                                                rotate=False, flip=False),
                       np.random.default_rng(4))
        assert np.std(loud) > np.std(quiet)


@torch_only
class TestContrastiveTraining:
    @pytest.fixture(scope="module")
    def pretrained(self):
        unlabelled = stamps_from_fields(field_config, range(400, 416),
                                        source="unlabelled")
        encoder = ContrastiveEncoder(cutout=48, width=16, random_state=1)
        encoder.fit(unlabelled.stamps, epochs=25, batch_size=64, verbose=False)
        return encoder, unlabelled

    def test_the_loss_falls(self, pretrained):
        encoder, _ = pretrained
        assert encoder.result_.loss[-1] < encoder.result_.loss[0]

    def test_it_never_sees_a_label(self, pretrained):
        """``fit`` takes stamps and nothing else, so a run that claims to be
        unsupervised cannot quietly have used labels."""
        import inspect

        encoder, _ = pretrained
        parameters = inspect.signature(ContrastiveEncoder.fit).parameters
        assert "labels" not in parameters
        assert encoder.result_.n_stamps > 0

    def test_the_embedding_has_the_advertised_width(self, pretrained):
        encoder, unlabelled = pretrained
        embedded = encoder.embed(unlabelled.stamps[:5])
        assert embedded.shape == (5, encoder.result_.embedding_dim)

    def test_two_views_of_one_stamp_land_near_each_other(self, pretrained):
        """What the loss asks for. If it does not hold, nothing was learned."""
        encoder, unlabelled = pretrained
        rng = np.random.default_rng(7)
        policy = AugmentationPolicy()
        stamps = unlabelled.stamps[:12]
        first = encoder.embed([augment(s, policy, rng) for s in stamps])
        second = encoder.embed([augment(s, policy, rng) for s in stamps])

        def unit(x):
            return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)

        similarity = unit(first) @ unit(second).T
        same = np.diag(similarity)
        different = similarity[~np.eye(len(stamps), dtype=bool)]
        assert float(np.mean(same)) > float(np.mean(different))

    def test_it_refuses_a_set_too_small_to_contrast(self):
        from astrovision.core.exceptions import ModelError

        encoder = ContrastiveEncoder(cutout=48, width=8)
        with pytest.raises(ModelError):
            encoder.fit([np.zeros((48, 48))], epochs=1)

    def test_the_features_carry_class_information(self, pretrained):
        """A linear probe asks what the representation *contains*, as opposed
        to what a network could be trained to do with it."""
        encoder, _ = pretrained
        labelled = stamps_from_fields(field_config, range(600, 606),
                                      source="labelled")
        test = stamps_from_fields(field_config, range(700, 704), source="test")
        report = linear_probe(encoder.embed(labelled.stamps),
                              [l.value for l in labelled.labels],
                              encoder.embed(test.stamps),
                              [l.value for l in test.labels])
        counts = test.counts()
        majority = max(counts.values()) / sum(counts.values())
        assert report["accuracy"] > majority

    def test_the_pretrained_features_become_a_classifier(self, pretrained):
        encoder, _ = pretrained
        classifier = encoder.to_classifier(CLASSES)
        assert classifier.classes == CLASSES
        probabilities = classifier.predict_proba([np.zeros((48, 48))])
        assert probabilities.shape == (1, len(CLASSES))

    def test_converting_before_training_is_an_error(self):
        from astrovision.core.exceptions import NotFittedError

        with pytest.raises(NotFittedError):
            ContrastiveEncoder().to_classifier(CLASSES)


class TestEmbeddingQuality:
    def test_anomaly_ranking_scores_a_separable_case_highly(self):
        rng = np.random.default_rng(0)
        ordinary = rng.normal(0, 1, (60, 8))
        odd = rng.normal(9, 1, (6, 8))
        flags = [False] * 60 + [True] * 6
        report = anomaly_ranking_quality(np.vstack([ordinary, odd]), flags)
        assert report["auc"] > 0.9
        assert report["n_anomalies"] == 6

    def test_an_embedding_carrying_nothing_scores_near_chance(self):
        rng = np.random.default_rng(1)
        embeddings = rng.normal(0, 1, (60, 8))
        flags = [i % 10 == 0 for i in range(60)]
        assert abs(anomaly_ranking_quality(embeddings, flags)["auc"] - 0.5) < 0.25

    def test_it_refuses_when_there_is_nothing_to_separate(self):
        assert np.isnan(anomaly_ranking_quality(np.zeros((10, 3)),
                                                [False] * 10)["auc"])

    def test_the_probe_reports_per_class_recall(self):
        rng = np.random.default_rng(2)
        train = np.vstack([rng.normal(0, 1, (40, 4)), rng.normal(4, 1, (40, 4))])
        labels = ["a"] * 40 + ["b"] * 40
        test = np.vstack([rng.normal(0, 1, (20, 4)), rng.normal(4, 1, (20, 4))])
        test_labels = ["a"] * 20 + ["b"] * 20
        report = linear_probe(train, labels, test, test_labels)
        assert report["accuracy"] > 0.9
        assert set(report["per_class_recall"]) == {"a", "b"}
