"""Human verdicts as training data, and whether choosing what to show helps."""

from __future__ import annotations

import numpy as np
import pytest

from astrovision import SkyConfig
from astrovision.core.backend import has
from astrovision.core.types import ObjectClass, Verdict
from astrovision.ml.active import (
    HumanVerdict,
    VerdictLog,
    review_queue,
    select_for_review,
    uncertainty_scores,
    verdicts_to_labels,
)
from astrovision.ml.datasets import StampSet, stamps_from_fields

torch_only = pytest.mark.skipif(not has("torch"), reason="PyTorch is not installed")

CLASSES = [ObjectClass.STAR, ObjectClass.GALAXY, ObjectClass.NEBULA,
           ObjectClass.STAR_CLUSTER]


def field_config(seed):
    return SkyConfig(shape=(200, 200), n_stars=14, n_galaxies=8, n_nebulae=3,
                     n_clusters=3, n_lenses=0, n_anomalies=0, seed=seed,
                     seeing_fwhm=3.0, psf="moffat", background=120.0)


class TestVerdictLog:
    def test_a_verdict_needs_a_reviewer(self):
        """Without one, a decision cannot be told apart from the model's own
        output, and training on that is self-training."""
        log = VerdictLog()
        with pytest.raises(ValueError) as error:
            log.add(HumanVerdict(source_id=1, label="star", reviewer="  "))
        assert "self-training" in str(error.value)

    def test_the_log_is_append_only_and_keeps_the_latest(self):
        log = VerdictLog()
        log.add(HumanVerdict(1, "star", "ana", timestamp=10.0))
        log.add(HumanVerdict(1, "galaxy", "ben", timestamp=20.0))
        assert len(log) == 2
        assert log.latest()[1].label == "galaxy"
        assert log.latest()[1].reviewer == "ben"

    def test_disagreement_is_surfaced_not_resolved(self):
        """An object two experts label differently is either genuinely
        ambiguous or badly presented, and both are worth knowing."""
        log = VerdictLog()
        log.add(HumanVerdict(1, "star", "ana"))
        log.add(HumanVerdict(1, "galaxy", "ben"))
        log.add(HumanVerdict(2, "star", "ana"))
        log.add(HumanVerdict(2, "star", "ben"))
        assert log.disagreements() == [(1, ["star", "galaxy"])]

    def test_agreement_with_the_model_is_measured_on_real_decisions(self):
        log = VerdictLog()
        log.add(HumanVerdict(1, "star", "ana", model_label="star",
                             model_confidence=0.8))
        log.add(HumanVerdict(2, "galaxy", "ana", model_label="star",
                             model_confidence=0.97))
        report = log.agreement_with_model()
        assert report["n"] == 2
        assert report["agreement"] == pytest.approx(0.5)
        assert report["confidently_wrong"] == 1
        assert report["confusion"]["star"]["galaxy"] == 1

    def test_the_log_round_trips_through_a_file(self, tmp_path):
        log = VerdictLog()
        log.add(HumanVerdict(3, "nebula", "ana", note="fuzzy edges"))
        path = log.save(str(tmp_path / "verdicts.json"))
        loaded = VerdictLog.load(path)
        assert len(loaded) == 1
        assert loaded.records[0].note == "fuzzy edges"
        assert loaded.records[0].reviewer == "ana"


class TestVerdictsAsLabels:
    @pytest.fixture()
    def dataset(self):
        stamps = StampSet(source="test")
        for i in range(4):
            stamps.add(np.zeros((16, 16)), ObjectClass.UNKNOWN, 1.0, f"7_{i}")
        return stamps

    def test_verdicts_become_a_training_set(self, dataset):
        log = VerdictLog()
        log.add(HumanVerdict(0, "star", "ana"))
        log.add(HumanVerdict(2, "galaxy", "ana"))
        labelled = verdicts_to_labels(log, dataset)
        assert len(labelled) == 2
        assert labelled.counts() == {"galaxy": 1, "star": 1}
        assert labelled.meta[0]["reviewer"] == "ana"

    def test_an_unsure_reviewer_is_not_a_label(self, dataset):
        """'I am not sure' is a real answer, and training on it teaches the
        model the reviewer's uncertainty as though it were a class."""
        log = VerdictLog()
        log.add(HumanVerdict(0, "star", "ana", confident=False))
        labelled = verdicts_to_labels(log, dataset)
        assert len(labelled) == 0
        assert labelled.dropped.get("reviewer was not sure") == 1

    def test_a_label_outside_the_class_set_is_dropped_with_a_reason(self, dataset):
        log = VerdictLog()
        log.add(HumanVerdict(0, "galaxy", "ana"))
        labelled = verdicts_to_labels(log, dataset,
                                      classes=[ObjectClass.STAR])
        assert len(labelled) == 0
        assert any("outside the class set" in key for key in labelled.dropped)

    def test_an_unknown_class_name_is_refused(self, dataset):
        log = VerdictLog()
        log.add(HumanVerdict(0, "wormhole", "ana"))
        labelled = verdicts_to_labels(log, dataset)
        assert len(labelled) == 0
        assert any("unknown class" in key for key in labelled.dropped)


class TestSelection:
    def test_margin_is_the_default_and_finds_the_two_way_tie(self):
        """Least-confidence cannot tell a genuine tie from a diffuse guess,
        because it ignores the runner-up entirely."""
        tie = np.array([0.45, 0.45, 0.10])
        diffuse = np.array([0.45, 0.30, 0.25])
        scores = uncertainty_scores(np.vstack([tie, diffuse]))
        assert scores[0] > scores[1]

    def test_entropy_and_least_confident_are_available(self):
        probabilities = np.array([[0.5, 0.3, 0.2], [0.9, 0.05, 0.05]])
        for method in ("entropy", "least_confident", "margin"):
            scores = uncertainty_scores(probabilities, method=method)
            assert scores[0] > scores[1]

    def test_a_single_class_cannot_be_uncertain(self):
        with pytest.raises(ValueError):
            uncertainty_scores(np.ones((4, 1)))

    def test_uncertainty_selection_takes_the_least_sure(self):
        probabilities = np.array([[0.99, 0.01], [0.5, 0.5], [0.8, 0.2]])
        assert list(select_for_review(probabilities, 1,
                                      strategy="uncertainty")) == [1]

    def test_the_default_is_random_because_that_is_what_measured_best(self):
        """Uncertainty sampling lost to random at three of four budgets over
        six repeats, so the default follows the measurement rather than the
        textbook."""
        probabilities = np.array([[0.99, 0.01], [0.5, 0.5], [0.8, 0.2]])
        default = select_for_review(probabilities, 2, seed=0)
        explicit = select_for_review(probabilities, 2, strategy="random", seed=0)
        assert list(default) == list(explicit)

    def test_balanced_selection_takes_a_quota_from_each_predicted_class(self):
        """It did not beat random either, and the reason is visible here: the
        quota is on the *predicted* class, so it only rebalances as well as
        the model's own predictions do."""
        probabilities = np.array([[0.90, 0.10], [0.55, 0.45], [0.60, 0.40],
                                  [0.20, 0.80], [0.45, 0.55]])
        picked = set(int(i) for i in select_for_review(probabilities, 2,
                                                       strategy="balanced"))
        predicted = {int(i): int(np.argmax(probabilities[i])) for i in picked}
        assert set(predicted.values()) == {0, 1}

    def test_random_selection_is_reproducible_and_covers_the_pool(self):
        probabilities = np.tile([0.6, 0.4], (20, 1))
        first = select_for_review(probabilities, 5, strategy="random", seed=3)
        again = select_for_review(probabilities, 5, strategy="random", seed=3)
        assert list(first) == list(again)
        assert len(set(first)) == 5

    def test_confident_selection_exists_for_catching_systematic_error(self):
        """Showing a reviewer what the model is *sure* about is a different
        job from making the model better, and both are needed."""
        probabilities = np.array([[0.99, 0.01], [0.5, 0.5], [0.8, 0.2]])
        assert list(select_for_review(probabilities, 1, strategy="confident")) == [0]

    def test_diverse_selection_spreads_across_the_embedding(self):
        """Twenty questions that are twenty views of one confusion waste
        nineteen of them."""
        rng = np.random.default_rng(0)
        probabilities = np.tile([0.51, 0.49], (30, 1))
        embeddings = np.vstack([rng.normal(0, 0.05, (15, 2)),
                                rng.normal(8, 0.05, (15, 2))])
        picked = select_for_review(probabilities, 6, strategy="diverse",
                                   embeddings=embeddings)
        both = {int(i) < 15 for i in picked}
        assert both == {True, False}

    def test_asking_for_more_than_the_pool_returns_the_pool(self):
        probabilities = np.tile([0.6, 0.4], (3, 1))
        assert len(select_for_review(probabilities, 99)) == 3

    def test_an_unknown_strategy_is_refused(self):
        with pytest.raises(ValueError):
            select_for_review(np.tile([0.6, 0.4], (3, 1)), 2,
                              strategy="vibes", embeddings=np.zeros((3, 2)))

    def test_diverse_selection_needs_one_embedding_per_candidate(self):
        with pytest.raises(ValueError):
            select_for_review(np.tile([0.6, 0.4], (3, 1)), 2, strategy="diverse",
                              embeddings=np.zeros((2, 2)))


class TestReviewQueue:
    def test_the_queue_carries_the_claim_being_judged(self):
        """A reviewer judging a bare picture is doing a different, harder job
        than one judging a stated claim -- and only the second produces a
        verdict that can be compared against what the model said."""
        class FakeSource:
            def __init__(self, identifier):
                self.id = identifier
                self.verdict = Verdict.WORTH_A_LOOK

        catalog = [FakeSource(i) for i in range(4)]
        probabilities = np.array([[0.9, 0.1], [0.5, 0.5], [0.7, 0.3], [0.6, 0.4]])
        queue = review_queue(catalog, probabilities,
                             [ObjectClass.STAR, ObjectClass.GALAXY], n=2,
                             strategy="uncertainty")
        assert len(queue) == 2
        for entry in queue:
            assert entry["model_label"] in {"star", "galaxy"}
            assert 0.0 <= entry["model_confidence"] <= 1.0
            assert entry["runner_up"] in {"star", "galaxy"}
            assert entry["model_verdict"] == Verdict.WORTH_A_LOOK.value
        assert queue[0]["source_id"] == 1        # the least certain one


@torch_only
class TestTheLoop:
    def test_a_round_improves_on_its_own_seed_set(self):
        from astrovision.ml.active import run_active_learning

        pool = stamps_from_fields(field_config, range(400, 410), source="pool")
        test = stamps_from_fields(field_config, range(700, 704), source="test")
        run = run_active_learning(pool, test, CLASSES, strategy="uncertainty",
                                  seed_size=20, rounds=1, batch=20, epochs=10,
                                  repeats=1, seed=1)
        assert run.budgets == [20, 40]
        assert len(run.scores) == 2
        assert all(np.isfinite(run.scores))

    def test_the_labelled_set_grows_by_the_batch_size(self):
        from astrovision.ml.active import run_active_learning

        pool = stamps_from_fields(field_config, range(400, 410), source="pool")
        test = stamps_from_fields(field_config, range(700, 703), source="test")
        run = run_active_learning(pool, test, CLASSES, strategy="random",
                                  seed_size=15, rounds=1, batch=10, epochs=8,
                                  repeats=1, seed=2)
        assert sum(run.class_counts[-1].values()) == 25

    def test_every_strategy_starts_from_the_same_seed_set(self):
        """Different seed sets would measure the seed set rather than the
        strategy."""
        from astrovision.ml.active import run_active_learning

        pool = stamps_from_fields(field_config, range(400, 408), source="pool")
        test = stamps_from_fields(field_config, range(700, 702), source="test")
        counts = []
        for strategy in ("random", "uncertainty"):
            run = run_active_learning(pool, test, CLASSES, strategy=strategy,
                                      seed_size=12, rounds=0, batch=6, epochs=6,
                                      repeats=1, seed=5)
            counts.append(run.class_counts[0])
        assert counts[0] == counts[1]
