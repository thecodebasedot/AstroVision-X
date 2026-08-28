"""Real-data loading, and moving a model between instruments."""

from __future__ import annotations

import csv

import numpy as np
import pytest

from astrovision import SkyConfig
from astrovision.core.backend import has
from astrovision.core.types import ObjectClass
from astrovision.ml.datasets import (
    MIN_VOTE_AGREEMENT,
    StampSet,
    class_balance_report,
    clean_stamp,
    load_alert_stamps,
    load_fits_cutouts,
    read_label_table,
    split_dataset,
    stamps_from_fields,
    write_fits_cutouts,
)

torch_only = pytest.mark.skipif(not has("torch"), reason="PyTorch is not installed")
astropy_only = pytest.mark.skipif(not has("astropy.io.fits"),
                                  reason="astropy is not installed")

CLASSES = [ObjectClass.STAR, ObjectClass.GALAXY, ObjectClass.NEBULA,
           ObjectClass.STAR_CLUSTER]


def source_config(seed):
    """Instrument A: a sharp Moffat PSF over a quiet background."""
    return SkyConfig(shape=(180, 180), n_stars=12, n_galaxies=8, n_nebulae=2,
                     n_clusters=2, n_lenses=0, n_anomalies=0, seed=seed,
                     seeing_fwhm=3.0, psf="moffat", background=120.0,
                     read_noise=5.0, gain=2.0)


def target_config(seed):
    """Instrument B: blurrier, Gaussian, and on a bright noisy sky."""
    return SkyConfig(shape=(180, 180), n_stars=12, n_galaxies=8, n_nebulae=2,
                     n_clusters=2, n_lenses=0, n_anomalies=0, seed=seed,
                     seeing_fwhm=5.2, psf="gaussian", background=380.0,
                     background_gradient=0.15, read_noise=9.0, gain=1.2)


class TestStampCleaning:
    def test_a_few_bad_pixels_are_filled(self):
        stamp = np.ones((16, 16))
        stamp[3, 4] = np.nan
        stamp[9, 1] = np.inf
        cleaned, reason = clean_stamp(stamp)
        assert np.isfinite(cleaned).all()
        assert "2 bad pixels" in reason

    def test_a_mostly_masked_stamp_is_refused(self):
        """Filling a few pixels is repair; filling half a stamp is invention."""
        stamp = np.full((16, 16), np.nan)
        stamp[:4] = 1.0
        cleaned, reason = clean_stamp(stamp)
        assert cleaned is None
        assert "unusable" in reason

    def test_holes_are_filled_with_the_local_level_not_zero(self):
        """A hole filled with zero is a dark patch, and a network will learn
        dark patches as a feature of whichever class had the most chip gaps."""
        stamp = np.full((16, 16), 500.0)
        stamp[8, 8] = np.nan
        cleaned, _ = clean_stamp(stamp)
        assert cleaned[8, 8] == pytest.approx(500.0)

    def test_a_clean_stamp_is_returned_unchanged(self):
        stamp = np.random.default_rng(0).normal(10.0, 1.0, (12, 12))
        cleaned, reason = clean_stamp(stamp)
        assert reason == "clean"
        assert np.allclose(cleaned, stamp)


class TestLabelTables:
    def _write(self, tmp_path, rows, header):
        path = tmp_path / "labels.csv"
        with open(path, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(rows)
        return str(path)

    def test_a_plain_class_column_is_read(self, tmp_path):
        path = self._write(tmp_path, [["a", "star"], ["b", "galaxy"]],
                           ["id", "class"])
        table = read_label_table(path, class_column="class")
        assert table == {"a": ("star", 1.0), "b": ("galaxy", 1.0)}

    def test_vote_fractions_become_a_label_and_a_weight(self, tmp_path):
        """A stamp 95 % of labellers agreed on is not the same training
        example as one they split over, and the weight is what says so."""
        path = self._write(
            tmp_path,
            [["a", "0.95", "0.05"], ["b", "0.55", "0.45"]],
            ["id", "p_star", "p_galaxy"])
        table = read_label_table(path, vote_columns={"star": "p_star",
                                                     "galaxy": "p_galaxy"},
                                 min_agreement=0.5)
        assert table["a"] == ("star", pytest.approx(0.95))
        assert table["b"][1] == pytest.approx(0.55)

    def test_an_unsettled_label_is_dropped(self, tmp_path):
        path = self._write(tmp_path, [["a", "0.51", "0.49"]],
                           ["id", "p_star", "p_galaxy"])
        table = read_label_table(path, vote_columns={"star": "p_star",
                                                     "galaxy": "p_galaxy"},
                                 min_agreement=MIN_VOTE_AGREEMENT)
        assert table == {}

    def test_a_missing_table_is_an_error(self):
        with pytest.raises(FileNotFoundError):
            read_label_table("/nowhere/labels.csv", class_column="class")


@astropy_only
class TestFitsLoading:
    @pytest.fixture(scope="class")
    def written(self, tmp_path_factory):
        directory = tmp_path_factory.mktemp("cutouts")
        rng = np.random.default_rng(4)
        dataset = StampSet(source="synthetic")
        for i in range(6):
            label = ObjectClass.STAR if i % 2 else ObjectClass.GALAXY
            dataset.add(rng.normal(100.0, 5.0, (32, 32)), label, 1.0, f"obj{i}")
        labels_path = str(directory / "labels.csv")
        write_fits_cutouts(str(directory), dataset, labels_path)
        return str(directory), labels_path

    def test_cutouts_round_trip_through_files(self, written):
        directory, labels_path = written
        table = read_label_table(labels_path, class_column="class")
        loaded = load_fits_cutouts(directory, table)
        assert len(loaded) == 6
        assert loaded.counts() == {"galaxy": 3, "star": 3}

    def test_a_label_without_a_file_is_counted(self, written):
        directory, labels_path = written
        table = read_label_table(labels_path, class_column="class")
        table["missing_object"] = ("star", 1.0)
        loaded = load_fits_cutouts(directory, table)
        assert loaded.dropped.get("label with no file") == 1

    def test_a_file_without_a_label_is_counted(self, written):
        directory, labels_path = written
        table = read_label_table(labels_path, class_column="class")
        table.pop("obj0")
        loaded = load_fits_cutouts(directory, table)
        assert loaded.dropped.get("no label for this file") == 1
        assert len(loaded) == 5

    def test_an_unknown_class_is_reported_not_guessed(self, written, tmp_path):
        directory, _ = written
        loaded = load_fits_cutouts(directory, {"obj0": ("quasar_like", 1.0)})
        assert any("unknown class" in key for key in loaded.dropped)

    def test_damaged_cutouts_are_dropped_at_the_door(self, tmp_path):
        from astropy.io import fits

        directory = tmp_path / "broken"
        directory.mkdir()
        broken = np.full((32, 32), np.nan)
        broken[:4] = 1.0
        fits.PrimaryHDU(broken.astype(np.float32)).writeto(directory / "bad.fits")
        loaded = load_fits_cutouts(str(directory), {"bad": ("star", 1.0)})
        assert len(loaded) == 0
        assert any("unusable" in key for key in loaded.dropped)


class TestAlertStamps:
    def test_the_difference_stamp_is_the_default(self, tmp_path):
        """The science stamp is dominated by the host galaxy, which is a
        property of where a transient is, not of whether it is real."""
        for name in ("science", "difference"):
            np.save(tmp_path / f"cand1_{name}.npy",
                    np.random.default_rng(1).normal(0, 1, (24, 24)))
        loaded = load_alert_stamps(str(tmp_path), {"cand1": ("star", 1.0)})
        assert len(loaded) == 1
        assert loaded.meta[0]["channel"] == "difference"

    def test_a_missing_channel_is_reported(self, tmp_path):
        np.save(tmp_path / "cand2_science.npy", np.zeros((24, 24)))
        loaded = load_alert_stamps(str(tmp_path), {"cand2": ("star", 1.0)})
        assert loaded.dropped.get("no difference stamp") == 1


class TestSplitting:
    @pytest.fixture(scope="class")
    def dataset(self):
        return stamps_from_fields(source_config, range(300, 306),
                                  source="split-test")

    def test_a_stratified_split_keeps_every_class_everywhere(self, dataset):
        parts = split_dataset(dataset, (0.6, 0.2, 0.2), seed=1)
        assert sum(len(p) for p in parts) == len(dataset)
        for part in parts:
            assert set(part.counts()) == set(dataset.counts())

    def test_a_grouped_split_does_not_share_a_field(self, dataset):
        """Stamps from one field share its noise, its PSF and its background.
        Splitting them across train and test measures memorisation."""
        parts = split_dataset(dataset, (0.6, 0.4), seed=1, by_group="seed")
        seeds = [{m["seed"] for m in part.meta} for part in parts]
        assert not (seeds[0] & seeds[1])

    def test_the_balance_report_names_the_majority_baseline(self, dataset):
        report = class_balance_report(dataset)
        assert report["imbalance"] > 1.0
        assert 0.0 < report["majority_accuracy"] < 1.0
        assert report["majority_class"] in dataset.counts()


@torch_only
class TestTransfer:
    @pytest.fixture(scope="class")
    def trained(self):
        from astrovision.ml import StampClassifier

        train = stamps_from_fields(source_config, range(400, 410), source="src")
        classifier = StampClassifier(backbone="cnn", classes=CLASSES, cutout=48,
                                     width=16, random_state=1)
        classifier.fit(train.stamps, train.labels, epochs=12, batch_size=32,
                       verbose=False)
        return classifier, train

    def test_freezing_leaves_only_the_head_trainable(self, trained):
        """Worth checking rather than believing: a typo in a layer name
        silently trains the whole network."""
        from astrovision.ml import freeze_backbone, unfreeze

        classifier, _ = trained
        counts = freeze_backbone(classifier)
        assert counts["trainable"] > 0
        assert counts["frozen"] > 10 * counts["trainable"]
        again = unfreeze(classifier)
        assert again["frozen"] == 0

    def test_the_head_can_be_replaced_for_new_classes(self, trained):
        from astrovision.ml import StampClassifier, replace_head

        train = stamps_from_fields(source_config, range(400, 403), source="src")
        classifier = StampClassifier(backbone="cnn", classes=CLASSES, cutout=48,
                                     width=16, random_state=2)
        classifier.fit(train.stamps, train.labels, epochs=2, verbose=False)
        replace_head(classifier, [ObjectClass.STAR, ObjectClass.GALAXY])
        assert classifier.classes == [ObjectClass.STAR, ObjectClass.GALAXY]
        assert classifier.predict_proba(train.stamps[:3]).shape == (3, 2)

    def test_evaluation_reports_per_class_recall_not_just_accuracy(self, trained):
        """On an imbalanced set, a model that has stopped predicting a rare
        class entirely still looks fine by accuracy."""
        from astrovision.ml import evaluate

        classifier, train = trained
        report = evaluate(classifier, train)
        assert set(report["per_class_recall"]) <= {c.value for c in CLASSES}
        assert 0.0 <= report["balanced_accuracy"] <= 1.0
        assert report["confusion"]

    def test_there_is_a_gap_between_the_two_instruments(self, trained):
        """The measurement this module exists for: a model does not transport
        between telescopes for free."""
        from astrovision.ml import evaluate

        classifier, _ = trained
        source_test = stamps_from_fields(source_config, range(700, 706),
                                         source="src-test")
        target_test = stamps_from_fields(target_config, range(800, 806),
                                         source="tgt-test")
        on_source = evaluate(classifier, source_test)["balanced_accuracy"]
        on_target = evaluate(classifier, target_test)["balanced_accuracy"]
        assert on_source > on_target

    def test_fine_tuning_stops_on_the_validation_split(self, trained):
        """With a few dozen examples the difference between fitting and
        memorising is a couple of epochs wide."""
        from astrovision.ml import StampClassifier, fine_tune

        pool = stamps_from_fields(target_config, range(500, 506), source="tgt")
        train, validation = split_dataset(pool, (0.7, 0.3), seed=0)
        classifier = StampClassifier(backbone="cnn", classes=CLASSES, cutout=48,
                                     width=16, random_state=3)
        classifier.build()
        result = fine_tune(classifier, train, validation, epochs=40, patience=5)
        assert result.epochs_run <= 40
        assert result.trainable_parameters > 0
        assert result.frozen_parameters > result.trainable_parameters
        assert result.validation_accuracy

    def test_fine_tuning_without_labels_refuses(self, trained):
        from astrovision.ml import fine_tune

        classifier, _ = trained
        result = fine_tune(classifier, StampSet(), epochs=2)
        assert result.n_target_labels == 0
        assert "no target-domain labels" in result.reason

    def test_a_label_outside_the_class_set_is_an_error(self, trained):
        from astrovision.core.exceptions import ModelError
        from astrovision.ml import fine_tune

        classifier, _ = trained
        odd = StampSet()
        odd.add(np.zeros((48, 48)), ObjectClass.ARTIFACT, 1.0, "x")
        with pytest.raises(ModelError):
            fine_tune(classifier, odd, epochs=1)

    def test_the_study_compares_against_training_from_scratch(self):
        """Without that comparison, 'fine-tuning reached 80 %' is
        unfalsifiable -- those examples alone might have got there."""
        from astrovision.ml import StampClassifier, domain_study

        source_train = stamps_from_fields(source_config, range(400, 408),
                                          source="src")
        source_test = stamps_from_fields(source_config, range(700, 704),
                                         source="src-test")
        target_pool = stamps_from_fields(target_config, range(500, 508),
                                         source="tgt-pool")
        target_test = stamps_from_fields(target_config, range(800, 804),
                                         source="tgt-test")

        def train_source():
            classifier = StampClassifier(backbone="cnn", classes=CLASSES,
                                         cutout=48, width=16, random_state=1)
            classifier.fit(source_train.stamps, source_train.labels, epochs=8,
                           batch_size=32, verbose=False)
            return classifier

        study = domain_study(train_source, source_test, target_pool, target_test,
                             label_budgets=(20,), epochs=8, seed=1, repeats=2)
        assert study.curve
        row = study.curve[0]
        assert "from_scratch" in row
        assert "transfer_advantage" in row
        assert np.isfinite(study.gap)
        assert "gap" in study.summary()

    def test_each_budget_is_drawn_more_than_once(self):
        """A single draw of 25 labels scored 0.837 here and five draws of the
        same size gave 0.795 +/- 0.059. Reporting the first draw would have
        claimed a recovery three of the five draws did not reach."""
        from astrovision.ml import StampClassifier, domain_study

        train = stamps_from_fields(source_config, range(400, 404), source="src")
        pool = stamps_from_fields(target_config, range(500, 506), source="tgt")

        def train_source():
            classifier = StampClassifier(backbone="cnn", classes=CLASSES,
                                         cutout=48, width=16, random_state=1)
            classifier.fit(train.stamps, train.labels, epochs=4, verbose=False)
            return classifier

        study = domain_study(train_source, train, pool, pool,
                             label_budgets=(16,), epochs=4, seed=2, repeats=3,
                             scratch_comparison=False)
        row = study.curve[0]
        assert row["repeats"] == 3
        assert len(row["fine_tuned_runs"]) == 3
        assert row["fine_tuned_sd"] >= 0.0

    def test_a_budget_larger_than_the_pool_is_refused_not_faked(self):
        from astrovision.ml import StampClassifier, domain_study

        source_train = stamps_from_fields(source_config, range(400, 403),
                                          source="src")
        target_pool = stamps_from_fields(target_config, range(500, 502),
                                         source="tgt")

        def train_source():
            classifier = StampClassifier(backbone="cnn", classes=CLASSES,
                                         cutout=48, width=16, random_state=1)
            classifier.fit(source_train.stamps, source_train.labels, epochs=2,
                           verbose=False)
            return classifier

        study = domain_study(train_source, source_train, target_pool,
                             target_pool, label_budgets=(10000,), epochs=2,
                             scratch_comparison=False)
        assert not study.curve
        assert any("exceeds" in note for note in study.notes)
