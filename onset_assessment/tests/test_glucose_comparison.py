#### Verify within-person class comparisons, participant weighting, ranking ties and displays.
#### Synthetic participants with opposing directions and unequal day counts expose pooling errors.

import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from onset_assessment.glucose_comparison import (FEATURES, centred_distributions, load_data,
                                                participant_comparisons, participant_coverage,
                                                summary_table, validate_data, write_report)
from onset_assessment.glucose_plots import plot_differences, plot_distributions
from onset_prediction import protocol
from preprocessing import file_sha256


def synthetic_features():
    #### Create two paired participants plus one all-negative and one all-positive participant.
    rows = []
    records = [(1, 0, value) for value in range(10)] + [(1, 1, 1)]
    records += [(2, 0, 0)] + [(2, 1, 9)] * 10 + [(3, 0, 100), (4, 1, -100)]
    for number, (person, outcome, value) in enumerate(records, start=1):
        rows.append({"id": person, "study_interval": 2022, "day_in_study": number,
                     protocol.TARGET: outcome, **{feature: float(value) for feature in FEATURES}})
    return pd.DataFrame(rows)


class GlucoseComparisonTests(unittest.TestCase):
    def test_coverage_and_equal_participant_weights(self):
        #### Person one's many negative days cannot outweigh person two's many positive days.
        data = synthetic_features()
        coverage = participant_coverage(data)
        self.assertEqual(coverage.set_index("id").reason.to_dict(), {
            1: "both_classes", 2: "both_classes", 3: "no_positive_days", 4: "no_negative_days"})
        eligible, comparisons = participant_comparisons(data, coverage)
        self.assertEqual(set(eligible.id), {1, 2})
        self.assertEqual(len(comparisons), 12)
        summary = summary_table(comparisons)
        np.testing.assert_allclose(summary.mean_difference, (-3.5 + 9) / 2)
        self.assertTrue(summary.positive_days.eq(11).all())
        self.assertTrue(summary.negative_days.eq(11).all())
        self.assertTrue(summary.positive_participants.eq(1).all())
        self.assertTrue(summary.negative_participants.eq(1).all())

    def test_auc_matches_all_pairs_and_opposing_directions_are_retained(self):
        #### AUC includes half-credit ties and never flips a participant's direction to improve it.
        data = synthetic_features()
        eligible, comparisons = participant_comparisons(data, participant_coverage(data))
        for row in comparisons.itertuples(index=False):
            person = eligible.loc[eligible.id.eq(row.id)]
            positive = person.loc[person[protocol.TARGET].eq(1), row.feature].to_numpy()
            negative = person.loc[person[protocol.TARGET].eq(0), row.feature].to_numpy()
            expected = ((positive[:, None] > negative).sum() +
                        0.5 * (positive[:, None] == negative).sum()) / (len(positive) * len(negative))
            self.assertAlmostEqual(row.higher_value_auc, expected)
        constant = data.copy()
        constant[list(FEATURES)] = 5
        _, same = participant_comparisons(constant, participant_coverage(constant))
        self.assertTrue(same.higher_value_auc.eq(0.5).all())
        self.assertTrue(summary_table(same).zero_participants.eq(2).all())
        inverted = data.copy()
        inverted[protocol.TARGET] = 1 - inverted[protocol.TARGET]
        _, reverse = participant_comparisons(inverted, participant_coverage(inverted))
        np.testing.assert_allclose(reverse.mean_difference, -comparisons.mean_difference)
        np.testing.assert_allclose(reverse.higher_value_auc, 1 - comparisons.higher_value_auc)

    def test_bootstrap_preserves_person_level_effect_vectors(self):
        #### Independently reconstruct percentile limits after resampling participant vectors.
        data = synthetic_features()
        _, comparisons = participant_comparisons(data, participant_coverage(data))
        summary = summary_table(comparisons)
        columns = ["mean_difference", "median_difference", "higher_value_auc"]
        for feature in FEATURES:
            values = comparisons.loc[comparisons.feature.eq(feature)].sort_values("id")[columns].to_numpy()
            rng = np.random.default_rng(20261021)
            draws = rng.integers(0, len(values), size=(2000, len(values)))
            limits = np.quantile(values[draws].mean(axis=1), [0.025, 0.975], axis=0)
            row = summary.loc[summary.feature.eq(feature)].iloc[0]
            for index, column in enumerate(columns):
                np.testing.assert_allclose(limits[:, index],
                    [row[f"{column}_ci_lower"], row[f"{column}_ci_upper"]])

    def test_display_weights_and_centring_do_not_change_inputs(self):
        #### Each person has the same weight within each class despite differing day counts.
        data = synthetic_features()
        before = data.copy(deep=True)
        eligible, comparisons = participant_comparisons(data, participant_coverage(data))
        display = centred_distributions(eligible, comparisons)
        totals = display.groupby(["feature", protocol.TARGET, "id"]).display_weight.sum()
        np.testing.assert_allclose(totals, 0.5)
        for _, group in display.loc[display[protocol.TARGET].eq(0)].groupby("feature"):
            self.assertAlmostEqual(np.dot(group.display_value, group.display_weight), 0)
        pd.testing.assert_frame_equal(data, before)

    def test_invalid_values_keys_and_no_paired_participants(self):
        #### Reject missing/infinite predictors, unknown labels, duplicate keys and non-2022 data.
        data = synthetic_features()
        for field, value in [(next(iter(FEATURES)), np.nan), (next(iter(FEATURES)), np.inf),
                             (protocol.TARGET, 2), ("study_interval", 2024)]:
            bad = data.copy()
            bad.loc[0, field] = value
            with self.assertRaises(ValueError):
                validate_data(bad)
        with self.assertRaises(ValueError):
            validate_data(pd.concat([data, data.iloc[:1]]))
        single = data.loc[data.id.isin([3, 4])]
        with self.assertRaises(ValueError):
            participant_comparisons(single, participant_coverage(single))

    def test_frozen_cohort_comparison_and_file_protection(self):
        #### Verify source hashes and reject changed model predictors even after updating a source hash.
        data = synthetic_features()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed, preparation, frozen = [root / name for name in ("processed", "preparation", "frozen")]
            for folder in (processed, preparation, frozen):
                folder.mkdir()
            source, reference = processed / "observations.csv", frozen / "observations.csv"
            data.to_csv(source, index=False)
            data.to_csv(reference, index=False)
            manifest = {"feasible": True, "files": {str(source): file_sha256(source)}}
            (preparation / "manifest.json").write_text(json.dumps(manifest))
            (frozen / "freeze_manifest.json").write_text(json.dumps({"files": {"observations.csv": file_sha256(reference)}}))
            loaded, _ = load_data(processed, preparation, frozen)
            self.assertEqual(len(loaded), len(data))
            data.loc[0, next(iter(FEATURES))] += 1
            data.to_csv(source, index=False)
            with self.assertRaises(ValueError):
                load_data(processed, preparation, frozen)
            manifest["files"][str(source)] = file_sha256(source)
            (preparation / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaises(ValueError):
                load_data(processed, preparation, frozen)

    def test_report_and_plots_use_comparison_tables(self):
        #### Exercise both figure exports and report generation with the same verified tables.
        data = synthetic_features()
        coverage = participant_coverage(data)
        eligible, comparisons = participant_comparisons(data, coverage)
        summary = summary_table(comparisons)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            plot_distributions(centred_distributions(eligible, comparisons), root)
            plot_differences(comparisons, summary, root)
            write_report(root, coverage, summary)
            self.assertEqual(len(list(root.glob("*.png"))), 2)
            self.assertEqual(len(list(root.glob("*.pdf"))), 2)
            self.assertIn("2 participants, 11 positive days", (root / "REPORT.md").read_text())


if __name__ == "__main__":
    unittest.main()
