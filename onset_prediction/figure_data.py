#### Verify saved three-day forecasting results before drawing thesis figures.
#### Inputs are completed evaluation tables and their frozen cohorts, folds and protocol.
#### Check source hashes, paired keys, probability summaries and saved bootstrap intervals.
#### No models are fitted, no bootstrap draws are generated and no outcomes are changed.

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from onset_prediction.metrics import (INPUT_SETS, calibration_bins, summary_tables,
                                      validate_paired)


FAMILIES = ('logistic_regression', 'random_forest')
EVALUATION_TABLES = ('test_predictions', 'model_summary', 'paired_comparison',
                     'calibration_bins', 'bootstrap_metrics')
FROZEN_TABLES = ('observations', 'participant_folds', 'validation_fold_assignments',
                 'bootstrap_participants')


def digest(path):
    #### Calculate a file checksum without changing the source.
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path):
    #### Read a saved UTF-8 manifest or protocol.
    return json.loads(Path(path).read_text(encoding='utf-8'))


def verified_tables(directory, manifest, names):
    #### Read only tables whose bytes match their saved manifest entries.
    tables = {}
    for name in names:
        path = directory / f'{name}.csv'
        if manifest['files'].get(path.name) != digest(path):
            raise ValueError(f'Saved figure input changed: {path}')
        tables[name] = pd.read_csv(path)
    return tables


def assert_same(actual, expected, keys):
    #### Compare saved and reconstructed values, allowing only CSV rounding differences.
    actual = actual.sort_values(keys).reset_index(drop=True)
    expected = expected.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(actual[expected.columns], expected, check_dtype=False,
                                      check_exact=False, rtol=1e-10, atol=1e-12)
    except AssertionError as error:
        raise ValueError('Saved figure values disagree with their underlying records') from error


def validate_design(config):
    #### Prevent fixed schematic labels from being used with a different analysis design.
    required = {'version': '2026-11-03-onset-3d-v1', 'study_interval': 2022,
                'horizon_days': [1, 2, 3], 'prediction_cutoff': '14:00:00',
                'test_folds': 5, 'validation_folds': 3}
    if any(config.get(key) != value for key, value in required.items()):
        raise ValueError('The saved protocol does not match the figure design')
    if config['reference']['days'] != 14 or config['reference']['minimum_observed_days'] != 7:
        raise ValueError('Unexpected personal-reference window')
    if config['trend']['days'] != 7 or config['trend']['minimum_observed_days'] != 4:
        raise ValueError('Unexpected trend window')
    if config['bootstrap']['resamples'] != 2000:
        raise ValueError('Expected 2,000 saved participant bootstrap draws')


def validate_estimates(tables, family):
    #### Reconstruct probability summaries and intervals from the original saved records.
    predictions = tables['test_predictions']
    validate_paired(predictions, tables['observations'], tables['participant_folds'], family)
    if set(predictions.study_interval) != {2022}:
        raise ValueError('Only 2022 observations may enter thesis plots')
    bootstrap = tables['bootstrap_metrics']
    if not bootstrap.replicate.equals(tables['bootstrap_participants'].replicate):
        raise ValueError('Bootstrap replicate identifiers do not match the saved draws')
    expected_difference = bootstrap.combined_average_precision - bootstrap.wearable_average_precision
    if not np.allclose(expected_difference, bootstrap.combined_minus_wearable_average_precision,
                       rtol=1e-10, atol=1e-12, equal_nan=True):
        raise ValueError('Bootstrap AP differences are not paired')
    summary, paired = summary_tables(predictions, bootstrap, family)
    assert_same(tables['model_summary'], summary, ['input_set', 'metric'])
    assert_same(tables['paired_comparison'], paired, ['metric'])
    bins = [calibration_bins(predictions.loc[predictions.input_set.eq(name)]).assign(
        input_set=name, model_family=family) for name in INPUT_SETS]
    assert_same(tables['calibration_bins'], pd.concat(bins), ['input_set', 'bin'])


def load_family(evaluation, family):
    #### Verify a completed evaluation and its linked historical freeze, without refitting.
    frozen = evaluation.parent / 'frozen'
    status = read_json(evaluation / 'run_status.json')
    if status.get('status') != 'complete' or status.get('model_family') != family:
        raise ValueError('Expected a completed evaluation for the requested model family')
    manifest = read_json(evaluation / 'output_manifest.json')
    reference_path = evaluation / 'frozen_protocol_reference.json'
    if manifest['files'].get(reference_path.name) != digest(reference_path):
        raise ValueError('The evaluation freeze reference changed')
    if read_json(reference_path)['manifest_sha256'] != digest(frozen / 'freeze_manifest.json'):
        raise ValueError('The linked historical freeze changed')
    frozen_manifest = read_json(frozen / 'freeze_manifest.json')
    config_path = frozen / 'evaluation_protocol.json'
    if frozen_manifest['files'].get(config_path.name) != digest(config_path):
        raise ValueError('The saved protocol changed')
    config = read_json(config_path)
    validate_design(config)
    tables = verified_tables(evaluation, manifest, EVALUATION_TABLES)
    tables.update(verified_tables(frozen, frozen_manifest, FROZEN_TABLES))
    validate_estimates(tables, family)
    inputs = [evaluation / f'{name}.csv' for name in EVALUATION_TABLES]
    inputs += [frozen / f'{name}.csv' for name in FROZEN_TABLES]
    inputs += [reference_path, config_path, evaluation / 'output_manifest.json',
               evaluation / 'run_status.json', frozen / 'freeze_manifest.json']
    return tables, config, inputs


def load_results(logistic, forest):
    #### Require identical participants, outcomes, folds and bootstrap draws across families.
    loaded = [load_family(path, family) for path, family in zip((logistic, forest), FAMILIES)]
    for name in FROZEN_TABLES:
        pd.testing.assert_frame_equal(loaded[0][0][name], loaded[1][0][name])
    for field in ('version', 'horizon_days', 'reference', 'trend', 'test_folds',
                  'validation_folds', 'seed', 'validation_seeds', 'bootstrap'):
        if loaded[0][1][field] != loaded[1][1][field]:
            raise ValueError(f'Model families have different saved designs: {field}')
    tables = {family: item[0] for family, item in zip(FAMILIES, loaded)}
    return tables, loaded[0][1], [path for item in loaded for path in item[2]]
