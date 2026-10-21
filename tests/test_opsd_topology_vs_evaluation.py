import csv
from types import SimpleNamespace

import numpy as np
import pytest

from GEFcom2014.forecast_quality import opsd_topology_vs_evaluation as topology
from GEFcom2014.forecast_quality.opsd_topology_vs_evaluation import (
    METRICS,
    _paired_block_summary,
    _read_existing_daily_metrics,
    _time_blocks,
    compute_daily_metrics,
)


def test_compute_daily_metrics_returns_daily_raw_metrics():
    targets = np.zeros((2, 24), dtype=np.float64)
    scenarios = np.zeros((48, 100), dtype=np.float64)
    metrics = compute_daily_metrics(scenarios, targets)

    assert set(metrics) == {
        "crps",
        "energy_score",
        "variogram_score",
        "ramp_crps",
        "picp90",
        "interval_width90",
        "bias",
    }
    for values in metrics.values():
        assert values.shape == (2,)
        assert np.isfinite(values).all()
    np.testing.assert_allclose(metrics["picp90"], 1.0)
    np.testing.assert_allclose(metrics["interval_width90"], 0.0)


def _paired_daily_fixture(days=90):
    base = np.arange(days, dtype=np.float64)
    return {
        arm: {
            seed: {metric: (base + seed if arm == "temporal" else np.zeros(days))
                   for metric in METRICS}
            for seed in (0, 1, 2)
        }
        for arm in ("temporal", "random")
    }


def test_topology_bootstrap_uses_common_daily_mean_and_weighted_tail_block():
    blocks = _time_blocks(np.arange(90, dtype=np.float64), 7)
    assert [block.size for block in blocks] == [7] * 12 + [6]
    fixture = _paired_daily_fixture()
    first = _paired_block_summary(fixture, repetitions=64, seed=123)
    second = _paired_block_summary(fixture, repetitions=64, seed=123)
    assert first == second
    row = first[0]
    np.testing.assert_allclose(row["mean_difference"], np.mean(np.arange(90) + 1.0))
    assert row["n_dates"] == 90
    assert row["n_date_blocks"] == 13
    assert row["date_block_lengths"] == "7;7;7;7;7;7;7;7;7;7;7;7;6"
    assert row["tail_block_days"] == 6
    assert row["bootstrap_method"] == "cross_seed_daily_mean_weighted_nonoverlap_blocks"


def test_topology_bootstrap_rejects_non_common_forecast_dates():
    fixture = _paired_daily_fixture()
    fixture["random"][2]["crps"] = np.zeros(89)
    with pytest.raises(ValueError, match="identical one-dimensional shapes"):
        _paired_block_summary(fixture, repetitions=8, seed=123)


def test_shuffled_daily_csv_has_identical_ordered_pairs_and_bootstrap(tmp_path):
    fixture = _paired_daily_fixture()
    fieldnames = ["arm", "seed", "date_index", *METRICS]
    rows = []
    for arm in ("temporal", "random"):
        for seed in (0, 1, 2):
            for day in range(90):
                rows.append({
                    "arm": arm,
                    "seed": seed,
                    "date_index": day,
                    **{metric: fixture[arm][seed][metric][day] for metric in METRICS},
                })
    shuffled = [rows[index] for index in np.random.default_rng(91).permutation(len(rows))]
    paths = [tmp_path / "ordered.csv", tmp_path / "shuffled.csv"]
    for path, values in zip(paths, (rows, shuffled)):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(values)

    _, ordered = _read_existing_daily_metrics(paths[0])
    _, shuffled_result = _read_existing_daily_metrics(paths[1])
    for arm in ("temporal", "random"):
        for seed in (0, 1, 2):
            for metric in METRICS:
                np.testing.assert_array_equal(
                    ordered[arm][seed][metric], shuffled_result[arm][seed][metric]
                )
    assert _paired_block_summary(ordered, repetitions=64, seed=123) == _paired_block_summary(
        shuffled_result, repetitions=64, seed=123
    )


@pytest.mark.parametrize("posthoc", [False, True], ids=["formal", "posthoc"])
@pytest.mark.parametrize("existing_kind", ["empty_directory", "results", "file"])
def test_existing_output_is_rejected_without_changes(tmp_path, monkeypatch, posthoc, existing_kind):
    source = tmp_path / "source"
    source.mkdir()
    for name in ("per_date_metrics.csv", "per_seed_metrics.csv", "arm_summary.csv"):
        (source / name).write_text("original source\n", encoding="utf-8")
    # Isolate the output boundary; no real data, model, or statistics are loaded.
    monkeypatch.setattr(topology, "_read_existing_daily_metrics",
                        lambda path: ([], _paired_daily_fixture()))

    output = tmp_path / "output"
    if existing_kind == "file":
        output.write_text("existing file\n", encoding="utf-8")
    else:
        output.mkdir()
        if existing_kind == "results":
            for name in ("per_date_metrics.csv", "paired_temporal_minus_random_7day.csv",
                         "provenance.json"):
                (output / name).write_text("preserve this result\n", encoding="utf-8")

    def snapshot():
        return {str(path.relative_to(tmp_path)): topology.sha256_file(path)
                for path in tmp_path.rglob("*") if path.is_file()}

    before = snapshot()

    def forbidden(*args, **kwargs):
        pytest.fail("Existing output must be rejected before evaluation or writes")

    monkeypatch.setattr(topology, "load_targets", forbidden)
    monkeypatch.setattr(topology, "_paired_block_summary", forbidden)
    args = SimpleNamespace(
        source_evaluation_dir=source if posthoc else None,
        output_dir=output,
        dataset_bundle=tmp_path / "unused.npz",
    )
    with pytest.raises(FileExistsError):
        topology.evaluate(args)
    assert snapshot() == before
