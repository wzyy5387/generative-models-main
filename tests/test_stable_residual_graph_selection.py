from GEFcom2014.forecast_quality.select_stable_residual_graph import evaluate_gate


def _metrics(value):
    return {
        "crps": value,
        "energy_score": value,
        "variogram_score": value,
        "ramp_crps": value,
    }


def test_vs_gate_requires_both_datasets_and_seed_direction():
    cells = {
        track: {
            "j0": _metrics(1.0),
            "temporal_mi": _metrics(0.98),
            "candidate": _metrics(0.97),
        }
        for track in ("wind", "opsd-wind")
    }
    per_seed = {
        track: {
            seed: {
                "j0": _metrics(1.0),
                "temporal_mi": _metrics(0.98),
                "candidate": _metrics(0.97 if seed < 2 else 0.99),
            }
            for seed in range(3)
        }
        for track in cells
    }

    checks, eligible = evaluate_gate(cells, per_seed)

    assert eligible is True
    assert all(check["passed"] for check in checks)


def test_vs_gate_rejects_one_dataset_regression():
    cells = {
        track: {
            "j0": _metrics(1.0),
            "temporal_mi": _metrics(0.98),
            "candidate": _metrics(0.97),
        }
        for track in ("wind", "opsd-wind")
    }
    cells["opsd-wind"]["candidate"]["ramp_crps"] = 1.02
    per_seed = {
        track: {
            seed: {
                family: dict(metrics)
                for family, metrics in families.items()
            }
            for seed in range(3)
        }
        for track, families in cells.items()
    }

    _, eligible = evaluate_gate(cells, per_seed)

    assert eligible is False
