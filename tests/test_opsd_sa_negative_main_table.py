import pytest

from GEFcom2014.forecast_quality.opsd_sa_negative_main_table import validate_complete_cells


def test_main_table_rejects_a_missing_date_instead_of_intersecting_cells():
    rows = []
    models = ("FA-BM-VAE", "Forecast-anchored Gaussian", "GMM-4", "Score-SDE", "J=0 ablation", "No anchor addition ablation (matched zero-mean init)")
    for model in models:
        for seed in (0, 1, 2):
            for variant in ("Raw", "Cal", "Cal+ECC"):
                for date in ("2019-01-01", "2019-01-02"):
                    if model == "GMM-4" and seed == 1 and variant == "Cal" and date.endswith("02"):
                        continue
                    rows.append({"model": model, "seed": str(seed), "variant": variant,
                                 "date": date, "n_scenarios": "100"})
    with pytest.raises(ValueError, match="must have 2 unique TEST dates"):
        validate_complete_cells(rows, expected_dates=2)
