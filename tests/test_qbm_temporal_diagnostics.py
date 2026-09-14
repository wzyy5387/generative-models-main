import numpy as np

from GEFcom2014.forecast_quality.qbm_temporal_diagnostics import (
    summarize_edges,
    summarize_fields,
    summarize_rho,
)


def test_temporal_diagnostics_have_period_level_interpretations():
    rho = np.zeros((5, 3))
    rho[:, 1] = 0.4
    rho_rows = summarize_rho(rho, seed=2)
    assert [row["hour"] for row in rho_rows] == [1, 2, 3]
    assert rho_rows[1]["mean_rho"] == 0.4

    j = np.zeros((6, 6))
    j[0:2, 2:4] = 0.2
    j[2:4, 0:2] = 0.2
    edge_rows = summarize_edges(j, n_periods=3, seed=2)
    assert len(edge_rows) == 1
    assert edge_rows[0]["source_hour"] == 1
    assert edge_rows[0]["target_hour"] == 2
    assert edge_rows[0]["active_latent_edges"] == 4

    fields = np.arange(30, dtype=float).reshape(5, 6)
    field_rows = summarize_fields(fields, n_periods=3, seed=2)
    assert len(field_rows) == 3
    assert all(row["seed"] == 2 for row in field_rows)
