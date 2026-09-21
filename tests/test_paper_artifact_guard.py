import pytest

from GEFcom2014.forecast_quality.paper_artifact_guard import assert_paper_eligible, parse_bool


def test_synthetic_artifact_is_rejected_from_paper_aggregation():
    with pytest.raises(ValueError, match="synthetic"):
        assert_paper_eligible({"smoke_only": True, "synthetic_data": True}, "smoke.json")


def test_formal_artifact_is_eligible():
    assert assert_paper_eligible({"smoke_only": False, "synthetic_data": False,
                                  "eligible_for_paper": True,
                                  "artifact_role": "formal_complete"})["eligible_for_paper"] is True


def test_csv_boolean_strings_are_parsed_and_missing_eligibility_is_rejected():
    assert parse_bool("True") is True
    assert parse_bool("False") is False
    assert parse_bool(True) is True
    with pytest.raises(ValueError, match="eligible_for_paper"):
        assert_paper_eligible({"smoke_only": "False", "synthetic_data": "False"})


def test_acceptance_and_formal_task_artifacts_are_rejected():
    for role in ("acceptance", "formal_task"):
        with pytest.raises(ValueError, match="non-complete"):
            assert_paper_eligible({"smoke_only": "False", "synthetic_data": "False",
                                   "eligible_for_paper": "True", "artifact_role": role})
