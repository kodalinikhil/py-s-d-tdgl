import csv
import math
from pathlib import Path

import pytest

from my_scripts.plots.plot_d_plus_d_prime_phase_scan import (
    TRANSITIONS_FILENAME,
    build_safe_transition_rows,
    default_output_directory,
    load_measurements,
    parse_thresholds,
)


FIELDS = (
    "alpha",
    "direction",
    "sequence_index",
    "reduced_field",
    "bulk_max_abs_d",
    "bulk_max_abs_d_prime",
    "bulk_mean_abs_d",
    "equilibrium_reached",
    "actual_solve_time",
    "accepted_steps",
    "output_file",
)


def write_measurements(path: Path, rows: list[dict], fields=FIELDS) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def measurement(**updates) -> dict:
    row = {
        "alpha": "0.8",
        "direction": "up",
        "sequence_index": "2",
        "reduced_field": "0.7",
        "bulk_max_abs_d": "0.8",
        "bulk_max_abs_d_prime": "0.2",
        "bulk_mean_abs_d": "0.5",
        "equilibrium_reached": "True",
        "actual_solve_time": "100",
        "accepted_steps": "1000",
        "output_file": "h5/original.h5",
    }
    row.update(updates)
    return row


def test_thresholds_are_configurable_and_validated():
    assert parse_thresholds("1e-4, 2e-3,1e-4") == pytest.approx((1e-4, 2e-3))
    with pytest.raises(Exception, match="finite and positive"):
        parse_thresholds("0,nan")


def test_loader_normalizes_blank_numeric_and_marks_unchecked(tmp_path):
    path = tmp_path / "measurements.csv"
    write_measurements(
        path,
        [measurement(bulk_max_abs_d_prime="", equilibrium_reached="")],
    )
    row = load_measurements([path])[0]
    assert math.isnan(row["bulk_max_abs_d_prime"])
    assert math.isnan(row["equilibrium_error"])
    assert row["equilibrium_reached"] is None
    assert row["_equilibrium_status"] == "unchecked"


@pytest.mark.parametrize("value", ["", "nan", "inf"])
def test_loader_rejects_nonfinite_key_fields(tmp_path, value):
    path = tmp_path / "measurements.csv"
    write_measurements(path, [measurement(alpha=value)])
    with pytest.raises(ValueError, match="required key field alpha must be finite"):
        load_measurements([path])


def test_loader_validates_schema(tmp_path):
    path = tmp_path / "measurements.csv"
    fields = tuple(field for field in FIELDS if field != "direction")
    row = measurement()
    row.pop("direction")
    write_measurements(path, [row], fields)
    with pytest.raises(ValueError, match="missing required columns: direction"):
        load_measurements([path])


def test_conflicting_duplicates_prefer_converged_continuation(tmp_path):
    original = tmp_path / "original.csv"
    continued = tmp_path / "continued.csv"
    write_measurements(
        original,
        [measurement(equilibrium_reached="False", actual_solve_time="1500")],
    )
    write_measurements(
        continued,
        [
            measurement(
                equilibrium_reached="True",
                actual_solve_time="2300",
                output_file="branch_checks/continued_b_0.7.h5",
            )
        ],
    )
    with pytest.warns(UserWarning, match="Conflicting duplicate measurement"):
        rows = load_measurements([original, continued])
    assert len(rows) == 1
    assert rows[0]["equilibrium_reached"] is True
    assert rows[0]["actual_solve_time"] == pytest.approx(2300)
    assert "continued" in rows[0]["output_file"]


def test_nonfinite_amplitude_cannot_become_a_transition(tmp_path):
    path = tmp_path / "measurements.csv"
    write_measurements(
        path,
        [
            measurement(
                sequence_index="0", reduced_field="0.6", bulk_max_abs_d_prime="0.1"
            ),
            measurement(
                sequence_index="1", reduced_field="0.7", bulk_max_abs_d_prime=""
            ),
        ],
    )
    transitions = build_safe_transition_rows(load_measurements([path]), (1e-3,))
    assert transitions[0]["status"] == "incomplete_data"
    assert math.isnan(transitions[0]["transition_field"])


def test_multi_input_default_uses_combined_directory(tmp_path):
    first = tmp_path / "alpha_0.7" / "measurements.csv"
    second = tmp_path / "alpha_0.8" / "measurements.csv"
    assert default_output_directory([first, second]) == tmp_path / "combined_phase_scan"
    assert TRANSITIONS_FILENAME != "transitions.csv"
