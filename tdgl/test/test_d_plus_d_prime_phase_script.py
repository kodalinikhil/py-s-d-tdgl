import math
from pathlib import Path

import numpy as np
import pytest

from my_scripts.simulations.simulate_d_plus_d_prime_phase_diagram import (
    MANIFEST_NAME,
    classify_state,
    configure_smoke_test,
    ensure_safe_overwrite_target,
    make_parser,
    paper_high_field_transition,
    paper_low_field_transition,
    parse_grid,
    prepare_output_directory,
    row_completion_quality,
    scan_configuration,
    transition_bracket,
    transition_confirmed,
    validate_scan_arguments,
)


def test_parse_grid_is_inclusive():
    assert parse_grid("0:0.3:0.1") == pytest.approx([0, 0.1, 0.2, 0.3])
    assert parse_grid("0.1,0.4,0.9") == pytest.approx([0.1, 0.4, 0.9])


@pytest.mark.parametrize("specification", ["nan,0.2", "0:inf:0.1", "0:1:nan"])
def test_parse_grid_rejects_nonfinite_values(specification):
    with pytest.raises(ValueError, match="finite"):
        parse_grid(specification)


def test_paper_transition_equations():
    alpha = 0.5
    low = paper_low_field_transition(alpha)
    assert low * math.log(1 / low) == pytest.approx((3 * alpha - 1) / 2)
    assert 0 < low < 1 / math.e
    assert paper_high_field_transition(1 / 3) == pytest.approx(1)
    assert paper_high_field_transition(2 / 3) == pytest.approx(0.5)
    assert paper_high_field_transition(1) == pytest.approx(1)


def test_transition_bracket_respects_sweep_order():
    upward = [
        {"reduced_field": 0.2, "bulk_max_abs_d_prime": 0.1},
        {"reduced_field": 0.3, "bulk_max_abs_d_prime": 0.02},
        {"reduced_field": 0.4, "bulk_max_abs_d_prime": 1e-5},
    ]
    bracket = transition_bracket(upward, threshold=1e-3)
    assert bracket["status"] == "bracketed"
    assert bracket["transition_field"] == pytest.approx(0.35)
    assert bracket["field_uncertainty"] == pytest.approx(0.05)

    downward = list(reversed(upward))
    assert transition_bracket(downward, threshold=1e-3) == bracket


def test_transition_confirmation_requires_converged_mixed_and_pure_points():
    rows = [
        {
            "bulk_max_abs_d_prime": 0.1,
            "equilibrium_reached": True,
        },
        {
            "bulk_max_abs_d_prime": 1e-5,
            "equilibrium_reached": True,
        },
        {
            "bulk_max_abs_d_prime": 1e-6,
            "equilibrium_reached": True,
        },
    ]
    assert transition_confirmed(rows, 1e-3, 2)
    rows[-1]["equilibrium_reached"] = False
    assert not transition_confirmed(rows, 1e-3, 2)


def test_unconverged_crossing_is_flagged():
    rows = [
        {
            "reduced_field": 0.6,
            "bulk_max_abs_d_prime": 0.1,
            "equilibrium_reached": True,
        },
        {
            "reduced_field": 0.7,
            "bulk_max_abs_d_prime": 1e-5,
            "equilibrium_reached": False,
        },
    ]
    bracket = transition_bracket(rows, threshold=1e-3)
    assert bracket["status"] == "bracket_contains_unconverged"


def test_state_classification_distinguishes_pure_d_from_normal():
    assert (
        classify_state(
            {"bulk_max_abs_d": 0.8, "bulk_max_abs_d_prime": 0.1},
            1e-3,
            1e-3,
        )
        == "mixed"
    )
    assert (
        classify_state(
            {"bulk_max_abs_d": 0.8, "bulk_max_abs_d_prime": 1e-5},
            1e-3,
            1e-3,
        )
        == "pure_d"
    )
    normal = {"bulk_max_abs_d": 1e-5, "bulk_max_abs_d_prime": 1e-5}
    assert classify_state(normal, 1e-3, 1e-3) == "normal"
    rows = [
        {"reduced_field": 0.8, "bulk_max_abs_d": 0.5, "bulk_max_abs_d_prime": 0.1},
        {"reduced_field": 1.0, **normal},
    ]
    assert transition_bracket(rows, 1e-3)["status"] == (
        "normal_state_reached_without_pure_d_crossing"
    )


def test_multiple_crossings_are_reported_not_silently_truncated():
    rows = [
        {
            "reduced_field": field,
            "bulk_max_abs_d": 0.8,
            "bulk_max_abs_d_prime": amplitude,
            "equilibrium_reached": True,
        }
        for field, amplitude in zip((0.1, 0.2, 0.3), (0.1, 1e-5, 0.1))
    ]
    bracket = transition_bracket(rows, threshold=1e-3)
    assert bracket["status"] == "multiple_crossings"
    assert bracket["crossing_count"] == 2
    assert bracket["crossing_brackets"] == "0.1:0.2;0.2:0.3"


def test_no_equilibrium_stop_is_explicit_and_not_failed():
    rows = [
        {
            "reduced_field": 0.2,
            "bulk_max_abs_d": 0.8,
            "bulk_max_abs_d_prime": 0.1,
            "equilibrium_reached": "",
            "equilibrium_status": "not_requested",
        },
        {
            "reduced_field": 0.3,
            "bulk_max_abs_d": 0.8,
            "bulk_max_abs_d_prime": 1e-5,
            "equilibrium_reached": "",
            "equilibrium_status": "not_requested",
        },
    ]
    assert row_completion_quality(rows[0]) == "unchecked"
    assert transition_bracket(rows, 1e-3)["status"] == (
        "bracketed_without_equilibrium_check"
    )


def test_early_stop_does_not_accept_normal_tail():
    rows = [
        {"bulk_max_abs_d": 0.8, "bulk_max_abs_d_prime": 0.1},
        {"bulk_max_abs_d": 1e-6, "bulk_max_abs_d_prime": 1e-6},
        {"bulk_max_abs_d": 1e-6, "bulk_max_abs_d_prime": 1e-6},
    ]
    assert not transition_confirmed(rows, 1e-3, 2, 1e-3)


def test_efficient_defaults_keep_1500_as_a_hard_cap():
    args = make_parser().parse_args([])
    assert args.solve_time == 1500
    assert args.equilibrium_tolerance == pytest.approx(1e-5)
    assert args.equilibrium_min_time == pytest.approx(20)
    assert args.save_every == 10000
    assert not args.down_sweep
    assert args.stop_after_pure_points == 2
    assert not args.resume


def test_smoke_preset_keeps_timing_options_valid():
    args = make_parser().parse_args(["--smoke-test"])
    configure_smoke_test(args)
    validate_scan_arguments(
        args,
        np.asarray([0.5]),
        np.asarray([0.0, 0.6]),
        np.asarray([1e-3]),
    )
    assert args.equilibrium_min_time == 0
    assert args.equilibrium_min_time <= args.solve_time


def test_numeric_cli_validation_precedes_output_changes():
    args = make_parser().parse_args(["--width", "nan"])
    with pytest.raises(ValueError, match="--width"):
        validate_scan_arguments(
            args,
            np.asarray([0.5]),
            np.asarray([0.0]),
            np.asarray([1e-3]),
        )


def test_overwrite_guard_rejects_broad_paths(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="protected"):
        ensure_safe_overwrite_target(Path.cwd())
    with pytest.raises(ValueError, match="protected"):
        ensure_safe_overwrite_target(Path.home())


def test_resume_manifest_accepts_fresh_directory_and_rejects_changes(tmp_path):
    args = make_parser().parse_args([])
    alphas = parse_grid(args.alphas)
    fields = np.unique(parse_grid(args.fields))
    thresholds = parse_grid(args.thresholds)
    configuration = scan_configuration(args, alphas, fields, thresholds)
    output = tmp_path / "nested" / "run"

    assert not prepare_output_directory(
        output, configuration, overwrite=False, resume=True
    )
    assert (output / MANIFEST_NAME).is_file()
    assert prepare_output_directory(output, configuration, overwrite=False, resume=True)

    changed = {**configuration, "width": configuration["width"] + 1}
    with pytest.raises(RuntimeError, match="width"):
        prepare_output_directory(output, changed, overwrite=False, resume=True)


def test_safe_leaf_output_can_be_overwritten(tmp_path):
    output = tmp_path / "scan" / "run"
    output.mkdir(parents=True)
    (output / "old.txt").write_text("old", encoding="utf-8")
    assert not prepare_output_directory(
        output, {"test": True}, overwrite=True, resume=False
    )
    assert not (output / "old.txt").exists()
    assert (output / MANIFEST_NAME).exists()
