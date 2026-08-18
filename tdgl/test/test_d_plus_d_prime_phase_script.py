import math
from pathlib import Path

import numpy as np
import pytest
from my_scripts.simulations.simulate_d_plus_d_prime_phase_diagram import (
    FIXED_BACKGROUND_CONTROL,
    MAGNETIC_PERIODIC_BACKEND,
    MANIFEST_NAME,
    ZERO_FIELD_CELL_AREA,
    build_cell,
    cell_geometry,
    classify_state,
    configure_smoke_test,
    ensure_safe_overwrite_target,
    fixed_step_stability_limit,
    independent_common_vortex_seed,
    make_parser,
    make_solver_options,
    paper_high_field_transition,
    paper_low_field_transition,
    parse_grid,
    prepare_output_directory,
    row_completion_quality,
    run_scan,
    scan_configuration,
    transition_bracket,
    transition_confirmed,
    validate_fixed_step_stability,
    validate_scan_arguments,
)

import tdgl
from tdgl.magnetic_periodic.operators import MagneticPeriodicOperators


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
    assert args.grid_points == 24
    assert args.aspect_ratio == 1
    assert args.dt_init == args.dt_max == pytest.approx(0.002)
    assert not args.adaptive
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
    args = make_parser().parse_args(["--aspect-ratio", "nan"])
    with pytest.raises(ValueError, match="--aspect-ratio"):
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

    changed = {
        **configuration,
        "aspect_ratio": configuration["aspect_ratio"] + 1,
    }
    with pytest.raises(RuntimeError, match="aspect_ratio"):
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


def test_magnetic_periodic_cell_geometry_fixes_flux_area_and_aspect_ratio():
    length_x, length_y, flux_quanta = cell_geometry(0.4, 2.0)
    assert flux_quanta == 1
    assert length_x * length_y == pytest.approx(2 * math.pi / 0.4)
    assert length_x / length_y == pytest.approx(2)

    cell = build_cell(0.5, 0.4, grid_points=18, aspect_ratio=2.0)
    assert cell.shape == (18, 18)
    assert cell.flux_quanta == 1
    assert cell.dimensionless_area == pytest.approx(2 * math.pi / 0.4)
    assert cell.mean_induction == pytest.approx(0.4)
    assert cell.dimensionless_lengths[0] / cell.dimensionless_lengths[1] == (
        pytest.approx(2.0)
    )

    zero = build_cell(0.5, 0.0, grid_points=18, aspect_ratio=2.0)
    assert zero.flux_quanta == 0
    assert zero.mean_induction == pytest.approx(0)
    assert zero.dimensionless_area == pytest.approx(ZERO_FIELD_CELL_AREA)


def test_each_periodic_seed_is_a_fresh_common_vortex_mixed_state():
    cell = build_cell(0.5, 0.6, grid_points=12, aspect_ratio=1.0)
    d1, d_prime1 = independent_common_vortex_seed(cell)
    d2, d_prime2 = independent_common_vortex_seed(cell)
    assert not np.shares_memory(d1, d2)
    assert not np.shares_memory(d_prime1, d_prime2)
    assert d1 == pytest.approx(d2)
    assert d_prime1 == pytest.approx(d_prime2)
    valid = np.abs(d1) > 1e-14
    expected_ratio = -1j * math.sqrt((3 * 0.5 - 1) / (3 - 0.5))
    assert d_prime1[valid] / d1[valid] == pytest.approx(expected_ratio)
    operators = MagneticPeriodicOperators(cell)
    operators.set_vector_potential(np.zeros((2,) + cell.shape))
    assert operators.vortex_count(d1) == 1


@pytest.mark.parametrize("field", [0.6, 0.8])
def test_magnetic_eigenmode_seed_is_safe_for_mac_fixed_step(field, tmp_path):
    cell = build_cell(0.8, field, grid_points=24, aspect_ratio=1.0)
    d_order, d_prime_order = independent_common_vortex_seed(cell)
    args = make_parser().parse_args([])
    solver = tdgl.MagneticPeriodicSolver(
        cell,
        make_solver_options(args, tmp_path / f"seed-{field:g}.h5"),
        initial_psi_d=d_order,
        initial_psi_s=d_prime_order,
    )

    _, _, change = solver._order_parameter_trial(d_order, d_prime_order, args.dt_init)
    assert np.isfinite(change)
    assert change < 0.1


def test_fixed_step_limit_and_options_are_explicit(tmp_path):
    cell = build_cell(0.5, 1.0, grid_points=24, aspect_ratio=1.0)
    limit = fixed_step_stability_limit(cell)
    assert limit == pytest.approx(1 / (2 / cell.hx**2 + 2 / cell.hy**2))
    validate_fixed_step_stability(cell, dt=limit, reduced_field=1.0)
    with pytest.raises(ValueError, match="requires dt <="):
        validate_fixed_step_stability(cell, dt=1.01 * limit, reduced_field=1.0)

    args = make_parser().parse_args([])
    options = make_solver_options(args, tmp_path / "point.h5")
    assert options.adaptive is False
    assert options.include_screening is False
    assert options.terminal_psi is None
    assert options.dt_init == options.dt_max


def test_unstable_fixed_step_is_rejected_before_output_creation(tmp_path):
    output = tmp_path / "must_not_exist"
    args = make_parser().parse_args(
        [
            "--alphas",
            "0.5",
            "--fields",
            "1",
            "--grid-points",
            "64",
            "--dt-init",
            "0.002",
            "--dt-max",
            "0.002",
            "--output-directory",
            str(output),
            "--no-plots",
        ]
    )
    with pytest.raises(ValueError, match="unstable for b="):
        run_scan(args)
    assert not output.exists()


def test_manifest_records_magnetic_periodic_backend_and_seed_policy():
    args = make_parser().parse_args([])
    configuration = scan_configuration(
        args,
        np.asarray([0.5]),
        np.asarray([0.0, 0.6]),
        np.asarray([1e-3]),
    )
    assert configuration["backend"]["name"] == MAGNETIC_PERIODIC_BACKEND
    assert configuration["backend"]["field_control"] == FIXED_BACKGROUND_CONTROL
    assert configuration["backend"]["include_screening"] is False
    assert configuration["backend"]["solution_schema_version"] == 2
    assert configuration["seed_policy"] == "independent_common_vortex_mixed"
    assert configuration["adaptive"] is False


def test_removed_open_boundary_cli_is_rejected():
    parser = make_parser()
    for option in ("--width", "--max-edge-length", "--boundary-strip", "--smooth"):
        with pytest.raises(SystemExit):
            parser.parse_args([option, "1"])
