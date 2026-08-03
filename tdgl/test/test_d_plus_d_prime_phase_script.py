import math

import pytest

from my_scripts.simulate_d_plus_d_prime_phase_diagram import (
    paper_high_field_transition,
    paper_low_field_transition,
    parse_grid,
    transition_bracket,
)


def test_parse_grid_is_inclusive():
    assert parse_grid("0:0.3:0.1") == pytest.approx([0, 0.1, 0.2, 0.3])
    assert parse_grid("0.1,0.4,0.9") == pytest.approx([0.1, 0.4, 0.9])


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
