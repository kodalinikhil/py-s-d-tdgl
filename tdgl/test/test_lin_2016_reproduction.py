from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial import cKDTree

from my_scripts.reproductions.reproduce_lin_maiti_chubukov_prb94_064519 import (
    MODEL_MAPPINGS,
    PRESETS,
    apply_overrides,
    build_parser,
    configuration_slug,
    defect_mask,
    defect_profile,
    make_device,
    paper_s_plus_id_model,
    paper_s_plus_is_model,
    paper_triangle_field,
    parse_figures,
    uniform_chiral_state,
)


def test_paper_coefficients_map_to_s_plus_is_model():
    model = paper_s_plus_is_model()

    assert (model.a1, model.a2) == (-1.0, -1.0)
    assert (model.b1, model.b2) == (1.0, 1.0)
    assert model.k2_over_k1 == pytest.approx(0.5)
    assert model.josephson_gamma == 0.0
    assert model.phase_gamma2 == pytest.approx(0.5)
    assert model.density_gamma3 == 0.0
    assert model.mixed_gradient_k12 == pytest.approx(0.5)
    assert model.beta_em == 1.0
    assert model.disorder_coupling1 == 1.0
    assert model.disorder_coupling2 == 1.0


def test_paper_coefficients_map_to_s_plus_id_model():
    model = paper_s_plus_id_model()

    assert model.eta_s == pytest.approx(2.0)
    assert model.eta_v == pytest.approx(-1.0)
    assert model.nu == pytest.approx(1.0)
    assert model.tau1 == pytest.approx(1.0)
    assert model.tau3 == 0.0
    assert model.tau4 == pytest.approx(0.5)
    assert model.beta_em == 1.0
    assert model.relaxation_s == 1.0
    assert model.nu_disorder_coupling == 1.0


@pytest.mark.parametrize("chirality", [-1, 1])
def test_uniform_chiral_state_is_stationary_for_both_models(chirality):
    psi1, psi2 = uniform_chiral_state(chirality)
    assert abs(psi1) == pytest.approx(np.sqrt(2.0))
    assert abs(psi2) == pytest.approx(np.sqrt(2.0))
    assert np.angle(psi2 * np.conj(psi1)) == pytest.approx(chirality * np.pi / 2)

    s_is = paper_s_plus_is_model()
    rho1, rho2 = abs(psi1) ** 2, abs(psi2) ** 2
    rhs1 = (
        -s_is.a1 * psi1
        - s_is.b1 * rho1 * psi1
        - 0.5 * s_is.density_gamma3 * rho2 * psi1
        - s_is.phase_gamma2 * np.conj(psi1) * psi2**2
        + s_is.josephson_gamma * psi2
    )
    rhs2 = (
        -s_is.a2 * psi2
        - s_is.b2 * rho2 * psi2
        - 0.5 * s_is.density_gamma3 * rho1 * psi2
        - s_is.phase_gamma2 * np.conj(psi2) * psi1**2
        + s_is.josephson_gamma * psi1
    )
    assert rhs1 == pytest.approx(0.0, abs=1e-14)
    assert rhs2 == pytest.approx(0.0, abs=1e-14)

    # SPlusDModel stores psi1=s and psi2=d.  Uniform fields have no gradient
    # contribution, so these are the bulk residuals used by the solver.
    s_id = paper_s_plus_id_model()
    rhs_d = (
        psi2
        - rho2 * psi2
        - 0.5 * s_id.tau3 * rho1 * psi2
        - s_id.tau4 * psi1**2 * np.conj(psi2)
    )
    rhs_s = (
        s_id.nu * psi1
        - s_id.tau1 * rho1 * psi1
        - 0.5 * s_id.tau3 * rho2 * psi1
        - s_id.tau4 * psi2**2 * np.conj(psi1)
    )
    assert rhs_d == pytest.approx(0.0, abs=1e-14)
    assert rhs_s == pytest.approx(0.0, abs=1e-14)


def test_uniform_chiral_state_rejects_invalid_chirality():
    with pytest.raises(ValueError, match="chirality"):
        uniform_chiral_state(0)


@pytest.mark.parametrize(
    ("model_key", "coordinate_scale", "field_scale"),
    [("s_plus_is", 1.0, 1.0), ("s_plus_id", np.sqrt(2.0), 2.0)],
)
def test_coordinate_and_field_conversion(model_key, coordinate_scale, field_scale):
    mapping = MODEL_MAPPINGS[model_key]
    assert mapping.coordinate_scale == pytest.approx(coordinate_scale)
    assert mapping.field_scale == pytest.approx(field_scale)

    solver_sites = coordinate_scale * np.array(
        [[0.0, 0.0], [3.0, 0.0], [0.0, 3.0], [3.0, 3.0]]
    )
    mesh = SimpleNamespace(
        sites=solver_sites,
        elements=np.array([[0, 1, 2], [1, 3, 2]]),
    )
    calls = []

    def local_magnetic_induction(**kwargs):
        calls.append(kwargs)
        return np.array([0.125, -0.25])

    solution = SimpleNamespace(
        device=SimpleNamespace(mesh=mesh),
        local_magnetic_induction=local_magnetic_induction,
    )
    centers, field = paper_triangle_field(solution, model_key)

    np.testing.assert_allclose(centers, [[1.0, 1.0], [2.0, 2.0]])
    assert field == pytest.approx(field_scale * np.array([0.125, -0.25]))
    assert calls == [{"units": "Bc2", "with_units": False}]


def test_square_and_angular_defect_masks_match_paper_shapes():
    square_points = np.array([[0.0, 0.0], [1.0, 1.0], [-1.0, -1.0], [1.001, 0.0]])
    assert np.array_equal(
        defect_mask(square_points, shape="square", r0=1.0),
        [True, True, True, False],
    )

    angular_points = np.array(
        [[0.0, 0.0], [0.9, 0.0], [0.0, 0.9], [0.5, 0.5], [1.0, 0.0]]
    )
    assert np.array_equal(
        defect_mask(angular_points, shape="angular_n2", r0=1.0),
        [True, True, True, False, False],
    )


def test_defect_profile_converts_solver_coordinates_and_strength():
    scale = np.sqrt(2.0)
    profile = defect_profile(
        strength=0.35,
        shape="square",
        r0=1.0,
        coordinate_scale=scale,
    )
    solver_points = scale * np.array([[0.5, 0.5], [1.0, -1.0], [1.01, 0.0]])

    assert profile(solver_points) == pytest.approx([0.65, 0.65, 1.0])


@pytest.mark.parametrize(
    "kwargs",
    [
        {"strength": -0.01, "shape": "square", "r0": 1, "coordinate_scale": 1},
        {"strength": 1.01, "shape": "square", "r0": 1, "coordinate_scale": 1},
        {"strength": 0.5, "shape": "square", "r0": 1, "coordinate_scale": 0},
    ],
)
def test_defect_profile_rejects_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        defect_profile(**kwargs)


def test_defect_mask_rejects_invalid_geometry():
    with pytest.raises(ValueError, match="shape"):
        defect_mask([[0, 0]], shape="circle", r0=1)
    with pytest.raises(ValueError, match=r"shape \(n, 2\)"):
        defect_mask(np.ones((2, 3)), shape="square", r0=1)
    with pytest.raises(ValueError, match="r0"):
        defect_mask([[0, 0]], shape="square", r0=0)


def test_parser_and_overrides_select_requested_configuration():
    parser = build_parser()
    args = parser.parse_args(
        [
            "--figures",
            "4,3",
            "--preset",
            "smoke",
            "--points",
            "11",
            "--side",
            "9",
            "--solve-time",
            "0.2",
            "--strengths",
            "0.1",
            "0.4",
            "--pattern-strength",
            "0.6",
        ]
    )
    original = PRESETS["smoke"]
    overridden = apply_overrides(original, args)

    assert parse_figures(args.figures) == ("3", "4")
    assert overridden.points == 11
    assert overridden.side_length == pytest.approx(9.0)
    assert overridden.solve_time == pytest.approx(0.2)
    assert overridden.scaling_strengths == (0.1, 0.4)
    assert overridden.pattern_strength == pytest.approx(0.6)
    assert PRESETS["smoke"] is original
    assert original.points == 9


@pytest.mark.parametrize(
    "arguments",
    [
        ["--solve-time", "0"],
        ["--strengths", "0"],
        ["--strengths", "1.01"],
        ["--pattern-strength", "0"],
        ["--pattern-strength", "1.01"],
    ],
)
def test_overrides_reject_invalid_numerics(arguments):
    args = build_parser().parse_args(["--preset", "smoke", *arguments])
    with pytest.raises(ValueError):
        apply_overrides(PRESETS["smoke"], args)


def test_figure_parser_rejects_unknown_or_empty_selection():
    with pytest.raises(ValueError, match="Unknown figure"):
        parse_figures(["2"])
    with pytest.raises(ValueError, match="Select Figure"):
        parse_figures([])


def test_configuration_slug_is_deterministic_and_separates_defect_radii():
    kwargs = dict(
        kappa=4.0,
        mesh_seed=2016,
        chirality=1,
        scaling_r0=1.5,
        pattern_r0=1.0,
    )
    slug = configuration_slug(PRESETS["smoke"], **kwargs)
    assert slug == configuration_slug(PRESETS["smoke"], **kwargs)
    assert slug != configuration_slug(
        PRESETS["smoke"], **{**kwargs, "pattern_r0": 1.25}
    )


@pytest.mark.parametrize("model_key", ["s_plus_is", "s_plus_id"])
def test_reproduction_mesh_is_d4_symmetric_with_positive_dual(model_key):
    device = make_device(
        model_key,
        side_length=8.0,
        points=7,
        kappa=4.0,
        mesh_seed=2016,
    )
    mesh = device.mesh
    tree = cKDTree(mesh.sites)

    for transformed in (
        np.column_stack((-mesh.sites[:, 1], mesh.sites[:, 0])),
        np.column_stack((-mesh.sites[:, 0], mesh.sites[:, 1])),
    ):
        distances, permutation = tree.query(transformed)
        assert np.max(distances) < 1e-12
        original_triangles = {
            tuple(triangle) for triangle in np.sort(mesh.elements, axis=1)
        }
        transformed_triangles = {
            tuple(triangle) for triangle in np.sort(permutation[mesh.elements], axis=1)
        }
        assert transformed_triangles == original_triangles

    assert np.all(np.isfinite(mesh.areas))
    assert np.all(mesh.areas > 0)
    dual_lengths = mesh.edge_mesh.dual_edge_lengths
    assert np.all(np.isfinite(dual_lengths))
    assert np.all(dual_lengths > 0)
