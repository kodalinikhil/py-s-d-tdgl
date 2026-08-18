import h5py
import numpy as np
import pytest

from tdgl.device.layer import Layer
from tdgl.device.models import (
    DPlusDPrimeModel,
    SingleBandModel,
    SPlusDModel,
    SPlusSModel,
)
from tdgl.magnetic_periodic.cell import MagneticPeriodicCell
from tdgl.magnetic_periodic.operators import MagneticPeriodicOperators
from tdgl.magnetic_periodic.solution import MagneticPeriodicSolution
from tdgl.magnetic_periodic.solver import (
    MagneticPeriodicSolver,
    d_plus_d_prime_free_energy_density,
    magnetic_periodic_virial_applied_field,
    s_plus_d_free_energy_density,
    s_plus_d_virial_applied_field,
    solve_magnetic_periodic,
)
from tdgl.solver.options import SolverOptions, SparseSolver


def make_cell(*, flux_quanta=0, model=None, name="solver-cell"):
    if model is None:
        model = SPlusDModel(eta_s=2, eta_v=-0.5, beta_em=2)
    return MagneticPeriodicCell(
        name=name,
        layer=Layer(
            coherence_length=1,
            london_lambda=2,
            thickness=0.1,
            conductivity=1,
            model=model,
        ),
        lengths=(4, 3),
        shape=(4, 5),
        flux_quanta=flux_quanta,
        origin=(-2, -1.5),
    )


def make_options(path, **updates):
    values = dict(
        solve_time=2e-3,
        dt_init=1e-3,
        dt_max=1e-3,
        adaptive=False,
        include_screening=True,
        terminal_psi=None,
        save_every=1,
        output_file=str(path),
    )
    values.update(updates)
    return SolverOptions(**values)


def small_complex_fields(cell, seed=123):
    """Return deterministic, nonuniform fields that remain in the Euler regime."""
    rng = np.random.default_rng(seed)
    first = 0.2 + 0.03 * (
        rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    )
    second = -0.1j + 0.02 * (
        rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    )
    return first, second


def test_single_band_kwt_step_holds_fixed_background_field(tmp_path):
    model = SingleBandModel(gamma=1.7)
    cell = make_cell(flux_quanta=1, model=model)
    initial, _ = small_complex_fields(cell)
    vector_potential = (
        0.01 * np.arange(2 * cell.num_sites).reshape((2, *cell.shape)) / cell.num_sites
    )
    dt = 1e-4
    solver = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "single-band-fixed.h5",
            solve_time=dt,
            dt_init=dt,
            dt_max=dt,
            include_screening=False,
        ),
        initial_psi_d=initial,
        initial_vector_potential=vector_potential,
    )

    solver.operators.set_vector_potential(vector_potential)
    laplacian = solver.operators.laplacian(initial)
    density = np.abs(initial) ** 2
    z = 0.5 * model.gamma**2 * initial
    w = (
        z * density
        + initial
        + (
            dt
            / cell.layer.u
            * np.sqrt(1 + model.gamma**2 * density)
            * ((1 - density) * initial + laplacian)
        )
    )
    two_c_plus_one = 2 * np.real(w * np.conj(z)) + 1
    discriminant = two_c_plus_one**2 - 4 * np.abs(z) ** 2 * np.abs(w) ** 2
    next_density = 2 * np.abs(w) ** 2 / (two_c_plus_one + np.sqrt(discriminant))
    expected = w - z * next_density

    solution = solver.solve()

    assert solution.get_component("psi") == pytest.approx(expected, abs=2e-13)
    assert solution.vector_potential == pytest.approx(vector_potential, abs=0)
    assert solution.final_frame.normal_current == pytest.approx(0, abs=0)
    assert solver._magnetic_lu is None


def test_single_band_rejects_legacy_second_component_alias(tmp_path):
    cell = make_cell(model=SingleBandModel())
    with pytest.raises(ValueError, match="no initial_psi_s"):
        MagneticPeriodicSolver(
            cell,
            make_options(
                tmp_path / "single-band-extra-component.h5",
                include_screening=False,
            ),
            initial_psi_s=np.zeros(cell.shape, dtype=complex),
        )


def test_fixed_background_computes_current_only_for_saved_frames(tmp_path, monkeypatch):
    model = DPlusDPrimeModel(alpha=0.8)
    cell = make_cell(flux_quanta=1, model=model)
    initial_d, initial_d_prime = small_complex_fields(cell)
    solver = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "fixed-current-diagnostics.h5",
            solve_time=5e-4,
            dt_init=1e-4,
            dt_max=1e-4,
            include_screening=False,
            save_every=3,
        ),
        initial_psi_d=initial_d,
        initial_psi_s=initial_d_prime,
    )
    original = solver._normalized_supercurrent
    calls = 0

    def counting_current(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(solver, "_normalized_supercurrent", counting_current)
    solution = solver.solve()

    # Initial, step 3, and final step 5: no diagnostic current is evaluated on
    # the other integration steps.
    assert [frame.step for frame in solution.iter_frames()] == [0, 3, 5]
    assert calls == solution.num_frames == 3


def test_native_d_plus_d_prime_zero_field_state_is_stationary(tmp_path):
    alpha = 0.8
    model = DPlusDPrimeModel(alpha=alpha)
    cell = make_cell(model=model)
    d_amplitude = np.sqrt(3 * (3 - alpha) / 8)
    d_prime_amplitude = np.sqrt(3 * (3 * alpha - 1) / 8)
    initial_d = np.full(cell.shape, d_amplitude, dtype=complex)
    initial_d_prime = np.full(cell.shape, -1j * d_prime_amplitude, dtype=complex)

    solution = solve_magnetic_periodic(
        cell,
        make_options(
            tmp_path / "d-plus-d-prime-stationary.h5",
            include_screening=False,
        ),
        initial_psi_d=initial_d,
        initial_psi_s=initial_d_prime,
    )

    assert solution.get_component("d") == pytest.approx(initial_d, abs=2e-13)
    assert solution.get_component("d_prime") == pytest.approx(
        initial_d_prime, abs=2e-13
    )
    assert solution.vector_potential == pytest.approx(0, abs=0)


def test_native_d_plus_d_prime_one_step_uses_fixed_uniform_induction(tmp_path):
    model = DPlusDPrimeModel(
        alpha=0.62,
        relaxation_d=1.7,
        relaxation_d_prime=2.3,
        zeeman_coupling=0.18,
    )
    cell = make_cell(flux_quanta=1, model=model)
    initial_d, initial_d_prime = small_complex_fields(cell, seed=211)
    zeros = np.zeros((2, *cell.shape))
    dt = 1e-4
    solver = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "d-plus-d-prime-one-step.h5",
            solve_time=dt,
            dt_init=dt,
            dt_max=dt,
            include_screening=False,
        ),
        initial_psi_d=initial_d,
        initial_psi_s=initial_d_prime,
        initial_vector_potential=zeros,
    )

    induction = solver.operators.induction(zeros)
    lap_d = solver.operators.laplacian(initial_d, zeros)
    lap_d_prime = solver.operators.laplacian(initial_d_prime, zeros)
    abs_d = np.abs(initial_d) ** 2
    abs_d_prime = np.abs(initial_d_prime) ** 2
    rhs_d = (
        lap_d
        + initial_d
        - abs_d * initial_d
        - (2 / 3) * abs_d_prime * initial_d
        - (1 / 3) * initial_d_prime**2 * np.conj(initial_d)
        + 1j * model.zeeman_coupling * induction * initial_d_prime
    )
    rhs_d_prime = (
        lap_d_prime
        + model.alpha * initial_d_prime
        - abs_d_prime * initial_d_prime
        - (2 / 3) * abs_d * initial_d_prime
        - (1 / 3) * initial_d**2 * np.conj(initial_d_prime)
        - 1j * model.zeeman_coupling * induction * initial_d
    )

    solution = solver.solve()

    assert solution.get_component("d") == pytest.approx(
        initial_d + dt * rhs_d / model.relaxation_d, abs=2e-13
    )
    assert solution.get_component("d_prime") == pytest.approx(
        initial_d_prime + dt * rhs_d_prime / model.relaxation_d_prime,
        abs=2e-13,
    )
    assert solution.vector_potential == pytest.approx(zeros, abs=0)
    assert solution.induction == pytest.approx(
        np.full(cell.shape, cell.mean_induction), abs=2e-13
    )
    assert np.isfinite(solution.free_energy_density())


def test_d_plus_d_prime_fixed_field_free_energy_is_gauge_invariant():
    model = DPlusDPrimeModel(alpha=0.73, zeeman_coupling=-0.2)
    cell = make_cell(flux_quanta=1, model=model)
    operators = MagneticPeriodicOperators(cell)
    d_order, d_prime_order = small_complex_fields(cell, seed=307)
    vector_potential = np.zeros((2, *cell.shape))
    j, i = np.indices(cell.shape)
    chi = 0.17 * np.sin(2 * np.pi * i / cell.nx) * np.cos(2 * np.pi * j / cell.ny)
    phase = np.exp(1j * chi)
    transformed_a = vector_potential + operators.scalar_gradient(chi)

    energy = d_plus_d_prime_free_energy_density(
        cell, d_order, d_prime_order, vector_potential
    )
    transformed_energy = d_plus_d_prime_free_energy_density(
        cell, phase * d_order, phase * d_prime_order, transformed_a
    )

    assert operators.induction(transformed_a) == pytest.approx(
        np.full(cell.shape, cell.mean_induction), abs=2e-13
    )
    assert transformed_energy == pytest.approx(energy, abs=2e-12)


def test_s_plus_s_coupled_one_step_and_current(tmp_path):
    model = SPlusSModel(
        a1=-0.7,
        a2=-0.3,
        b1=1.1,
        b2=0.9,
        k2_over_k1=1.3,
        josephson_gamma=0.2,
        relaxation1=1.7,
        relaxation2=2.1,
        em_coupling=1.0,
        phase_gamma2=0.07,
        density_gamma3=0.3,
        mixed_gradient_k12=0.2,
        beta_em=1.6,
    )
    cell = make_cell(flux_quanta=1, model=model)
    initial1, initial2 = small_complex_fields(cell, seed=409)
    zeros = np.zeros((2, *cell.shape))
    dt = 1e-4
    solver = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "s-plus-s-one-step.h5",
            solve_time=dt,
            dt_init=dt,
            dt_max=dt,
            include_screening=False,
        ),
        initial_psi_d=initial1,
        initial_psi_s=initial2,
        initial_vector_potential=zeros,
    )

    lap1 = solver.operators.laplacian(initial1, zeros)
    lap2 = solver.operators.laplacian(initial2, zeros)
    rho1 = np.abs(initial1) ** 2
    rho2 = np.abs(initial2) ** 2
    rhs1 = (
        lap1
        + model.mixed_gradient_k12 * lap2
        - model.a1 * initial1
        - model.b1 * rho1 * initial1
        - 0.5 * model.density_gamma3 * rho2 * initial1
        - model.phase_gamma2 * np.conj(initial1) * initial2**2
        + model.josephson_gamma * initial2
    )
    rhs2 = (
        model.k2_over_k1 * lap2
        + model.mixed_gradient_k12 * lap1
        - model.a2 * initial2
        - model.b2 * rho2 * initial2
        - 0.5 * model.density_gamma3 * rho1 * initial2
        - model.phase_gamma2 * np.conj(initial2) * initial1**2
        + model.josephson_gamma * initial1
    )
    expected1 = initial1 + dt * rhs1 / model.relaxation1
    expected2 = initial2 + dt * rhs2 / model.relaxation2

    solution = solver.solve()
    expected_current = solver.operators.isotropic_two_component_supercurrent(
        expected1,
        expected2,
        k2=model.k2_over_k1,
        mixed_gradient=model.mixed_gradient_k12,
        vector_potential=zeros,
    ) / (model.em_coupling * model.beta_em)

    assert solution.get_component("s1") == pytest.approx(expected1, abs=2e-13)
    assert solution.get_component("s2") == pytest.approx(expected2, abs=2e-13)
    assert solution.final_frame.supercurrent == pytest.approx(
        expected_current, abs=2e-13
    )
    assert solution.vector_potential == pytest.approx(zeros, abs=0)


def test_s_plus_d_can_freeze_or_evolve_periodic_vector_potential(tmp_path):
    cell = make_cell(flux_quanta=0)
    initial_d, initial_s = small_complex_fields(cell, seed=503)
    zeros = np.zeros((2, *cell.shape))
    dt = 1e-4
    fixed = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "s-plus-d-fixed.h5",
            solve_time=dt,
            dt_init=dt,
            dt_max=dt,
            include_screening=False,
        ),
        initial_psi_d=initial_d,
        initial_psi_s=initial_s,
        initial_vector_potential=zeros,
    )
    screened = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "s-plus-d-screened.h5",
            solve_time=dt,
            dt_init=dt,
            dt_max=dt,
            include_screening=True,
        ),
        initial_psi_d=initial_d,
        initial_psi_s=initial_s,
        initial_vector_potential=zeros,
    )

    fixed_solution = fixed.solve()
    screened_solution = screened.solve()

    assert fixed_solution.vector_potential == pytest.approx(zeros, abs=0)
    assert fixed_solution.final_frame.normal_current == pytest.approx(0, abs=0)
    assert fixed._magnetic_lu is None
    assert not np.allclose(
        screened_solution.vector_potential, zeros, atol=1e-14, rtol=0
    )
    assert screened._magnetic_lu is not None
    assert screened_solution.psi_d == pytest.approx(fixed_solution.psi_d, abs=2e-13)
    assert screened_solution.psi_s == pytest.approx(fixed_solution.psi_s, abs=2e-13)


@pytest.mark.parametrize(
    "model",
    [SingleBandModel(), DPlusDPrimeModel(alpha=0.7, zeeman_coupling=0.2)],
)
def test_evolving_screening_is_restricted_to_supported_multicomponent_models(
    tmp_path, model
):
    with pytest.raises(ValueError, match="include_screening=False"):
        MagneticPeriodicSolver(
            make_cell(model=model),
            make_options(tmp_path / f"screening-{type(model).__name__}.h5"),
        )


@pytest.mark.parametrize("model", [SPlusDModel(), SPlusSModel()])
def test_evolving_screening_accepts_s_plus_d_and_s_plus_s(tmp_path, model):
    solver = MagneticPeriodicSolver(
        make_cell(model=model),
        make_options(tmp_path / f"screening-{type(model).__name__}.h5"),
    )
    assert solver.options.include_screening is True


def test_normal_state_drive_has_expected_electric_field(tmp_path):
    cell = make_cell()
    drive = np.array([0.08, -0.03])
    solution = solve_magnetic_periodic(
        cell,
        make_options(
            tmp_path / "normal-state.h5",
            s_plus_d_drive_current_x=drive[0],
            s_plus_d_drive_current_y=drive[1],
        ),
        initial_psi_d=np.zeros(cell.shape, dtype=complex),
    )

    assert solution.num_frames == 3
    assert solution.final_time == pytest.approx(2e-3)
    assert solution.final_step == 2
    assert solution.electric_field == pytest.approx(
        drive / cell.layer.model.beta_em, abs=2e-13
    )
    assert solution.final_frame.normal_current.mean(axis=(1, 2)) == pytest.approx(
        drive / cell.layer.model.beta_em, abs=2e-13
    )
    assert solution.final_frame.supercurrent == pytest.approx(0, abs=2e-13)
    assert solution.electric_block_delta() == pytest.approx(0, abs=2e-13)
    assert solution.state["vorticity_defined"] is False


def test_skip_stage_resets_measurement_clock_and_preserves_unwrapped_drive(tmp_path):
    cell = make_cell()
    drive = 0.08
    skip_time = 3e-3
    solve_time = 2e-3
    solution = solve_magnetic_periodic(
        cell,
        make_options(
            tmp_path / "normal-state-skip.h5",
            solve_time=solve_time,
            skip_time=skip_time,
            s_plus_d_drive_current_x=drive,
        ),
        initial_psi_d=np.zeros(cell.shape, dtype=complex),
    )

    beta = cell.layer.model.beta_em
    assert solution.times == pytest.approx([0, 1e-3, 2e-3])
    assert np.mean(solution.frame(0).vector_potential[0]) == pytest.approx(
        -skip_time * drive / beta
    )
    assert np.mean(solution.vector_potential[0]) == pytest.approx(
        -(skip_time + solve_time) * drive / beta
    )
    assert solution.electric_field == pytest.approx([drive / beta, 0])


def test_adaptive_retry_replaces_the_cached_factorization(tmp_path):
    cell = make_cell()
    solver = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "adaptive.h5",
            solve_time=2e-4,
            dt_init=1e-3,
            dt_max=1e-3,
            adaptive=True,
        ),
        initial_psi_d=np.full(cell.shape, 10 + 0j),
    )
    solution = solver.solve()

    assert solution.final_time == pytest.approx(2e-4)
    assert np.all(np.isfinite(solution.psi_d))
    assert solver._magnetic_lu is not None
    assert solver._magnetic_lu_dt is not None
    assert 0 < solver._magnetic_lu_dt < 2e-4


def test_adaptive_change_history_is_bounded(tmp_path):
    cell = make_cell()
    solver = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "bounded-adaptive-history.h5",
            adaptive=True,
            adaptive_window=3,
        ),
    )
    for _ in range(20):
        solver._update_adaptive_step(1e-3, 1e-3)
    assert len(solver._change_history) == solver.options.adaptive_window == 3


def test_implicit_maxwell_step_satisfies_its_linear_residual(tmp_path):
    cell = make_cell(flux_quanta=1)
    solver = MagneticPeriodicSolver(
        cell,
        make_options(
            tmp_path / "maxwell-residual.h5",
            s_plus_d_drive_current_x=0.03,
            s_plus_d_drive_current_y=-0.02,
        ),
    )
    rng = np.random.default_rng(19)
    psi_d = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    psi_s = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    vector_potential = 0.05 * rng.normal(size=(2, *cell.shape))
    dt = 3e-4
    solver.operators.set_vector_potential(vector_potential)

    new_vector_potential, _ = solver._advance_vector_potential(
        psi_d, psi_s, vector_potential, dt
    )
    current = solver.operators.s_plus_d_supercurrent(
        psi_d,
        psi_s,
        eta_s=solver.model.eta_s,
        eta_v=solver.model.eta_v,
    )
    rate = dt * cell.kappa**2 / solver.model.beta_em
    lhs = new_vector_potential.ravel() + rate * (
        solver.operators.magnetic_diffusion @ new_vector_potential.ravel()
    )
    rhs = (
        vector_potential.ravel()
        + (dt / solver.model.beta_em) * (current - solver.drive).ravel()
    )
    assert lhs == pytest.approx(rhs, abs=2e-12)


@pytest.mark.parametrize("bond", [(0, 1, 4), (1, 3, 2)])
def test_mixed_gradient_current_is_energy_derivative(bond):
    cell = make_cell(flux_quanta=1)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(23)
    psi_d = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    psi_s = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    vector_potential = 0.04 * rng.normal(size=(2, *cell.shape))
    current = operators.s_plus_d_supercurrent(
        psi_d,
        psi_s,
        eta_s=cell.layer.model.eta_s,
        eta_v=cell.layer.model.eta_v,
        vector_potential=vector_potential,
    )

    delta = 1e-7
    plus = vector_potential.copy()
    minus = vector_potential.copy()
    plus[bond] += delta
    minus[bond] -= delta
    derivative = (
        s_plus_d_free_energy_density(cell, psi_d, psi_s, plus, include_magnetic=False)
        - s_plus_d_free_energy_density(
            cell, psi_d, psi_s, minus, include_magnetic=False
        )
    ) / (2 * delta)
    assert derivative == pytest.approx(
        -2 * current[bond] / cell.num_sites, rel=2e-7, abs=2e-8
    )


def test_zero_site_vortex_diagnostic_does_not_abort_fixed_flux(tmp_path):
    cell = make_cell(flux_quanta=1)
    solution = solve_magnetic_periodic(
        cell,
        make_options(tmp_path / "one-flux-zero-state.h5"),
        initial_psi_d=np.zeros(cell.shape, dtype=complex),
    )

    assert solution.vortex_count == 1
    assert solution.state["vortex_count"] == 1
    assert solution.state["vorticity_defined"] is False
    assert np.mean(solution.induction) == pytest.approx(cell.mean_induction, abs=2e-13)
    assert solution.induction_statistics() == pytest.approx(
        (cell.mean_induction, 0), abs=2e-13
    )
    assert solution.free_energy_density() == pytest.approx(
        (cell.layer.london_lambda / cell.layer.coherence_length) ** 2
        * cell.mean_induction**2
    )
    assert solution.virial_applied_field() == pytest.approx(
        cell.mean_induction, abs=2e-13
    )


def test_nonzero_order_parameter_evolves_in_one_flux_sector(tmp_path):
    cell = make_cell(flux_quanta=1)
    initial_d = np.full(cell.shape, 0.4 + 0.1j)
    solution = solve_magnetic_periodic(
        cell,
        make_options(
            tmp_path / "one-flux-nonzero.h5",
            solve_time=2e-4,
            dt_init=1e-4,
            dt_max=1e-4,
        ),
        initial_psi_d=initial_d,
    )

    assert np.all(np.isfinite(solution.psi_d))
    assert not np.allclose(solution.psi_d, initial_d)
    assert solution.vortex_count == 1
    assert solution.state["vorticity_defined"] is True
    assert np.mean(solution.induction) == pytest.approx(cell.mean_induction, abs=2e-13)


def test_equilibrium_early_stop_uses_coupled_stationary_state(tmp_path):
    cell = make_cell()
    options = make_options(
        tmp_path / "equilibrium.h5",
        solve_time=2e-2,
        equilibrium_tolerance=1e-12,
        equilibrium_window=2,
        equilibrium_min_time=0,
        save_every=50,
    )
    solution = solve_magnetic_periodic(
        cell,
        options,
        initial_psi_d=np.zeros(cell.shape, dtype=complex),
    )

    assert solution.state["equilibrium_reached"] is True
    assert solution.state["equilibrium_error"] == pytest.approx(0)
    assert solution.final_step == 2
    assert solution.final_time == pytest.approx(2e-3)
    assert solution.final_time < options.solve_time


def test_virial_applied_field_rejects_zero_flux():
    cell = make_cell(flux_quanta=0)
    with pytest.raises(ValueError, match="zero mean flux"):
        s_plus_d_virial_applied_field(
            cell,
            np.ones(cell.shape, dtype=complex),
            np.zeros(cell.shape, dtype=complex),
            np.zeros((2, *cell.shape)),
        )


def test_virial_applied_field_rejects_spatial_disorder():
    cell = make_cell(flux_quanta=1)
    epsilon = np.ones(cell.shape)
    epsilon[0, 0] = 0.9
    with pytest.raises(ValueError, match="homogeneous coefficients"):
        s_plus_d_virial_applied_field(
            cell,
            np.ones(cell.shape, dtype=complex),
            np.zeros(cell.shape, dtype=complex),
            np.zeros((2, *cell.shape)),
            epsilon,
        )


@pytest.mark.parametrize(
    "model",
    [
        SingleBandModel(),
        SPlusDModel(),
        DPlusDPrimeModel(),
        SPlusSModel(),
    ],
)
def test_model_generic_virial_reduces_to_mean_field_in_normal_state(model):
    cell = make_cell(flux_quanta=1, model=model)
    zero_site = np.zeros(cell.shape, dtype=complex)
    zero_links = np.zeros((2, *cell.shape))
    assert magnetic_periodic_virial_applied_field(
        cell, zero_site, zero_site, zero_links
    ) == pytest.approx(cell.mean_induction, abs=2e-13)


def test_model_generic_virial_rejects_orbital_zeeman_term():
    cell = make_cell(
        flux_quanta=1,
        model=DPlusDPrimeModel(zeeman_coupling=0.2),
    )
    zero_site = np.zeros(cell.shape, dtype=complex)
    with pytest.raises(ValueError, match="orbital-Zeeman"):
        magnetic_periodic_virial_applied_field(
            cell,
            zero_site,
            zero_site,
            np.zeros((2, *cell.shape)),
        )


def test_free_energy_and_virial_field_are_gauge_invariant():
    cell = make_cell(flux_quanta=1)
    operators = MagneticPeriodicOperators(cell)
    rng = np.random.default_rng(31)
    psi_d = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    psi_s = rng.normal(size=cell.shape) + 1j * rng.normal(size=cell.shape)
    vector_potential = 0.1 * rng.normal(size=(2, *cell.shape))
    j, i = np.indices(cell.shape)
    chi = 0.2 * np.sin(2 * np.pi * i / cell.nx) * np.cos(2 * np.pi * j / cell.ny)
    phase = np.exp(1j * chi)
    transformed_a = vector_potential + operators.scalar_gradient(chi)

    energy = s_plus_d_free_energy_density(cell, psi_d, psi_s, vector_potential)
    transformed_energy = s_plus_d_free_energy_density(
        cell, phase * psi_d, phase * psi_s, transformed_a
    )
    field = s_plus_d_virial_applied_field(cell, psi_d, psi_s, vector_potential)
    transformed_field = s_plus_d_virial_applied_field(
        cell, phase * psi_d, phase * psi_s, transformed_a
    )
    assert transformed_energy == pytest.approx(energy, abs=2e-12)
    assert transformed_field == pytest.approx(field, abs=2e-12)


def test_uniform_zero_field_free_energy_and_local_dynamics(tmp_path):
    model = SPlusDModel(
        eta_s=2,
        eta_v=0,
        nu=0.4,
        tau1=1.2,
        beta_em=1,
        relaxation_s=4,
    )
    cell = make_cell(model=model)
    zeros = np.zeros((2, *cell.shape))
    assert s_plus_d_free_energy_density(
        cell,
        np.ones(cell.shape, dtype=complex),
        np.zeros(cell.shape, dtype=complex),
        zeros,
    ) == pytest.approx(-0.5)

    d_initial = np.zeros(cell.shape, dtype=complex)
    s_initial = np.full(cell.shape, 0.5 + 0j)
    solution = solve_magnetic_periodic(
        cell,
        make_options(tmp_path / "uniform-local.h5", solve_time=1e-3, save_every=1),
        initial_psi_d=d_initial,
        initial_psi_s=s_initial,
    )
    expected_s = 0.5 + 1e-3 * (0.4 * 0.5 - 1.2 * 0.5**3) / (2 * 4)
    assert solution.psi_d == pytest.approx(0, abs=2e-13)
    assert solution.psi_s == pytest.approx(expected_s, abs=2e-13)


def test_hdf5_roundtrip_and_incomplete_checkpoint_rejection(tmp_path):
    cell = make_cell()
    solution = solve_magnetic_periodic(
        cell,
        make_options(tmp_path / "complete.h5"),
        initial_psi_d=np.zeros(cell.shape, dtype=complex),
    )
    loaded = MagneticPeriodicSolution.from_hdf5(solution.path)

    assert loaded.cell == cell
    assert loaded.options == solution.options
    assert loaded.final_frame.psi_d == pytest.approx(solution.psi_d)
    assert loaded.final_frame.vector_potential == pytest.approx(
        solution.vector_potential
    )
    with h5py.File(solution.path, "r+") as h5file:
        assert bool(h5file.attrs["complete"])
        assert h5file.attrs["final_time"] == pytest.approx(solution.final_time)
        assert h5file.attrs["final_step"] == solution.final_step
        h5file.attrs["complete"] = False
    with pytest.raises(IOError, match="incomplete"):
        MagneticPeriodicSolution.from_hdf5(solution.path)


@pytest.mark.parametrize(
    "corruption, message",
    [
        ("backend", "Expected backend"),
        ("schema", "Unsupported magnetic-periodic HDF5 schema"),
        ("shape", "dataset 'psi1' has shape"),
        ("nonfinite", "dataset 'vector_potential' is non-finite"),
        ("final_time", "final metadata does not match"),
    ],
)
def test_corrupt_checkpoint_is_rejected(tmp_path, corruption, message):
    path = tmp_path / f"corrupt-{corruption}.h5"
    solution = solve_magnetic_periodic(
        make_cell(),
        make_options(path, solve_time=1e-3),
        initial_psi_d=np.zeros((4, 5), dtype=complex),
    )
    with h5py.File(solution.path, "r+") as h5file:
        if corruption == "backend":
            h5file.attrs["backend"] = "open_mesh"
        elif corruption == "schema":
            h5file.attrs["schema_version"] += 1
        elif corruption == "shape":
            final = h5file["data"][str(len(h5file["data"]) - 1)]
            del final["psi1"]
            final["psi1"] = np.zeros((2, 2), dtype=complex)
        elif corruption == "nonfinite":
            h5file["data"]["0"]["vector_potential"][0, 0, 0] = np.nan
        elif corruption == "final_time":
            h5file.attrs["final_time"] += 1
    with pytest.raises(IOError, match=message):
        MagneticPeriodicSolution.from_hdf5(solution.path)


def test_seed_requires_the_identical_magnetic_periodic_cell(tmp_path):
    cell = make_cell()
    seed = solve_magnetic_periodic(
        cell,
        make_options(tmp_path / "seed.h5", solve_time=1e-3),
        initial_psi_d=np.zeros(cell.shape, dtype=complex),
    )
    continuation = MagneticPeriodicSolver(
        cell,
        make_options(tmp_path / "continued.h5", solve_time=1e-3),
        seed_solution=seed,
    )
    assert continuation.psi_d == pytest.approx(seed.psi_d)
    assert continuation.vector_potential == pytest.approx(seed.vector_potential)

    different_sector = make_cell(flux_quanta=1)
    with pytest.raises(ValueError, match="exactly match"):
        MagneticPeriodicSolver(
            different_sector,
            make_options(tmp_path / "wrong-sector.h5", solve_time=1e-3),
            seed_solution=seed,
        )
    with pytest.raises(TypeError, match="MagneticPeriodicSolution"):
        MagneticPeriodicSolver(
            cell,
            make_options(tmp_path / "wrong-backend.h5", solve_time=1e-3),
            seed_solution=object(),
        )


@pytest.mark.parametrize(
    "updates, message",
    [
        ({"terminal_psi": 0}, "no terminals"),
        ({"monitor": True}, "unstructured mesh"),
    ],
)
def test_backend_rejects_open_mesh_only_options(tmp_path, updates, message):
    with pytest.raises(ValueError, match=message):
        MagneticPeriodicSolver(
            make_cell(),
            make_options(tmp_path / "unsupported.h5", **updates),
        )


def test_backend_rejects_unimplemented_sparse_solver(tmp_path):
    with pytest.raises(ValueError, match="superlu.*only"):
        MagneticPeriodicSolver(
            make_cell(),
            make_options(
                tmp_path / "unsupported-solver.h5",
                sparse_solver=SparseSolver.PARDISO,
            ),
        )
