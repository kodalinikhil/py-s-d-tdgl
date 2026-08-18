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
from tdgl.magnetic_periodic.solution import (
    COMPONENT_NAMES_ATTRIBUTE,
    GENERIC_COMPONENT_HDF5_SCHEMA_VERSION,
    HDF5_BACKEND,
    LEGACY_HDF5_SCHEMA_VERSION,
    MagneticPeriodicSolution,
    component_names_for_model,
    options_to_json,
    write_frame_components,
)
from tdgl.solver.options import SolverOptions


def make_cell(model):
    return MagneticPeriodicCell(
        layer=Layer(
            coherence_length=1,
            london_lambda=2,
            thickness=0.1,
            model=model,
        ),
        lengths=(4, 3),
        shape=(3, 4),
        flux_quanta=1,
    )


def write_checkpoint(path, model, components, *, schema_version):
    cell = make_cell(model)
    options = SolverOptions(
        solve_time=1,
        output_file=str(path),
        terminal_psi=None,
    )
    with h5py.File(path, "w") as h5file:
        h5file.attrs["backend"] = HDF5_BACKEND
        h5file.attrs["schema_version"] = schema_version
        h5file.attrs["complete"] = True
        h5file.attrs["total_seconds"] = 0.1
        h5file.attrs["final_step"] = 0
        h5file.attrs["final_time"] = 0.0
        cell.to_hdf5(h5file.create_group("cell"))
        h5file.create_group("options").attrs["json"] = options_to_json(options)
        frame = h5file.create_group("data").create_group("0")
        frame.attrs["step"] = 0
        frame.attrs["time"] = 0.0
        frame.attrs["dt"] = 0.0
        if schema_version == GENERIC_COMPONENT_HDF5_SCHEMA_VERSION:
            write_frame_components(frame, components, model=model)
        else:
            # Schema v1 was s+d only and stored fields by physical name.
            frame["psi_s"] = components[0]
            frame["psi_d"] = components[1]
        frame["vector_potential"] = np.zeros((2, *cell.shape))
        frame["supercurrent"] = np.zeros((2, *cell.shape))
        frame["normal_current"] = np.zeros((2, *cell.shape))
        frame["epsilon"] = np.ones(cell.shape)
    return cell


@pytest.mark.parametrize(
    ("model", "names"),
    [
        (SingleBandModel(), ("psi",)),
        (SPlusDModel(), ("s", "d")),
        (DPlusDPrimeModel(), ("d", "d_prime")),
        (SPlusSModel(), ("s1", "s2")),
    ],
)
def test_component_names_cover_supported_models(model, names):
    assert component_names_for_model(model) == names


@pytest.mark.parametrize(
    ("model", "names"),
    [
        (SingleBandModel(), ("psi",)),
        (SPlusDModel(), ("s", "d")),
        (DPlusDPrimeModel(), ("d", "d_prime")),
        (SPlusSModel(), ("s1", "s2")),
    ],
)
def test_schema_v2_roundtrip_is_model_neutral(tmp_path, model, names):
    path = tmp_path / f"{type(model).__name__}.h5"
    shape = (3, 4)
    components = tuple(
        np.full(shape, index + 1j * (index + 1), dtype=complex)
        for index in range(len(names))
    )
    write_checkpoint(
        path,
        model,
        components,
        schema_version=GENERIC_COMPONENT_HDF5_SCHEMA_VERSION,
    )

    solution = MagneticPeriodicSolution.from_hdf5(path)
    frame = solution.final_frame
    assert solution.component_names == names
    assert frame.component_names == names
    assert len(solution.components) == len(names)
    for name, expected in zip(names, components):
        assert solution.get_component(name) == pytest.approx(expected)
        assert frame.component_map[name] == pytest.approx(expected)
    assert solution.psi1 == pytest.approx(components[0])
    if len(components) == 1:
        assert solution.psi == pytest.approx(components[0])
        with pytest.raises(AttributeError, match="one component"):
            _ = solution.psi2
    else:
        assert solution.psi2 == pytest.approx(components[1])
    assert np.isfinite(solution.free_energy_density(include_magnetic=False))


def test_named_component_accessors_are_unambiguous(tmp_path):
    shape = (3, 4)
    first = np.full(shape, 1 + 2j)
    second = np.full(shape, 3 + 4j)

    sd_path = tmp_path / "s-plus-d.h5"
    write_checkpoint(
        sd_path,
        SPlusDModel(),
        (first, second),
        schema_version=GENERIC_COMPONENT_HDF5_SCHEMA_VERSION,
    )
    sd = MagneticPeriodicSolution.from_hdf5(sd_path)
    assert sd.psi_s == pytest.approx(first)
    assert sd.psi_d == pytest.approx(second)

    ddp_path = tmp_path / "d-plus-d-prime.h5"
    write_checkpoint(
        ddp_path,
        DPlusDPrimeModel(),
        (first, second),
        schema_version=GENERIC_COMPONENT_HDF5_SCHEMA_VERSION,
    )
    ddp = MagneticPeriodicSolution.from_hdf5(ddp_path)
    assert ddp.psi_d == pytest.approx(first)
    assert ddp.psi_d_prime == pytest.approx(second)
    assert ddp.get_component("d'") == pytest.approx(second)
    assert ddp.get_component("d_xy") == pytest.approx(second)

    ss_path = tmp_path / "s-plus-s.h5"
    write_checkpoint(
        ss_path,
        SPlusSModel(),
        (first, second),
        schema_version=GENERIC_COMPONENT_HDF5_SCHEMA_VERSION,
    )
    ss = MagneticPeriodicSolution.from_hdf5(ss_path)
    assert ss.psi_s1 == pytest.approx(first)
    assert ss.psi_s2 == pytest.approx(second)


def test_schema_v1_s_plus_d_files_remain_compatible(tmp_path):
    path = tmp_path / "legacy.h5"
    psi_s = np.full((3, 4), 2 + 1j)
    psi_d = np.full((3, 4), 4 + 3j)
    write_checkpoint(
        path,
        SPlusDModel(),
        (psi_s, psi_d),
        schema_version=LEGACY_HDF5_SCHEMA_VERSION,
    )

    solution = MagneticPeriodicSolution.from_hdf5(path)
    assert solution.component_names == ("s", "d")
    assert solution.psi1 == pytest.approx(psi_s)
    assert solution.psi2 == pytest.approx(psi_d)
    assert solution.psi_s == pytest.approx(psi_s)
    assert solution.psi_d == pytest.approx(psi_d)


def test_schema_v1_rejects_non_s_plus_d_model_metadata(tmp_path):
    path = tmp_path / "legacy-mislabeled-ddp.h5"
    fields = (np.ones((3, 4), dtype=complex), np.zeros((3, 4), dtype=complex))
    write_checkpoint(
        path,
        DPlusDPrimeModel(),
        fields,
        schema_version=LEGACY_HDF5_SCHEMA_VERSION,
    )

    with pytest.raises(IOError, match="valid only for SPlusDModel"):
        MagneticPeriodicSolution.from_hdf5(path)


def test_component_writer_records_generic_datasets_and_names(tmp_path):
    path = tmp_path / "components.h5"
    first = np.ones((2, 3), dtype=complex)
    second = 2 * first
    with h5py.File(path, "w") as h5file:
        frame = h5file.create_group("frame")
        names = write_frame_components(
            frame,
            (first, second),
            model=DPlusDPrimeModel(),
        )
        assert names == ("d", "d_prime")
        assert set(frame) == {"psi1", "psi2"}
        stored_names = tuple(
            value.decode() if isinstance(value, bytes) else str(value)
            for value in frame.attrs[COMPONENT_NAMES_ATTRIBUTE]
        )
        assert stored_names == names


def test_component_writer_and_reader_reject_ambiguous_metadata(tmp_path):
    path = tmp_path / "bad-components.h5"
    arrays = (np.ones((3, 4)), np.zeros((3, 4)))
    with h5py.File(path, "w") as h5file:
        frame = h5file.create_group("frame")
        with pytest.raises(ValueError, match="do not match"):
            write_frame_components(
                frame,
                arrays,
                model=DPlusDPrimeModel(),
                component_names=("s", "d"),
            )

    write_checkpoint(
        path,
        DPlusDPrimeModel(),
        arrays,
        schema_version=GENERIC_COMPONENT_HDF5_SCHEMA_VERSION,
    )
    with h5py.File(path, "r+") as h5file:
        names = np.asarray(("s", "d"), dtype=h5py.string_dtype("utf-8"))
        h5file["data"]["0"].attrs.modify(COMPONENT_NAMES_ATTRIBUTE, names)
    with pytest.raises(IOError, match="do not match DPlusDPrimeModel"):
        MagneticPeriodicSolution.from_hdf5(path)
