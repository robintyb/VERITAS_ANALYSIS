"""Droplet cooling: the paper's set-up (Supplementary Table 1), numerics, and literature points."""
import numpy as np
import pytest

from droplet_temperature import (
    K_B,
    M_WATER,
    N_A,
    DropletCooling,
    density,
    heat_capacity,
    p_vap_water,
    vaporization_enthalpy,
)

SETUP = {"diameter_um": 18.7, "spacing_um": 85.5, "frequency_hz": 157e3, "pressure_pa": 1.6}


@pytest.fixture(scope="module")
def cool():
    return DropletCooling(**SETUP, max_time=0.6e-3)             # about 8 mm


def test_water_vapour_pressure_matches_known_values():
    assert p_vap_water(273.15) == pytest.approx(611.2, rel=0.01)
    assert p_vap_water(298.15) == pytest.approx(3169, rel=0.01)


def test_flight_time_is_distance_over_frequency_times_spacing(cool):
    assert cool.velocity == pytest.approx(13.42, rel=1e-3)
    assert cool.flight_time(5) == pytest.approx(5e-3 / (157e3 * 85.5e-6))
    by_speed = DropletCooling(18.7, None, None, 1.6, velocity_ms=13.4235, max_time=1e-5)
    assert by_speed.flight_time(5) == pytest.approx(cool.flight_time(5))


def test_temperature_falls_with_distance_and_glycerol_rises(cool):
    z = np.linspace(0, 8, 20)
    assert cool.temperature(0) == pytest.approx(298.0)
    assert np.all(np.diff(cool.temperature(z)) < 0)
    assert np.all(np.diff(cool.glycerol_fraction(z)) > 0)
    assert np.all(np.diff(cool.radius_um(z)) < 0)


def test_matches_the_paper_at_5_mm(cool):
    assert cool.temperature(5) == pytest.approx(259.3, abs=1.0)     # Supplementary Fig. 6


def test_volume_weighting_is_cooler_at_the_surface_than_the_core():
    shell = DropletCooling(**SETUP, max_time=0.4e-3, weighting="shell")
    volume = DropletCooling(**SETUP, max_time=0.4e-3)
    assert volume.temperature(5) < shell.temperature(5) - 0.5


def test_range_and_array_input(cool):
    assert cool.temperature([1, 2, 3]).shape == (3,)
    for bad in (cool.max_distance_mm + 1, -1, np.nan):
        with pytest.raises(ValueError):
            cool.temperature(bad)


def test_interpolation_between_records_matches_a_finer_cache():
    coarse = DropletCooling(**SETUP, max_time=2e-4, record_interval=5e-6)
    fine = DropletCooling(**SETUP, max_time=2e-4, record_interval=1e-7)
    z = np.array([0.37, 1.1, 2.3])
    np.testing.assert_allclose(coarse.temperature(z), fine.temperature(z), atol=0.05)


def test_result_does_not_depend_on_time_step():
    a = DropletCooling(**SETUP, max_time=2e-4, dt=10e-9)
    b = DropletCooling(**SETUP, max_time=2e-4, dt=3e-9)
    assert a.temperature(2.5) == pytest.approx(b.temperature(2.5), abs=0.01)


def test_unstable_time_step_is_refused_and_default_is_stable_for_small_droplets():
    with pytest.raises(ValueError, match="stability"):
        DropletCooling(10, 85.5, 157e3, 1.6, dt=10e-9, max_time=1e-5)
    auto = DropletCooling(10, 85.5, 157e3, 1.6, max_time=5e-5)
    reference = DropletCooling(10, 85.5, 157e3, 1.6, max_time=5e-5, dt=1e-9)
    assert auto.temperature(0.6) == pytest.approx(reference.temperature(0.6), abs=0.05)
    assert np.all(np.isfinite(auto._temperature))


def test_smaller_droplet_cools_faster():
    small = DropletCooling(**{**SETUP, "diameter_um": 10}, max_time=1.5e-4)
    big = DropletCooling(**SETUP, max_time=1.5e-4)
    assert small.temperature(2) < big.temperature(2) - 1


@pytest.mark.parametrize("change", [
    {"diameter_um": 0}, {"diameter_um": -5}, {"x_glycerol": 1.0}, {"x_glycerol": -0.1},
    {"T0": 0}, {"T0": 500}, {"pressure_pa": -1}, {"pressure_pa": 5000}, {"n_shells": 1},
    {"spacing_um": None}, {"max_time": 0}, {"weighting": "mean"},
])
def test_bad_inputs_raise(change):
    with pytest.raises(ValueError):
        DropletCooling(**{**SETUP, "max_time": 1e-5, **change})


def test_warns_outside_the_property_range():
    with pytest.warns(UserWarning, match="below"):
        DropletCooling(**{**SETUP, "pressure_pa": 0.1}, T0=222.0, max_time=1e-5)
    with pytest.warns(UserWarning, match="310"):
        DropletCooling(**SETUP, T0=320.0, max_time=1e-5)


def test_higher_chamber_pressure_cools_less():
    low = DropletCooling(**{**SETUP, "pressure_pa": 1.6}, T0=235.0, max_time=2e-4)
    high = DropletCooling(**{**SETUP, "pressure_pa": 10.0}, T0=235.0, max_time=2e-4)
    assert high.temperature(2) > low.temperature(2)


def test_pure_water_has_no_glycerol_and_is_colder_than_the_mixture(cool):
    water = DropletCooling(**{**SETUP, "pressure_pa": 1.6}, x_glycerol=0.0, max_time=0.6e-3)
    assert np.all(water.glycerol_fraction([0, 4, 8]) == 0)
    assert water.temperature(8) < cool.temperature(8)


def test_initial_shrink_rate_matches_hertz_knudsen():
    first_step = DropletCooling(**SETUP, dt=10e-9, max_time=10e-9, record_interval=10e-9)
    m = M_WATER / N_A
    flux = (p_vap_water(298.0) * (1 - 0.032) - 1.6) / np.sqrt(2 * np.pi * m * K_B * 298.0)
    expected = flux * m / 1000.0                                    # dr/dt in m/s, rho = 1000
    shrink = (first_step._radius[0] - first_step._radius[1]) / first_step.time[1]
    assert shrink == pytest.approx(expected, rel=0.05)


def test_latent_heat_balances_the_enthalpy_lost(cool):
    # evaporated water times its enthalpy of vaporisation vs. what the cooled droplet gave up
    r0, r1 = cool._radius[0], cool._radius[-1]
    lost_mass = 4 / 3 * np.pi * (r0 ** 3 - r1 ** 3) * 1000.0
    latent = lost_mass / M_WATER * vaporization_enthalpy(260.0, 0.032)
    T = np.linspace(cool._temperature[-1], 298.0, 200)
    w_g = 0.145
    remaining = 4 / 3 * np.pi * r1 ** 3 * density(280.0, w_g)
    given_up = remaining * np.trapezoid(heat_capacity(T, w_g), T)
    assert given_up / latent == pytest.approx(1.0, rel=0.1)         # catches J/mol vs J/molecule and kJ vs J


# Sellberg et al., Nature 510, 381 (2014) droplet temperatures as tabulated in Laksmono et al.,
# J. Phys. Chem. Lett. 6, 2826 (2015), SI Table S1: (distance mm, temperature K), pure water.
def test_matches_sellberg_small_droplets():
    c = DropletCooling(8.7, None, None, 0.0, velocity_ms=19.23, x_glycerol=0.0, n_shells=50, max_time=2.5e-3)
    for z, T in [(25.88, 233), (36.04, 230), (46.09, 228)]:
        assert c.temperature(z) == pytest.approx(T, abs=1.5)


def test_matches_sellberg_large_droplets_once_cold():
    c = DropletCooling(12.4, None, None, 0.0, velocity_ms=10.35, x_glycerol=0.0, n_shells=50, max_time=3.1e-3)
    for z, T in [(30.60, 232)]:
        assert c.temperature(z) == pytest.approx(T, abs=3)


def test_diagnostics_describe_the_run(cool):
    d = cool.diagnostics
    assert d.n_steps == cool.n_steps and d.dt == cool.dt and d.n_shells == 100
    assert d.dt / d.dt_limit == pytest.approx(0.8)
    assert d.runtime_s > 0 and d.steps_per_second == pytest.approx(d.n_steps / d.runtime_s)
    assert d.T_min == pytest.approx(cool._temperature[-1], abs=20)            # surface is colder than the mean
    assert d.T_min < cool.temperature(cool.max_distance_mm)
    assert d.max_core_surface_gap > 5                                          # surface leads the cooling
    x0, x1 = 0.032, float(cool.glycerol_fraction(cool.max_distance_mm))        # glycerol does not evaporate
    assert d.evaporated_fraction == pytest.approx(1 - (1 - x1) / x1 * x0 / (1 - x0), rel=1e-3)
    assert d.energy_error < 1e-9
    assert abs(d.mass_drift) < 0.05
    assert "stability limit" in str(d)


def test_progress_bar_reports_to_stderr(capsys):
    DropletCooling(**SETUP, max_time=2e-5, progress=True)
    assert "droplet cooling" in capsys.readouterr().err
    DropletCooling(**SETUP, max_time=2e-5)
    assert capsys.readouterr().err == ""


def test_progress_without_tqdm_warns_and_still_runs(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "tqdm.auto", None)
    with pytest.warns(UserWarning, match="tqdm"):
        c = DropletCooling(**SETUP, max_time=2e-5, progress=True)
    assert c.temperature(0.1) < 298


def test_numpy_backend_can_be_forced_and_is_reported():
    c = DropletCooling(**SETUP, max_time=2e-5, fast=False)
    assert c.backend == "numpy" and c.diagnostics.backend == "numpy"


def test_missing_numba_falls_back_or_raises(monkeypatch):
    monkeypatch.setitem(__import__("sys").modules, "droplet_kernel", None)
    assert DropletCooling(**SETUP, max_time=2e-5).backend == "numpy"
    with pytest.raises(ImportError, match="numba"):
        DropletCooling(**SETUP, max_time=2e-5, fast=True)


def test_numba_and_numpy_backends_agree():
    pytest.importorskip("numba")
    kwargs = {**SETUP, "max_time": 4e-4}
    slow = DropletCooling(**kwargs, fast=False)
    fast = DropletCooling(**kwargs, fast=True)
    assert fast.backend == "numba"
    for name in ("_temperature", "_x_g", "_radius"):
        np.testing.assert_allclose(getattr(fast, name), getattr(slow, name), rtol=1e-9)
    assert fast.diagnostics.evaporated_fraction == pytest.approx(slow.diagnostics.evaporated_fraction, rel=1e-9)
    assert fast.diagnostics.energy_error < 1e-9


def test_numba_agrees_for_a_small_droplet_with_few_shells_and_pure_water():
    pytest.importorskip("numba")
    kwargs = {"diameter_um": 8.7, "spacing_um": None, "frequency_hz": None, "pressure_pa": 0.0,
              "velocity_ms": 19.23, "x_glycerol": 0.0, "n_shells": 30, "max_time": 3e-4}
    slow = DropletCooling(**kwargs, fast=False)
    fast = DropletCooling(**kwargs, fast=True)
    np.testing.assert_allclose(fast._temperature, slow._temperature, rtol=1e-9)
