"""Spectrum operations and peak fitting on synthetic data."""
from datetime import datetime, timedelta

import lmfit
import numpy as np
import pytest

import fitting_functions as ff
from spectrum import (
    FWHM_TO_SIGMA,
    Spectrum,
    _gauss_uneven,
    combine,
    compare,
    runs_needed,
)


def peak(n=400, centre=200.0, counts=1000.0, start=None, seconds=10, arm=525.0):
    x = np.arange(float(n))
    y = counts * np.exp(-0.5 * ((x - centre) / 5) ** 2) / (5 * np.sqrt(2 * np.pi))
    start = start or datetime(2026, 11, 17)
    return Spectrum(x, y, {"start_time": start, "end_time": start + timedelta(seconds=seconds),
                           "arm_energy": arm})


def test_rerunning_an_operation_gives_the_same_result():
    s = peak()
    assert np.array_equal(s.smooth(5).intensity, s.smooth(5).intensity)
    with pytest.raises(ValueError):
        s.intensity[0] = 1


def test_smooth_gauss_keeps_area_and_widens_by_fwhm():
    s = peak()
    wide = s.smooth_gauss(10)
    assert wide.intensity.sum() == pytest.approx(s.intensity.sum())
    assert wide.intensity.max() < s.intensity.max()


def test_normalised_has_unit_area():
    s = peak().normalised()
    assert np.trapezoid(s.intensity, s.energy) == pytest.approx(1)


def test_combine_sums_counts_and_time():
    short, long = peak(counts=10, seconds=5), peak(counts=10_000, seconds=7200)
    total = combine([short, long])
    assert total.intensity.sum() == pytest.approx(short.intensity.sum() + long.intensity.sum())
    assert total.meta["acquisition_time"] == 7205


def test_combine_refuses_different_axes_and_warns_on_arm_energy():
    with pytest.raises(ValueError):
        combine([peak(), peak().shifted(1)])
    with pytest.warns(UserWarning, match="arm energies"):
        combine([peak(arm=523.0), peak(arm=525.0)])


def test_residual_fit_gives_sane_uncertainty():
    x = np.linspace(520, 530, 400)
    y = ff.gauss(x, 525, 1.0, 1.0) + np.random.default_rng(0).normal(0, 0.02, x.size)
    params = ff.make_parameters([524.5], amp=(0.8, 0, 2), width=(1.5, 0.01, 5))
    result = lmfit.minimize(ff.residual, params, args=(x, y, 1, "gauss"))
    assert result.params["position_0"].value == pytest.approx(525, abs=0.01)
    assert 0.001 < result.params["amp_0"].stderr < 0.02


@pytest.mark.parametrize("profile", ff.PROFILES)
def test_legacy_wrappers_match_peak_sum(profile):
    x = np.linspace(520, 530, 50)
    params = ff.make_parameters([523, 526], gamma=(0.1, 1e-3, 1))
    assert np.array_equal(getattr(ff, f"plot_{profile}")(params, x, 2)[1], ff.peak_sum(params, x, 2, profile))
    with pytest.warns(DeprecationWarning):
        squared = getattr(ff, f"sum_{profile}")(params, x, x, 2)
    assert np.array_equal(squared, ff.residual(params, x, x, 2, profile) ** 2)


def test_smooth_gauss_on_an_uneven_axis_keeps_counts_and_matches_even_case():
    x = np.arange(400.0)
    uneven = Spectrum(500 + 0.002 * x + 2e-6 * x**2, peak().intensity)
    smoothed = uneven.smooth_gauss(0.05)
    assert smoothed.intensity.sum() == pytest.approx(uneven.intensity.sum())
    even = peak()
    spread = _gauss_uneven(even.energy, even.intensity, 10 * FWHM_TO_SIGMA)
    assert np.allclose(spread[50:350], even.smooth_gauss(10).intensity[50:350])


def test_binned_sums_counts_and_averages_energy():
    s = Spectrum(np.arange(10.0), np.arange(10.0))
    b = s.binned(3)
    assert np.array_equal(b.energy, [1, 4, 7])
    assert np.array_equal(b.intensity, [3, 12, 21])


def test_compare_checks_each_run_against_the_others_and_finds_the_odd_one():
    rng = np.random.default_rng(0)
    expected = peak(n=2000, centre=1000, counts=2e5).intensity + 20
    a, b, c = (Spectrum(np.arange(2000.0), rng.poisson(expected), {"acq": i}) for i in range(3))
    results = compare([a, b, c])
    assert [chi2 for chi2, _ in results] == pytest.approx([1, 1, 1], abs=0.1)
    assert [pull.meta["acq"] for _, pull in results] == [0, 1, 2]
    assert np.array_equal(results[0][1].energy, a.energy)

    odd = Spectrum(a.energy, np.roll(c.intensity, 5))
    chi2s = [chi2 for chi2, _ in compare([a, b, odd])]
    assert chi2s[2] > 3 and chi2s[2] == max(chi2s)

    (chi2_a, pull_a), (chi2_b, pull_b) = compare([a, b])
    assert chi2_a == pytest.approx(chi2_b)
    assert np.allclose(pull_a.intensity, -pull_b.intensity)
    with pytest.raises(ValueError):
        compare([a])
    with pytest.raises(ValueError):
        compare([a, a.shifted(1)])


def test_runs_needed_scales_with_inverse_square_of_the_target():
    flat = Spectrum(np.arange(50.0), np.full(50, 100.0))
    assert runs_needed([flat, flat], target=1 / np.sqrt(200)) == pytest.approx(2)
    assert runs_needed([flat, flat], target=0.5 / np.sqrt(200)) == pytest.approx(8)
    assert runs_needed([flat], target=0.1, lo=10, hi=20) == pytest.approx(1)
