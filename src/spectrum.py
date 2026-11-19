"""A spectrum as an immutable value: energy axis, intensity and metadata.

Every operation returns a new Spectrum, so re-running a notebook cell gives
the same result every time. The energy axis can be detector channels or eV.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field, replace

import numpy as np

FWHM_TO_SIGMA = 1 / (2 * np.sqrt(2 * np.log(2)))


def _frozen(a) -> np.ndarray:
    a = np.array(a, dtype=float)
    a.flags.writeable = False
    return a


@dataclass(frozen=True, eq=False)
class Spectrum:
    energy: np.ndarray
    intensity: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        energy, intensity = _frozen(self.energy), _frozen(self.intensity)
        if energy.ndim != 1 or energy.shape != intensity.shape:
            raise ValueError(f"energy {energy.shape} and intensity {intensity.shape} must be equal 1-D shapes")
        object.__setattr__(self, "energy", energy)
        object.__setattr__(self, "intensity", intensity)
        object.__setattr__(self, "meta", dict(self.meta))

    def __len__(self):
        return self.energy.size

    def _with(self, intensity=None, energy=None) -> Spectrum:
        return replace(self,
                       intensity=self.intensity if intensity is None else intensity,
                       energy=self.energy if energy is None else energy)

    def _convolved(self, kernel) -> Spectrum:
        if kernel.size > self.intensity.size:
            raise ValueError(f"smoothing kernel ({kernel.size} bins) is longer than the spectrum ({self.intensity.size})")
        return self._with(np.convolve(self.intensity, kernel, mode="same"))

    def smooth(self, k: int) -> Spectrum:
        """Box average over k bins."""
        return self._convolved(np.ones(k) / k)

    def smooth_gauss(self, fwhm: float) -> Spectrum:
        """Gaussian smoothing; fwhm is in the units of `energy` (channels or eV).

        Also works on an unevenly spaced axis such as a quadratic eV scale, where
        each bin's counts are spread over its neighbours by energy distance.
        """
        steps = np.abs(np.diff(self.energy))
        step = steps.mean()
        if np.ptp(steps) > 0.01 * step:
            return self._with(_gauss_uneven(self.energy, self.intensity, fwhm * FWHM_TO_SIGMA))
        sigma = fwhm * FWHM_TO_SIGMA / step
        half = int(np.ceil(4 * sigma))
        x = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return self._convolved(kernel / kernel.sum())

    def lowpass(self, frequency: float = 0.1) -> Spectrum:
        """Zero every Fourier component above `frequency` (cycles per bin)."""
        f = np.fft.fft(self.intensity)
        f[np.abs(np.fft.fftfreq(self.intensity.size)) > frequency] = 0
        return self._with(np.fft.ifft(f).real)

    def normalised(self) -> Spectrum:
        """Divide by the area under the curve."""
        return self._with(self.intensity / np.trapezoid(self.intensity, x=self.energy))

    def shifted(self, dE: float) -> Spectrum:
        return self._with(energy=self.energy + dE)

    def cropped(self, lo: float, hi: float) -> Spectrum:
        keep = (self.energy >= lo) & (self.energy <= hi)
        return replace(self, energy=self.energy[keep], intensity=self.intensity[keep])

    def on_grid(self, grid) -> Spectrum:
        """Linear interpolation onto `grid`; zero outside the original range."""
        grid = np.asarray(grid, dtype=float)
        order = np.argsort(self.energy)
        return replace(self, energy=grid,
                       intensity=np.interp(grid, self.energy[order], self.intensity[order], left=0, right=0))

    def binned(self, k: int) -> Spectrum:
        """Sum the counts of every k neighbouring bins; the energy is each group's mean.

        Bins left over at the end are dropped.
        """
        n = len(self) // k * k
        return replace(self, energy=self.energy[:n].reshape(-1, k).mean(1),
                       intensity=self.intensity[:n].reshape(-1, k).sum(1))

    def plot(self, ax=None, **kw):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.plot(self.energy, self.intensity, **kw)
        return ax


def _gauss_uneven(energy, intensity, sigma):
    n = energy.size
    half = int(np.ceil(4 * sigma / np.abs(np.diff(energy)).min()))
    if 2 * half + 1 > n:
        raise ValueError(f"smoothing kernel ({2 * half + 1} bins) is longer than the spectrum ({n})")
    target = np.arange(n)[:, None] + np.arange(-half, half + 1)
    inside = (target >= 0) & (target < n)
    target = np.clip(target, 0, n - 1)
    weights = np.where(inside, np.exp(-0.5 * ((energy[target] - energy[:, None]) / sigma) ** 2), 0.0)
    weights /= weights.sum(axis=1, keepdims=True)  # each source bin keeps its total counts
    return np.bincount(target[inside], weights=(weights * intensity[:, None])[inside], minlength=n)


def _acquisition_seconds(meta: dict):
    start, end = meta.get("start_time"), meta.get("end_time")
    return (end - start).total_seconds() if start and end else None


def combine(spectra) -> Spectrum:
    """Sum the raw counts of spectra on one energy axis.

    Use this to merge acquisitions: a 2 h run then weighs 1440 times a 5 s run.
    Normalise afterwards if needed.
    """
    spectra = list(spectra)
    first = spectra[0]
    for s in spectra[1:]:
        if not np.array_equal(s.energy, first.energy):
            raise ValueError("combine needs identical energy axes; sum on channels, then calibrate")
    parts = [s.meta for s in spectra]
    arms = {round(m["arm_energy"], 2) for m in parts if np.isfinite(m.get("arm_energy", np.nan))}
    if len(arms) > 1:
        warnings.warn(f"combining spectra measured at different arm energies {sorted(arms)}: "
                      "the same channel is a different emission energy", stacklevel=2)
    seconds = [_acquisition_seconds(m) for m in parts]
    meta = {"parts": parts, "counts": float(sum(s.intensity.sum() for s in spectra)),
            "acquisition_time": None if None in seconds else sum(seconds)}
    return Spectrum(first.energy, np.sum([s.intensity for s in spectra], axis=0), meta)


def average(spectra, grid=None) -> Spectrum:
    """Unweighted mean of spectra (e.g. already normalised ones) on `grid`, default the first's axis."""
    spectra = list(spectra)
    grid = spectra[0].energy if grid is None else grid
    stack = np.stack([s.on_grid(grid).intensity for s in spectra])
    return Spectrum(grid, stack.mean(axis=0), {"parts": [s.meta for s in spectra]})


def _normalised_with_sigma(s: Spectrum):
    area = np.trapezoid(s.intensity, x=s.energy)
    return s.intensity / area, np.sqrt(np.clip(np.abs(s.intensity), 1, None)) / area


def _window(s: Spectrum, lo, hi) -> np.ndarray:
    return (s.energy >= (-np.inf if lo is None else lo)) & (s.energy <= (np.inf if hi is None else hi))


def compare(spectra, lo=None, hi=None) -> list[tuple[float, Spectrum]]:
    """Reduced chi2 and pull of each raw-count run against the sum of the other runs.

    All are normalised to unit area. A pull is the run minus the others, in units
    of their combined Poisson error, and carries the run's meta; the reduced chi2
    is its mean square between lo and hi. About 1 means the run differs from the
    rest only by counting noise, so it can be combined with them; the run with the
    largest value is the odd one out. With two runs both results are the same
    comparison, mirrored. The Poisson error leaves out the noise of the dark counts
    the DAQ subtracts, so expect a little above 1 for DAQ spectra.
    """
    spectra = list(spectra)
    if len(spectra) < 2:
        raise ValueError("compare needs at least two spectra")
    if any(not np.array_equal(s.energy, spectra[0].energy) for s in spectra):
        raise ValueError("compare needs identical energy axes")
    results = []
    for i, run in enumerate(spectra):
        others = combine(spectra[:i] + spectra[i + 1:])
        (y, sigma), (yo, sigma_o) = _normalised_with_sigma(run), _normalised_with_sigma(others)
        pull = Spectrum(run.energy, (y - yo) / np.hypot(sigma, sigma_o), run.meta)
        inside = pull.intensity[_window(pull, lo, hi)]
        results.append((float(np.sum(inside ** 2) / (inside.size - 1)), pull))
    return results


def runs_needed(spectra, target: float, lo=None, hi=None) -> float:
    """How many runs like these to combine for a relative counting error of `target` per bin.

    The answer is a total that includes the runs given, counted in runs of their
    average length, so subtract len(spectra) for the number still to measure.
    The error is taken at the mean counts per bin between lo and hi, so bin the runs
    first (Spectrum.binned) to judge it at the resolution you plot. It falls as
    1/sqrt(counts): halving the target takes four times the runs.
    """
    spectra = list(spectra)
    total = combine(spectra)
    mean = total.intensity[_window(total, lo, hi)].mean()
    if mean <= 0:
        raise ValueError(f"no counts between {lo} and {hi}")
    return len(spectra) / (mean * target ** 2)
