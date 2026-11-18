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

    def __len__(self):
        return self.energy.size

    def _with(self, intensity=None, energy=None) -> Spectrum:
        return replace(self,
                       intensity=self.intensity if intensity is None else intensity,
                       energy=self.energy if energy is None else energy)

    def smooth(self, k: int) -> Spectrum:
        """Box average over k bins."""
        return self._with(np.convolve(self.intensity, np.ones(k) / k, mode="same"))

    def smooth_gauss(self, fwhm: float) -> Spectrum:
        """Gaussian smoothing; fwhm is in the units of `energy` (channels or eV)."""
        steps = np.diff(self.energy)
        step = np.abs(steps).mean()
        if np.ptp(np.abs(steps)) > 0.01 * step:
            raise ValueError("smooth_gauss needs an evenly spaced axis; resample with on_grid() first")
        sigma = fwhm * FWHM_TO_SIGMA / step
        half = int(np.ceil(4 * sigma))
        x = np.arange(-half, half + 1)
        kernel = np.exp(-0.5 * (x / sigma) ** 2)
        return self._with(np.convolve(self.intensity, kernel / kernel.sum(), mode="same"))

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

    def plot(self, ax=None, **kw):
        import matplotlib.pyplot as plt
        ax = ax or plt.gca()
        ax.plot(self.energy, self.intensity, **kw)
        return ax


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
    arms = {round(m["arm_energy"], 2) for m in parts if "arm_energy" in m}
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
