"""Temperature of an evaporatively cooled water-glycerol microdroplet in vacuum.

Knudsen evaporation model of Supplementary Note 3 of Berkowicz, Andronis et al.
(Nat. Commun. 2024, "Supercritical density fluctuations and structural
heterogeneity in supercooled water-glycerol microdroplets"), equation numbers
(S15-S26) below refer to it.

The droplet set-up (size, spacing, frequency, chamber pressure, composition)
fixes the temperature as a function of flight time, so `DropletCooling` runs the
cooling simulation once when it is created and then answers any travel distance
by interpolating the cached curve:

    cool = DropletCooling(diameter_um=18.7, spacing_um=85.5,
                          frequency_hz=157e3, pressure_pa=1.6)
    cool.temperature(5)                 # K at 5 mm from the nozzle
    cool.temperature([5, 20, 65])

The thermodynamic properties are simple closed forms or short interpolation
tables (the paper interpolates experimental data from several references), so
temperatures agree with the paper to a few K rather than exactly.
"""
from __future__ import annotations

import warnings

import numpy as np

K_B = 1.380649e-23         # J/K
N_A = 6.02214076e23        # 1/mol
R_GAS = 8.314462618        # J/mol/K
M_WATER = 18.01528e-3      # kg/mol
M_GLYCEROL = 92.09382e-3   # kg/mol

# Pure-water density (kg/m^3), read off Supplementary Fig. 7A.
_RHO_W_T = [228.0, 235.0, 240.0, 250.0, 260.0, 277.0, 310.0]
_RHO_W = [957.0, 972.0, 982.0, 993.0, 998.0, 1000.0, 1000.0]


def p_vap_water(T):
    """Saturation vapour pressure of supercooled liquid water in Pa (Murphy & Koop 2005)."""
    T = np.asarray(T, dtype=float)
    return np.exp(54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T
                  + np.tanh(0.0415 * (T - 218.8))
                  * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))


def density(T, w_g):
    """Mixture density in kg/m^3: pure water plus a linear glycerol correction (cf. S22)."""
    return np.interp(T, _RHO_W_T, _RHO_W) * (1 + 0.25 * w_g)


def heat_capacity(T, w_g):
    """Isobaric heat capacity in J/kg/K, mass-fraction weighted water + glycerol (cf. S24).

    Water follows 4.18 + 4.0 exp(-(T - 228)/9.8) kJ/kg/K, a fit to Supplementary
    Fig. 7B that captures the rise on supercooling; glycerol is taken as 2.3 kJ/kg/K.
    """
    cp_w = 4180.0 + 4000.0 * np.exp(-(np.asarray(T, dtype=float) - 228.0) / 9.8)
    return (1 - w_g) * cp_w + w_g * 2300.0


def conductivity(T, w_g):
    """Thermal conductivity in W/m/K, mass-fraction weighted (S26)."""
    k_w = 0.46 + 0.0021 * (np.asarray(T, dtype=float) - 228.0)   # Supplementary Fig. 7E
    return (1 - w_g) * k_w + w_g * 0.29


def vapor_pressure(T, x_g):
    """Mixture saturation pressure in Pa (S23), with Raoult's law for P_rel(x_g)."""
    return p_vap_water(T) * (1 - x_g)


def vaporization_enthalpy(T, x_g):
    """Enthalpy of vaporisation in J/mol (S25); pure water is linear in T (Supplementary Fig. 7C)."""
    T = np.asarray(T, dtype=float)
    dh_w = 1e3 * (44.0 + 0.05 * (300.0 - T))
    return -R_GAS * T * np.log(1 - x_g) + dh_w


def mass_fraction(x_g):
    """Glycerol mass fraction from its mole fraction."""
    return x_g * M_GLYCEROL / (x_g * M_GLYCEROL + (1 - x_g) * M_WATER)


def _shell_geometry(r, n_shells):
    """Shell thickness, outer radii and volumes for a droplet of radius r split into equal-thickness shells."""
    dr = r / n_shells
    r_out = np.arange(1, n_shells + 1) * dr
    return dr, r_out, 4 / 3 * np.pi * (r_out ** 3 - (r_out - dr) ** 3)


def stable_time_step(radius, n_shells, T, x_g):
    """Largest explicit time step (s) at which no shell gives away more heat than it holds.

    Each shell needs dt < C_i / (sum of its conductances), with C_i its heat capacity.
    The limit scales with radius squared, so it is evaluated for a droplet 5 % smaller
    to leave room for the shrinking during the flight.
    """
    w_g = mass_fraction(x_g)
    dr, r_out, volume = _shell_geometry(0.95 * radius, n_shells)
    T = np.full(n_shells, float(T))
    heat_per_K = heat_capacity(T, w_g) * density(T, w_g) * volume
    G = conductivity(T, w_g) * 4 * np.pi * r_out ** 2 / dr              # outer face of each shell
    G_total = G + np.concatenate(([0.0], G[:-1]))
    G_total[-1] = G[-2]                                                  # no conduction outside the surface
    return float(np.min(heat_per_K / G_total))


class DropletCooling:
    """Cooling curve of one droplet set-up, computed once and interpolated.

    Parameters
    ----------
    diameter_um : initial droplet diameter (um).
    spacing_um, frequency_hz : droplet-droplet distance (um) and generation frequency (Hz);
        the droplet speed is their product. Pass None for both if `velocity_ms` is given.
    pressure_pa : chamber vacuum pressure (Pa).
    x_glycerol : initial glycerol mole fraction (0.032 in the paper; 0 is pure water).
    T0 : initial droplet temperature (K).
    gamma : evaporation coefficient.
    velocity_ms : droplet speed (m/s), instead of frequency * spacing.
    dt : explicit time step (s). The default picks 10 ns or, for small droplets, 0.8 of the
        stability limit. An explicit dt above that limit raises ValueError, because the
        scheme then diverges quietly.
    max_time : longest flight time simulated (s); distances beyond speed * max_time raise.
    n_shells : number of spherical shells the droplet is divided into.
    record_interval : time between cached points (s).
    weighting : "volume" (default) averages the shell temperatures by shell volume, which
        is what a probe of the whole droplet sees; "shell" takes the plain mean over shells.

    Only water evaporates, so the kinetic flux uses the mass of a water molecule. The
    droplet volume follows the evaporated molecules at water's molar volume (S19) while the
    shell masses use the mixture density, so the shell mass drifts a few percent from the
    true mass; this changes T by under 0.1 K.

    Property fits are only tabulated down to about 228 K; a warning is raised if the
    droplet goes below 225 K or starts above 310 K.
    """

    def __init__(self, diameter_um, spacing_um, frequency_hz, pressure_pa, *,
                 x_glycerol=0.032, T0=298.0, gamma=1.0, velocity_ms=None,
                 dt=None, max_time=7e-3, n_shells=100, record_interval=1e-6,
                 weighting="volume"):
        self.diameter_um = diameter_um
        self.spacing_um = spacing_um
        self.frequency_hz = frequency_hz
        self.pressure_pa = pressure_pa
        self.x_glycerol = x_glycerol
        self.T0 = T0
        self.gamma = gamma
        self.n_shells = n_shells
        self.weighting = weighting
        self._velocity = self._check_inputs(velocity_ms, max_time, record_interval, n_shells)
        self.dt = self._choose_dt(dt)
        self.n_steps = int(np.ceil(max_time / self.dt))
        self.record_every = max(1, round(record_interval / self.dt))
        self.time, self._temperature, self._x_g, self._radius = self._simulate()

    def _check_inputs(self, velocity_ms, max_time, record_interval, n_shells):
        def need(ok, msg):
            if not ok:
                raise ValueError(msg)

        need(self.diameter_um > 0, "diameter_um must be positive")
        need(0 <= self.x_glycerol < 1, "x_glycerol must be in [0, 1)")
        need(200 < self.T0 < 373, "T0 must be between 200 and 373 K")
        need(0 < self.gamma <= 1, "gamma must be in (0, 1]")
        need(max_time > 0 and record_interval > 0, "max_time and record_interval must be positive")
        need(isinstance(n_shells, (int, np.integer)) and n_shells >= 3, "n_shells must be an integer >= 3")
        need(self.weighting in ("volume", "shell"), 'weighting must be "volume" or "shell"')
        need(self.pressure_pa >= 0, "pressure_pa must not be negative")
        p_start = float(vapor_pressure(self.T0, self.x_glycerol))
        need(self.pressure_pa < p_start,
             f"pressure_pa={self.pressure_pa} Pa is not below the vapour pressure at T0 "
             f"({p_start:.0f} Pa): the droplet would not evaporate")
        if velocity_ms is None:
            need(self.spacing_um and self.frequency_hz and self.spacing_um > 0 and self.frequency_hz > 0,
                 "give spacing_um and frequency_hz (both positive) or velocity_ms")
            velocity_ms = self.frequency_hz * self.spacing_um * 1e-6
        need(velocity_ms > 0, "velocity_ms must be positive")
        if self.T0 > 310:
            warnings.warn("T0 above 310 K: property fits are only tabulated up to about 310 K", stacklevel=3)
        return float(velocity_ms)

    def _choose_dt(self, dt):
        limit = stable_time_step(self.diameter_um * 1e-6 / 2, self.n_shells, self.T0, self.x_glycerol)
        if dt is None:
            return min(10e-9, 0.8 * limit)
        if dt > limit:
            raise ValueError(f"dt={dt:.3g} s exceeds the stability limit {limit:.3g} s for this droplet "
                             f"and n_shells; use dt=None, or fewer shells")
        return dt

    @property
    def velocity(self):
        """Droplet speed in m/s (frequency * droplet-droplet distance unless given directly)."""
        return self._velocity

    @property
    def max_distance_mm(self):
        return self.time[-1] * self.velocity * 1e3

    def flight_time(self, z_mm):
        """Travel time in s for a distance z from the nozzle in mm (S21)."""
        return np.asarray(z_mm, dtype=float) * 1e-3 / self.velocity

    def _lookup(self, values, z_mm):
        t = self.flight_time(z_mm)
        if not np.all((t >= 0) & (t <= self.time[-1])):        # also rejects nan
            raise ValueError(f"distance outside the simulated range 0-{self.max_distance_mm:.1f} mm; "
                             "raise max_time to go further")
        return np.interp(t, self.time, values)

    def temperature(self, z_mm):
        """Droplet temperature in K at distance z_mm (scalar or array)."""
        return self._lookup(self._temperature, z_mm)

    def glycerol_fraction(self, z_mm):
        """Glycerol mole fraction at distance z_mm."""
        return self._lookup(self._x_g, z_mm)

    def radius_um(self, z_mm):
        """Droplet radius in um at distance z_mm."""
        return self._lookup(self._radius, z_mm) * 1e6

    def _simulate(self):
        N, dt = self.n_shells, self.dt
        r0 = self.diameter_um * 1e-6 / 2
        V = 4 / 3 * np.pi * r0 ** 3
        w_g0 = mass_fraction(self.x_glycerol)
        m_mean = self.x_glycerol * M_GLYCEROL + (1 - self.x_glycerol) * M_WATER
        n_total = V * density(self.T0, w_g0) / m_mean * N_A          # molecules
        n_g = self.x_glycerol * n_total
        n_w = n_total - n_g
        m_water = M_WATER / N_A
        n_evap = 0.0

        T = np.full(N, float(self.T0))
        T_min = self.T0
        n_rec = self.n_steps // self.record_every + 1
        time = np.arange(n_rec) * self.record_every * dt
        T_mean, x_rec, r_rec = np.empty(n_rec), np.empty(n_rec), np.empty(n_rec)

        for step in range(self.n_steps + 1):
            r = (3 * V / (4 * np.pi)) ** (1 / 3)
            x_g = n_g / (n_g + n_w - n_evap)
            dr, r_out, shell_volume = _shell_geometry(r, N)
            if step % self.record_every == 0:
                i = step // self.record_every
                weights = shell_volume if self.weighting == "volume" else None
                T_mean[i], x_rec[i], r_rec[i] = np.average(T, weights=weights), x_g, r
                T_min = min(T_min, T.min())
            if step == self.n_steps:
                break

            w_g = mass_fraction(x_g)
            rho, cp, kappa = density(T, w_g), heat_capacity(T, w_g), conductivity(T, w_g)
            heat_per_K = cp * shell_volume * rho                        # J/K per shell

            # heat flowing into shell i from shell i+1 (S17); inner shell is a full sphere
            q = 0.5 * (kappa[:-1] + kappa[1:]) * 4 * np.pi * r_out[:-1] ** 2 * (T[1:] - T[:-1]) / dr
            Q = np.zeros(N)
            Q[:-1] += q
            Q[1:] -= q
            dT = Q * dt / heat_per_K                                    # S18

            # Knudsen evaporation from the outer shell (S15, S16)
            Ts = T[-1]
            p_eff = max(vapor_pressure(Ts, x_g) - self.pressure_pa, 0.0)
            rate = self.gamma * p_eff * 4 * np.pi * r ** 2 / np.sqrt(2 * np.pi * m_water * K_B * Ts)
            dn = rate * dt
            dT[-1] -= dn * vaporization_enthalpy(Ts, x_g) / N_A / heat_per_K[-1]
            T += dT

            V -= dn * m_water / np.interp(Ts, _RHO_W_T, _RHO_W)         # S19, molar volume of water
            n_evap += dn

        if not (np.all(np.isfinite(T_mean)) and np.all(np.isfinite(x_rec))):
            raise RuntimeError("the cooling simulation diverged; try a smaller dt or fewer shells")
        if T_min < 225.0:
            warnings.warn(f"droplet cools to {T_min:.1f} K, below the ~228 K range of the property "
                          "fits (heat capacity is extrapolated); treat those temperatures with care",
                          stacklevel=3)
        return time, T_mean, x_rec, r_rec
