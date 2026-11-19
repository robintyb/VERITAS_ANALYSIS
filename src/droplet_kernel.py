"""Compiled time-stepping loop for `droplet_temperature` (optional, needs numba).

Same physics as `_advance_numpy` in droplet_temperature.py, written as scalar loops so numba
can compile it; `tests/test_droplet_temperature.py` checks the two agree. Install with
`uv sync --extra dev --extra fast`. If you change the model, change both.
"""
import math

import numpy as np
from numba import njit

from droplet_temperature import _RHO_W, _RHO_W_T, K_B, M_GLYCEROL, M_WATER, N_A, R_GAS

_T_TABLE = np.array(_RHO_W_T)
_RHO_TABLE = np.array(_RHO_W)
_M_WATER_MOLECULE = M_WATER / N_A


@njit(cache=True)
def _rho_water(T):
    if T <= _T_TABLE[0]:
        return _RHO_TABLE[0]
    for j in range(1, _T_TABLE.size):
        if T <= _T_TABLE[j]:
            f = (T - _T_TABLE[j - 1]) / (_T_TABLE[j] - _T_TABLE[j - 1])
            return _RHO_TABLE[j - 1] + f * (_RHO_TABLE[j] - _RHO_TABLE[j - 1])
    return _RHO_TABLE[-1]


@njit(cache=True)
def _p_vap_water(T):
    return math.exp(54.842763 - 6763.22 / T - 4.210 * math.log(T) + 0.000367 * T
                    + math.tanh(0.0415 * (T - 218.8))
                    * (53.878 - 1331.22 / T - 9.44523 * math.log(T) + 0.014025 * T))


@njit(cache=True)
def advance(T, state, n_steps, dt, pressure, gamma, n_g, n_w, volume_unit, face_unit):
    """Advance T (in place) and state = [volume, n_evap, latent_total, stored_total] by n_steps."""
    N = T.size
    hpk = np.empty(N)
    kappa = np.empty(N)
    dT = np.empty(N)
    q = np.empty(N - 1)
    for _ in range(n_steps):
        V, n_evap = state[0], state[1]
        r = (3 * V / (4 * math.pi)) ** (1 / 3)
        x_g = n_g / (n_g + n_w - n_evap)
        w_g = x_g * M_GLYCEROL / (x_g * M_GLYCEROL + (1 - x_g) * M_WATER)
        dr = r / N
        r3 = r ** 3
        for i in range(N):
            Ti = T[i]
            rho = _rho_water(Ti) * (1 + 0.25 * w_g)
            cp = (1 - w_g) * (4180.0 + 4000.0 * math.exp(-(Ti - 228.0) / 9.8)) + w_g * 2300.0
            kappa[i] = (1 - w_g) * (0.46 + 0.0021 * (Ti - 228.0)) + w_g * 0.29
            hpk[i] = cp * volume_unit[i] * r3 * rho
        for i in range(N - 1):
            q[i] = 0.5 * (kappa[i] + kappa[i + 1]) * face_unit[i] * (r * r / dr) * (T[i + 1] - T[i])
        for i in range(N):
            Q = 0.0
            if i < N - 1:
                Q += q[i]
            if i > 0:
                Q -= q[i - 1]
            dT[i] = Q * dt / hpk[i]

        Ts = T[N - 1]
        p_eff = max(_p_vap_water(Ts) * (1 - x_g) - pressure, 0.0)
        rate = gamma * p_eff * 4 * math.pi * r ** 2 / math.sqrt(2 * math.pi * _M_WATER_MOLECULE * K_B * Ts)
        dn = rate * dt
        latent = dn * (-R_GAS * Ts * math.log(1 - x_g) + 1e3 * (44.0 + 0.05 * (300.0 - Ts))) / N_A
        dT[N - 1] -= latent / hpk[N - 1]
        state[2] += latent
        stored = 0.0
        for i in range(N):
            stored += hpk[i] * dT[i]
            T[i] += dT[i]
        state[3] += stored
        state[0] = V - dn * _M_WATER_MOLECULE / _rho_water(Ts)
        state[1] = n_evap + dn
