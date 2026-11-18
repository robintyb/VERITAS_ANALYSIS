"""Peak models for lmfit. Parameters are named position_i, amp_i, width_i, gamma_i.

Fit with `lmfit.minimize(residual, params, args=(x, data, number, profile))`.
"""
import warnings

import lmfit
import numpy as np
from scipy.special import voigt_profile


def lorentz(x,x0,gamma): 
    one_o_pi=0.3183098861837907
    return one_o_pi*(gamma/((x-x0)**2+gamma**2))

def gauss(x,position,amp,width):
    return amp*np.exp(-(x-position)**2/2/(width/2.355)**2)

voigt = lambda x,x0,amp,width,gamma:np.nan_to_num(amp*voigt_profile(x-x0,width/2.355,gamma))  # noqa: E731


def pseudo_voigt(x, center, amplitude, fwhm_g, gamma): #heightnorm
    """
    Pseudo-Voigt using peak-height-normalized Gaussian and Lorentzian.
    amplitude -> peak height (value at center)
    fwhm_g, fwhm_l -> FWHMs of Gaussian and Lorentzian components.
    """
    fwhm_l=2*gamma
    # combined FWHM (Thompson-Cox-Hastings style)
    f = (fwhm_g**5 + 2.69269*fwhm_g**4*fwhm_l + 2.42843*fwhm_g**3*fwhm_l**2 +
         4.47163*fwhm_g**2*fwhm_l**3 + 0.07842*fwhm_g*fwhm_l**4 + fwhm_l**5)**(1/5)

    # mixing parameter (r = fwhm_l / f)
    r = fwhm_l / f
    eta = 1.36603*r - 0.47719*r**2 + 0.11116*r**3
    eta = np.clip(eta, 0.0, 1.0)

    # height-normalized Gaussian: G(0)=1, FWHM = fwhm_g
    gauss = np.exp(-4*np.log(2) * ((x - center)**2) / (fwhm_g**2))

    # height-normalized Lorentzian: L(0)=1, FWHM = fwhm_l
    lorentz = 1.0 / (1.0 + 4.0 * ((x - center)**2) / (fwhm_l**2))

    # weighted sum -> amplitude is peak height at center
    return amplitude * (eta * lorentz + (1.0 - eta) * gauss)


# name: (function, takes gamma). amp is the peak height for gauss and
# pseudo_voigt, and the area for voigt. width is a Gaussian FWHM; gamma is
# a Lorentzian HWHM. A new line shape is one new entry here.
PROFILES = {
    "gauss": (gauss, False),
    "voigt": (voigt, True),
    "pseudo_voigt": (pseudo_voigt, True),
}


def _peak(params, x, i, profile):
    f, takes_gamma = PROFILES[profile]
    args = [params[f'position_{i}'], params[f'amp_{i}'], params[f'width_{i}']]
    if takes_gamma:
        args.append(params[f'gamma_{i}'])
    return f(x, *args)


def peak_sum(params, x, number, profile="gauss"):
    """The model: the sum of `number` peaks."""
    total = np.zeros(len(x))
    for i in range(number):
        total += _peak(params, x, i, profile)
    return total


def components(params, x, number, profile="gauss"):
    """Each peak on its own, for plotting the decomposition."""
    return [_peak(params, x, i, profile) for i in range(number)]


def residual(params, x, data, number, profile="gauss"):
    """data - model, the residual lmfit.minimize expects."""
    return data - peak_sum(params, x, number, profile)


def create_parameters(number):
    fit_params=lmfit.Parameters()
    for thing in range(number):
        fit_params.add(f'amp_{thing}',value=2e-4,min=1e-5,max=5e-4)
        if thing<3:
            fit_params.add(f'width_{thing}',value=1,min=0.001)
        elif thing<5:
            fit_params.add(f'width_{thing}',value=0.2,min=0.001,max=1)
        else:
            fit_params.add(f'width_{thing}',value=0.1,min=0.001,max=1)

    for v,thing in enumerate(np.linspace(521,525,3)):
        fit_params.add(f'position_{v}', value=thing)
        
    for v, thing in enumerate(np.linspace(525,527,number-3)):
        fit_params.add(f'position_{v+3}', value=thing,min=520,max=527.5)
    return fit_params

def add_lorentz_gamma(fit_params,number,value=1,min=0.001,max=10):
    for thing in range(number):
        fit_params.add(f'gamma_{thing}',value=value,min=min,max=max)
    return fit_params


def make_parameters(positions, amp=(2e-4, 1e-5, 5e-4), width=(0.2, 1e-3, 1.0),
                    gamma=None, position_window=None):
    """Parameters for len(positions) peaks; amp, width and gamma are (value, min, max)."""
    p = lmfit.Parameters()
    for i, x0 in enumerate(positions):
        lo, hi = (x0 - position_window, x0 + position_window) if position_window else (-np.inf, np.inf)
        p.add(f'position_{i}', value=x0, min=lo, max=hi)
        p.add(f'amp_{i}', value=amp[0], min=amp[1], max=amp[2])
        p.add(f'width_{i}', value=width[0], min=width[1], max=width[2])
        if gamma:
            p.add(f'gamma_{i}', value=gamma[0], min=gamma[1], max=gamma[2])
    return p


class SquaredResidualWarning(DeprecationWarning):
    pass

# Python hides DeprecationWarnings raised from library code, and lmfit is
# what calls sum_*, so show this one explicitly.
warnings.simplefilter("default", SquaredResidualWarning)


def _deprecated_squared(name, profile, params, x, data, number):
    warnings.warn(f"{name} returns the squared residual, so lmfit minimises the sum of r**4 and "
                  f"its uncertainties are wrong. Use residual(params, x, data, number, '{profile}').",
                  SquaredResidualWarning, stacklevel=3)
    return residual(params, x, data, number, profile)**2

def sum_gauss(params,x,data,number):
    return _deprecated_squared('sum_gauss', 'gauss', params, x, data, number)

def sum_voigt(params,x,data,number):
    return np.asarray(_deprecated_squared('sum_voigt', 'voigt', params, x, data, number))

def sum_pseudo_voigt(params,x,data,number):
    return np.asarray(_deprecated_squared('sum_pseudo_voigt', 'pseudo_voigt', params, x, data, number))

def plot_voigt(params,x,number):
    return x, peak_sum(params, x, number, 'voigt')

def plot_pseudo_voigt(params,x,number):
    return x, peak_sum(params, x, number, 'pseudo_voigt')

def plot_gauss(params,x,number):
    return x, peak_sum(params, x, number, 'gauss')


def set_initial_positions(fit_params,positions,limit=0.5):
    for v,position in enumerate(positions):
        fit_params.add(f'position_{v}', value=position,min=position-limit,max=position+limit)
    return fit_params



def set_bounds(fit_params,param,indice,bounds):
    fit_params[f'{param}_{indice}'].min=bounds[0]
    fit_params[f'{param}_{indice}'].max=bounds[1]
    return fit_params

def reconstruct_from_npy(filepath='./535_final_fitting_all.npy'):
    fit_params_ndarray=np.load(filepath)
    fit_params_ndarray=fit_params_ndarray.reshape((int(len(fit_params_ndarray)/4),4))
    fit_params=lmfit.Parameters()
    for v,row in enumerate(fit_params_ndarray):
        fit_params.add(f'position_{v}',value=row[0])
        fit_params.add(f'amp_{v}',value=row[1])
        fit_params.add(f'width_{v}',value=row[2])
        fit_params.add(f'gamma_{v}',value=row[3])
    return fit_params,fit_params_ndarray

