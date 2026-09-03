#Let's see if we can build something with lmfit
import numpy as np
import lmfit

def lorentz(x,x0,gamma): 
    one_o_pi=0.3183098861837907
    return one_o_pi*(gamma/((x-x0)**2+gamma**2))

def gauss(x,position,amp,width):
    return amp*np.exp(-(x-position)**2/2/(width/2.355)**2)

from scipy.special import voigt_profile
voigt = lambda x,x0,amp,width,gamma:np.nan_to_num(amp*voigt_profile(x-x0,width/2.355,gamma))

def pseudo_voigt(x,x0,amp,width,gamma):
    fL=2*gamma
    fg=width
    #eta=1.36603(f_{L}/f)-0.47719(f_{L}/f)^{2}+0.11116(f_{L}/f)^{3},}

def pseudo_voigt(x, center, amplitude, fwhm_g, gamma):
    """
    Compute the pseudo-Voigt profile (Thompson-Cox-Hastings approximation)
    accurate to ~1% of the true Voigt.

    Parameters
    ----------
    x : array_like
        Energy (or other) axis values.
    amplitude : float
        Peak height (max intensity).
    center : float
        Peak center position.
    fwhm_g : float
        Gaussian FWHM (e.g. instrumental broadening).
    fwhm_l : float
        Lorentzian FWHM (e.g. lifetime broadening).

    Returns
    -------
    y : ndarray
        Pseudo-Voigt profile values.
    """
    fwhm_l=2*gamma
    # Total FWHM
    fwhm = (fwhm_g**5 + 2.69269*fwhm_g**4*fwhm_l + 2.42843*fwhm_g**3*fwhm_l**2 +
            4.47163*fwhm_g**2*fwhm_l**3 + 0.07842*fwhm_g*fwhm_l**4 +
            fwhm_l**5)**(1/5)

    # Mixing parameter η (fraction Lorentzian)
    eta = 1.36603*(fwhm_l/fwhm) - 0.47719*(fwhm_l/fwhm)**2 + 0.11116*(fwhm_l/fwhm)**3
    eta = np.clip(eta, 0, 1)  # avoid numerical overshoot

    # Gaussian and Lorentzian components (normalized)
    sigma = fwhm_g / (2*np.sqrt(2*np.log(2)))
    gaussian = np.exp(-((x - center)**2) / (2*sigma**2)) / (sigma*np.sqrt(2*np.pi))
    lorentzian = (fwhm_l / 2) / ((x - center)**2 + (fwhm_l / 2)**2) / np.pi

    # Weighted sum
    return amplitude * (eta * lorentzian + (1 - eta) * gaussian)




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
        
def sum_gauss(params,x,data,number):
    
    function=np.zeros(len(x))
    for thing in range(number):
        function+=gauss(x,params[f'position_{thing}'],params[f'amp_{thing}'],params[f'width_{thing}'])
        res=(data-function)**2

    return res

def sum_voigt(params,x,data,number):
    function=np.zeros(len(x))
    for thing in range(number):
        function+=voigt(x,params[f'position_{thing}'],params[f'amp_{thing}'],
                       params[f'width_{thing}'],params[f'gamma_{thing}'])
        res=np.array([(d-function[v])**2 for v,d in enumerate(data)])
    return res


def sum_pseudo_voigt(params,x,data,number):
    function=np.zeros(len(x))
    for thing in range(number):
        function+=pseudo_voigt(x,params[f'position_{thing}'],params[f'amp_{thing}'],
                       params[f'width_{thing}'],params[f'gamma_{thing}'])
        res=np.array([(d-function[v])**2 for v,d in enumerate(data)])
    return res
    
def plot_voigt(params,x,number):
    function=np.zeros(len(x))
    for thing in range(number):
        function+=voigt(x,params[f'position_{thing}'],params[f'amp_{thing}'],
                       params[f'width_{thing}'],params[f'gamma_{thing}'])
    return x,function

def plot_pseudo_voigt(params,x,number):
    function=np.zeros(len(x))
    for thing in range(number):
        function+=pseudo_voigt(x,params[f'position_{thing}'],params[f'amp_{thing}'],
                       params[f'width_{thing}'],params[f'gamma_{thing}'])
    return x,function
    
def plot_gauss(params,x,number):
    function=np.zeros(len(x))
    for thing in range(number):
        function+=gauss(x,params[f'position_{thing}'],params[f'amp_{thing}'],params[f'width_{thing}'])
    return x,function


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

