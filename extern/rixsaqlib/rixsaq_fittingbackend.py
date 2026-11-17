import json
import lmfit
import numpy as np
import scipy.special

def find_nearest_idx(array, value):
    # Helper function from: https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx

class RixsAqFitModel():
    """
    Base class for any fit model
    The fit region sets the area to cut before performing the fit
    """
    name = "PlaceHolder"

    def __init__(self):
        self.parameters = lmfit.Parameters()
        self.lastFitOutput = None
        self.fitregion = None # Set to None if not used, otherwise it is [12.2, 900]
        self.minimizer = 'least_squares'

    def addParameter(self, name, startguess=None, minimum=None, maximum=None, description=None, hasGuiAdjustLines=False):
        """
        Adds an lmfit parameter, wrapping to make it easier to take care of in pyqt later,
        hasGuiAdjustLines, is a flag to the GUI that it should draw lines for it
        """
        self.parameters.add(name, startguess, min=minimum, max=maximum)
        self.parameters[name].description = description # Ugly hack to add attribute to lmfit parameter?
        self.parameters[name].hasGuiAdjustLines = hasGuiAdjustLines

    @staticmethod
    def model(parameters, x):
        """
        The model which to fit to, the curve
        """
        raise NotImplementedError

    @classmethod
    def residual(cls, parameters, x, y):
        """
        Returns the residual after the function has been evaluated over x and compared to the data y
        """
        return cls.model(parameters, x) - y

    def Fit(self, x, y):
        """
        Perform a fit, store output, and return output
        """
        # Begin with cutting x, and y to region
        if self.fitregion is not None:
            lo,hi = self.fitregion[0], self.fitregion[1]
            loidx, hiidx = find_nearest_idx(x, lo), find_nearest_idx(x, hi)
            x = x[loidx:hiidx]
            y = y[loidx:hiidx]
        output = lmfit.minimize(self.residual, self.parameters, args=(x,y), method=self.minimizer)
        self.lastFitOutput = output
        return output

class RixsAqFitModelLinear(RixsAqFitModel):
    """
    This is a simple example showing the base needed to put fit a function
    """
    name = "Linear"
    def __init__(self):
        RixsAqFitModel.__init__(self)
        self.addParameter('k',1.0, description="Slope")#, minimum=0, maximum=10.0)
        self.addParameter('m',0.0, description="Offset")#,  minimum=0, maximum=10.0)

    @staticmethod
    def model(parameters, x):
        """
        Evaluate this function over x, based on parameters
        """
        return x*parameters['k']+parameters['m']


class RixsAqFitModelGaussian(RixsAqFitModel):
    """
    Fitting a simple gaussian function, with FWHM, rather than sigma, and with a linear offset
    """
    name = "Gaussian"

    def __init__(self):
        RixsAqFitModel.__init__(self)
        self.addParameter('amp',2.0, description="Amplitude", minimum=0)
        self.addParameter('mu',2.0, description="Peak center position", hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('FWHM',10.0, description="Peak width, FWHM=2.355*sigma", minimum=0)
        self.addParameter('offset', 0)

    @staticmethod
    def model(parameters, x):
        """
        Evaluate this function over x, based on parameters
        """
        sigma = parameters['FWHM']/2.355
        return parameters['amp']*np.exp(-np.power((x - parameters['mu'])/sigma, 2.)/2.)+parameters['offset']

class RixsAqFitModel2Gaussian(RixsAqFitModel):
    name = "2 Gaussians"

    def __init__(self):
        RixsAqFitModel.__init__(self)
        self.addParameter('amp1',2.0, description="Amplitude", minimum=0)
        self.addParameter('mu1',2.0, description="Peak center position", hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('FWHM1',10.0, description="Peak width, FWHM=2.355*sigma", minimum=0)
        self.addParameter('amp2',2.0, description="", minimum=0)
        self.addParameter('mu2',2.0, hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('FWHM2',10.0, description="", minimum=0)
        self.addParameter('offset', 0)

    @staticmethod
    def model(parameters, x):
        """
        Evaluate this function over x, based on parameters
        """
        sigma1 = parameters['FWHM1']/2.355
        gauss1 = parameters['amp1']*np.exp(-np.power((x - parameters['mu1'])/sigma1, 2.)/2.)
        sigma2 = parameters['FWHM2']/2.355
        gauss2 = parameters['amp2']*np.exp(-np.power((x - parameters['mu2'])/sigma2, 2.)/2.)
        return gauss1+gauss2+parameters['offset']


class RixsAqFitModel3Gaussian(RixsAqFitModel):
    name = "3 Gaussians"

    def __init__(self):
        RixsAqFitModel.__init__(self)
        self.addParameter('amp1',2.0, description="Amplitude", minimum=0)
        self.addParameter('mu1',2.0, description="Peak center position", hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('FWHM1',10.0, description="Peak width, FWHM=2.355*sigma", minimum=0)
        self.addParameter('amp2',2.0, description="", minimum=0)
        self.addParameter('mu2',2.0, description="", hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('FWHM2',10.0, description="", minimum=0)
        self.addParameter('amp3',2.0, description="", minimum=0)
        self.addParameter('mu3',2.0, description="", hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('FWHM3',10.0, description="", minimum=0)
        self.addParameter('offset', 0)

    @staticmethod
    def model(parameters, x):
        """
        Evaluate this function over x, based on parameters
        """
        sigma1 = parameters['FWHM1']/2.355
        gauss1 = parameters['amp1']*np.exp(-np.power((x - parameters['mu1'])/sigma1, 2.)/2.)
        sigma2 = parameters['FWHM2']/2.355
        gauss2 = parameters['amp2']*np.exp(-np.power((x - parameters['mu2'])/sigma2, 2.)/2.)
        sigma3 = parameters['FWHM3']/2.355
        gauss3 = parameters['amp3']*np.exp(-np.power((x - parameters['mu3'])/sigma3, 2.)/2.)
        return gauss1+gauss2+gauss3+parameters['offset']


class RixsAqFitModelLorentzian(RixsAqFitModel):
    name = "Lorentzian"

    def __init__(self):
        RixsAqFitModel.__init__(self)
        self.addParameter('amp',2.0, description="Amplitude", minimum=0)
        self.addParameter('mu',2.0, description="Peak center position", hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('FWHM',10.0, description="Peak width, FWHM=2*gamma", minimum=0)
        self.addParameter('offset', 0)

    @staticmethod
    def model(parameters, x):
        """
        Evaluate this function over x, based on parameters
        """
        gamma = parameters['FWHM']/2
        l = ((gamma**2)/(((x-parameters['mu'])**2)+gamma**2))
        l = l/np.max(l)
        return l*parameters['amp']+parameters['offset']

class RixsAqFitModelVoigt(RixsAqFitModel):
    name = "Voigt"

    def __init__(self):
        RixsAqFitModel.__init__(self)
        self.addParameter('amp',2.0, description="Amplitude", minimum=0)
        self.addParameter('mu',2.0, description="Peak center position", hasGuiAdjustLines=True, minimum=0, maximum=1024)
        self.addParameter('sigma',10.0, description="Gaussian peak width", minimum=0)
        self.addParameter('gamma',10.0, description="Lorentzan peak width", minimum=0)
        self.addParameter('offset', 0)

    @staticmethod
    def model(parameters, x):
        """
        Evaluate this function over x, based on parameters
        """
        mu = parameters['mu']
        sigma = parameters['sigma']
        gamma = parameters['gamma']
        voigt = np.real(scipy.special.wofz(((x-mu) + 1j*gamma)/sigma/np.sqrt(2))) / sigma\
                                                           /np.sqrt(2*np.pi)
        return parameters['amp']*voigt/np.max(voigt)+parameters['offset']


defaultExpertModel = """
class RixsAqFitModelExpert(RixsAqFitModel):
    name = "Expert"

    def __init__(self):
        RixsAqFitModel.__init__(self)
        self.addParameter('a',2.0)
        self.addParameter('b',2.0)

    @staticmethod
    def model(parameters, x):
        return parameters['a']*x + parameters['b']
"""


def expertModel(classtr):
    """
    Executing the input string. This is not very safe,
    but we can trust our people?, Trying to limit it as much as possible to
    be a little bit safer
    """
    local = dict() # To extract the new class
    # global to limit the useage
    globaldict = {'RixsAqFitModel':RixsAqFitModel, 'np': np, '__builtins__': {"__name__": __name__,
                  "staticmethod": staticmethod,
                  "True": True,
                  "False": False,
                  "__build_class__": __build_class__}}
    exec(classtr, globaldict, local)
    RixsAqFitModelExpert = local['RixsAqFitModelExpert']
    RixsAqFitModelExpert.stringrep = classtr
    return RixsAqFitModelExpert


models = [RixsAqFitModelGaussian, RixsAqFitModelLorentzian, RixsAqFitModelVoigt, RixsAqFitModelLinear, RixsAqFitModel2Gaussian, RixsAqFitModel3Gaussian]
modelnames = [model.name for model in models]
modelnames.append("Expert")
minimizers = ["least_squares", "leastsq", "nelder", "differential_evolution", "ampgo", "basinhopping"]


def to_json(fitmodel):
    """
    Creates a json_string with the neccesary settings to recreate the 
    fitmodel
    """
    parametersdict = {}
    params = fitmodel.parameters
    for key in params.keys():
        parametersdict[key] = {"value": params[key].value,
                               "min":   params[key].min,
                               "max":   params[key].max,
                               "vary":  params[key].vary}
    fitmodeldict = {"model": fitmodel.name,
                    "minimizer": fitmodel.minimizer,
					"fitregion": fitmodel.fitregion,
                    "parameters": parametersdict}
    if fitmodel.name == "Expert":
        fitmodeldict['expertmodelstr'] = fitmodel.stringrep
    jsonstr = json.dumps(fitmodeldict)
    return jsonstr


def from_json(jsonstr):
    """
    Takes a jsonstring and creates the fitmodel from it
    """
    modeldict = json.loads(jsonstr)
    modelname = modeldict["model"]
    if modelname != "Expert":
        modeltypeidx = modelnames.index(modelname)
        modeltype = models[modeltypeidx]
    if modelname == "Expert":
        modeltype = expertModel(modeldict['expertmodelstr'])
    newmodel = modeltype()
    newmodel.fitregion = modeldict["fitregion"]
    newmodel.minimizer = modeldict["minimizer"]
    for key, param in modeldict["parameters"].items():
        newmodel.parameters[key].min = param["min"]
        newmodel.parameters[key].max = param["max"]
        newmodel.parameters[key].value = param["value"]
        newmodel.parameters[key].vary = param["vary"]
    return newmodel


if __name__ == "__main__":
    datay = np.array([5,6,7])
    datax = np.array([1,2,3])

    myModel = RixsAqFitModelLinear()
    out = myModel.Fit(datax, datay)
    print(out)
    jsonout = to_json(myModel)
    print(jsonout)
    print(to_json(from_json(jsonout)))
