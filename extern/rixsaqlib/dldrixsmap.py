import numpy as np
#import threading


class DLDRixsMap:
    """
    Helper class to keep a rixsmapbuffer for the GUI
    """
    def __init__(self):
        self.spectra = []
        self.steps = []
        self.incidentenergies = []
        self.emissionscales = []

    def add_spectrum(self, spectrum, stepnr, incidentenergy, emissionscale):
        self.spectra.append(spectrum)
        self.steps.append(stepnr)
        self.incidentenergies.append(incidentenergy)
        self.emissionscales.append(emissionscale)

    def create_rixsmap(self, emissionbinsize=1):
        if len(self.emissionscales) == 0:
            return np.array([])
        binsize = emissionbinsize
        spectra = self.spectra
        if binsize != 1:
            spectra = [y.reshape((y.shape[0]//binsize, binsize)).sum(axis=1) for y in self.spectra]
        return np.array(spectra)

    def get_last_emissionscale(self):
        return self.emissionscales[-1]
