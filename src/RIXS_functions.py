from veritas_paths import ENERGY_SCALES  # before h5py, see veritas_paths
import h5py
import beamtimes as bts
from spectrum import Spectrum
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import rixsaqlib.dldfile
import rixsaqlib.dlddata
import lmfit

BEAMTIMES=tuple(bts.BEAMTIMES)

def rolling_avg(data,kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')

def split_spectrum(filep,acnr,provide_cutoff=False,**kwargs):
    #The spectra are not background corrected yet. 
    #That doesn't make that much of a difference but is something to fix later on.
    
    with h5py.File(filep, 'r', locking=False) as file:
        beamline_energy=file[f'{acnr}/endmetadata/beamline_energy/position'][()]
    file1 = rixsaqlib.dldfile.DLDFile(filep, acnr)
    points = file1.loadPoints()
    countrate=points[:,3][1:]-points[:,3][:-1]
    countrate=rolling_avg(countrate,512)

    histogram=np.histogram(1/countrate,bins=400)

    if not provide_cutoff:
        plt.plot(histogram[1][:-1],np.log(histogram[0]),label=f'{int(beamline_energy)} eV')
        plt.xlabel('countrate')
        plt.title('histogram over countrate (high countrate peak is probably crystallised)')
        plt.legend()
        plt.grid()
        plt.show()
        limit=float(input('set countrate cutoff'))
        limit=1/limit
    
    else:
        limit=kwargs['cutoff']
        limit=1/float(limit)
        
    filtered_points=points[np.where(countrate>limit)]
    filtered_points_high=points[np.where(countrate<limit)]
    
    calibration = file1.loadCalibration()
    corrected_points=calibration.corrected_points(filtered_points)
    corrected_points_2=calibration.corrected_points(filtered_points_high)
    
    spec_low=np.zeros(8192)
    spec_high=np.zeros(8192)
    index=corrected_points[:,0].astype(int)
    index2=corrected_points_2[:,0].astype(int)
    
    for indice in index:
        spec_low[indice]+=1
    
    for indice in index2:
        spec_high[indice]+=1

    
    return spec_low,spec_high,beamline_energy

# Old module-level names, now computed on first access (PEP 562).
_LEGACY = {
    **{f'runinfo_{k}': (lambda k=k: bts.run_table(k)) for k in bts.BEAMTIMES},
    '_RUNINFO': lambda: {k: bts.run_table(k) for k in bts.BEAMTIMES},
    '_DLD_FILES': lambda: {k: list(b.dld_files) for k, b in bts.BEAMTIMES.items()},
    'shift_m23': lambda: -1.14213434,
    'shift_f24': lambda: -1,
    'escale_m23_1': lambda: bts.energy_scale('m23', 113),
    'escale_f24_1': lambda: bts.energy_scale('f24', 0),
    'escale_f24_2': lambda: bts.energy_scale('f24', 56),
    'escale_m24_1': lambda: bts.energy_scale('m24', 0),
    'escale_d22': lambda: bts.energy_scale('d22', 0) if (ENERGY_SCALES / bts.energy_scale_entry('d22', 0)['file']).exists() else None,
}


def __getattr__(name):
    if name in _LEGACY:
        return _LEGACY[name]()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")



#valence=runinfo[runinfo['RIXS or XAS']=='RIXS-valence']
#water_valence=valence[valence['Sample name']=='Water']
#escale_day_6=np.load('escale_day_6.npy')
#escale_day_10=np.load('escale_day_10.npy')
#escale_day_11=np.load('escale_day_11.npy')


"""
Alright, what are the runinfo files? There is one for:
December 2022
March 2023
February 2024
March 2024

Let's just load them all here. Then for each, let's just manually make calibrated axes for every calibration instance. Then when averaging,
let's just have it interpolate. This is not rocket sience. Not motivated to do it though...


Step by step:
1. load all the runinfo in this or a separate config file. I say just do it here.
2. now do the energy scales. You have feb 24 and mar 24 now. what remains is:
    dec 22 (probably single file, the one you have. Maybe make them match the really old data??)
    mar 23 - day 10 is final escale water. shift that to match the others
    day 11 - I don't know how to do that yet. make the  D2O match somehow


"""


def _position(metadata, name):
    """A motor position from start/end metadata, or nan if the DAQ recorded none."""
    try:
        return float(np.array(metadata[name]['position']))
    except (KeyError, ValueError, TypeError):
        return np.nan


LEGACY_DARK_START = 500  # used for files without calibration/dark_time_window


def _dark_time_window(filename, run_number, calibration):
    """The dark window the DAQ recorded, or the legacy one of the light window's width."""
    with h5py.File(filename, 'r', locking=False) as f:
        recorded = f[run_number].get('Instrument/DLD8080/calibration/dark_time_window')
        if recorded is not None:
            return recorded[()]
    width = np.abs(calibration.time_window[1]-calibration.time_window[0])
    return [LEGACY_DARK_START, LEGACY_DARK_START+width]


class RIXS_run:
    def __init__(self,filename, run_number,index=np.nan, **kwargs):


        #this bit is so that you can determine what runinfo file to use with kwargs
        if 'runinfo' not in kwargs:
            raise ValueError(f'specify runinfo as one of {BEAMTIMES}')
        self.marker=bts.get(str(kwargs['runinfo']).lower()).key
        self.runinfo=bts.run_table(self.marker)

        #now come the calibration files. let's start with f24 because that is most solid. Then we define all other scales with the help of that 
        #one.
        
        
        self.run_number=run_number
        self.filename=filename
        self.spectrum=np.array([])
        self.energy_scale=np.array([])
        self.dark=np.array([])
        self.corrected_spectrum=([])
        self.binsize=1
        self.index=index
        self.excitation_energy=np.nan
        self.arm_energy=np.nan
        self.beamline_energy=np.nan
        with h5py.File(filename, 'r', locking=False) as file:
            metadata = file[run_number].get('startmetadata', {})
            self.arm_energy = _position(metadata, 'veritasarm_energy')
            self.beamline_energy = _position(metadata, 'beamline_energy')
    def print_information(self):
        with h5py.File(self.filename, 'r', locking=False) as f:
            metadata = f[self.run_number]['startmetadata']
            print('{},  Beamline energy: {:.2f} eV, Slit: {:.2f}, Arm energy: {:.2f} eV'.format(
                self.run_number, _position(metadata, 'beamline_energy'),
                _position(metadata, 'a_slit1_v'), _position(metadata, 'veritasarm_energy')))
        
    def initialise_calib_indices(self,runinfo):
        pass
        
    def loadspectrum(self):
        file1 = rixsaqlib.dldfile.DLDFile(self.filename, self.run_number)
        #with h5py.File(file1.hdf5filename, "r") as f:
        #    arme = f[self.run_number]["startmetadata/veritasarm_energy/position"][()]
        #    slit = f[self.run_number]["startmetadata/a_slit1_v/position"][()]
        #    beamline = f[self.run_number]["startmetadata/beamline_energy/position"][()]
        #    #print("{},  Beamline energy: {:.2f} eV, Slit: {:.2f}, Arm energy: {:.2f} eV".format(dataset,beamline,slit,arme))

        points = file1.loadPoints()

        calibration = file1.loadCalibration()
        # Open the same calibration again, but this time we will change it a bit
        calibration2 = file1.loadCalibration()
        darktimes = _dark_time_window(self.filename, self.run_number, calibration)
        #print("Light timewindow:", calibration.time_window)
        ## Change it in the second calibration
        calibration2.set_time_windows(darktimes)
        #print("Dark timewindow:", calibration2.time_window)

        # Go from rawpoints and get corrected points
        # also get dark_points using the second calibration
        #corrected_points = calibration.corrected_points(points)
        dark_points = calibration2.corrected_points(points)
        corrected_points = calibration.corrected_points(points)

        # Hack to get functions for creating spectrum
        data = rixsaqlib.dlddata.DLDData(calibration)
        escale = calibration.energy_scale

        spec = data.create_spectrum(corrected_points)
        # Lets do a little rebinning as well
        escale = escale.reshape((escale.shape[0]//self.binsize, self.binsize)).mean(axis=1)
        spec = spec.reshape((spec.shape[0]//self.binsize, self.binsize)).sum(axis=1)        

        spec_dark = data.create_spectrum(dark_points)
        spec_dark = spec_dark.reshape((spec_dark.shape[0]//self.binsize, self.binsize)).sum(axis=1)
        
        self.spectrum=spec
        self.dark=spec_dark
        self.energy_scale=escale
        self.corrected_spectrum=spec-spec_dark
        return spec, spec_dark

    def as_spectrum(self):
        """The corrected spectrum on the current energy scale, as a Spectrum (loads it if needed)."""
        if not len(self.corrected_spectrum):
            self.loadspectrum()
        return Spectrum(self.energy_scale, self.corrected_spectrum,
                        {"beamtime": self.marker, "run": self.run_number, "row": self.index,
                         "file": self.filename, "beamline_energy": self.beamline_energy,
                         "arm_energy": self.arm_energy})

    def set_energy_scale(self,energy_scale):
        self.energy_scale=np.array(energy_scale)
    
    def rolling_avg(self,kernel_size):
        self.corrected_spectrum=rolling_avg(self.corrected_spectrum,kernel_size)
        
    def plot_spectrum(self):
        fig,ax=plt.subplots()
        ax.plot(self.energy_scale,self.corrected_spectrum)
        return fig,ax
    def normalise(self):
        self.corrected_spectrum=self.corrected_spectrum/np.trapezoid(self.corrected_spectrum,x=self.energy_scale)
    def load_energy_scale(self,offset=0):
        #Set calibration indices based on file
        assert not np.isnan(self.index), 'specify index first'

        self.energy_scale=bts.energy_scale(self.marker,int(self.index))
        self.energy_scale=self.energy_scale+offset

        """
        if self.index<100:
            self.energy_scale=escale_day_6+offset
        elif self.index<172:
            self.energy_scale=escale_day_10+offset
        else:
            self.energy_scale=escale_day_11+offset
        """
        
    def get_excitation_energy(self):
        if not np.isnan(self.index):
            self.excitation_energy=self.runinfo.loc[int(self.index)]['Veritas beamline energy (eV)']

    def fourier_correction(self,frequency=0.1):
        if frequency:
            fft=np.fft.fft(self.corrected_spectrum)
            freq=np.fft.fftfreq(len(self.corrected_spectrum))
            fft[np.where(np.abs(freq)>frequency)]=0
            self.corrected_spectrum=np.fft.ifft(fft).real

def filter_samples(excitation_energy,**kwargs):
    """
    specify runinfo as m23, f24 or m24
    specify excitation_energy
    specify sample_name
    index_start and index_end can be entered
    """

    if kwargs.get('runinfo') not in BEAMTIMES:
        raise ValueError(f'specify runinfo as one of {BEAMTIMES}')
    bt=bts.get(kwargs['runinfo'])
    table=bts.run_table(bt.key)
    mask=table['Veritas beamline energy (eV)']==excitation_energy
    for arg,column in {'sample_name':'Sample name','type_of_ice':'Type of ice',
                       'temperature':bt.temperature_column}.items():
        if arg in kwargs:
            mask&=table[column]==kwargs[arg]
    index=table.index[mask].tolist()
    index=np.array(index)
    if 'index_start' in kwargs:
        index=index[np.where(index>kwargs['index_start'])]
    if 'index_end' in kwargs:
        index=index[np.where(index<kwargs['index_end'])]
    return index


def load_from_index(index,**kwargs):
    """ choose d22, m23, f24 or m24 as runinfo,"""
    beamtime = kwargs.get('runinfo')
    if beamtime not in BEAMTIMES:
        raise ValueError(f'specify runinfo as one of {BEAMTIMES}')
    return RIXS_run(str(bts.dld_file(beamtime, index)), bts.run_table(beamtime).loc[index]['Run name'],
                    index=index, runinfo=beamtime)
