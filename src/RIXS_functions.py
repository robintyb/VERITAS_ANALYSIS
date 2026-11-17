import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import rixsaqlib.dldfile
import rixsaqlib.dlddata
import lmfit
from copy import deepcopy
import warnings
from veritas_paths import ENERGY_SCALES, RUNINFO, raw_data_dir

BEAMTIMES=('d22','m23','f24','m24')

def rolling_avg(data,kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')

def split_spectrum(filep,acnr,provide_cutoff=False,**kwargs):
    #The spectra are not background corrected yet. 
    #That doesn't make that much of a difference but is something to fix later on.
    
    with h5py.File(filep) as file:
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

runinfo_m23=pd.read_excel(RUNINFO/'Veritas jet march 2023 - runinfo-2.xlsx')
runinfo_d22=pd.read_csv(RUNINFO/'runinfo-Table 1.csv')
runinfo_f24=pd.read_excel(RUNINFO/'VERITAS FEB 2024 - runinfo.xlsx')
runinfo_m24=pd.read_excel(RUNINFO/'Veritas water jet vol 2 03-2024 .xlsx')

#I will calibrate these all in a way so that the O2 absorption is 531.2 eV
# (the shifts below are applied on load; see energy_scales/index.toml)

shift_m23=-1.14213434
shift_f24=-1

escale_m23_1=np.load(ENERGY_SCALES/'escale_m23_1.npy')-shift_m23
escale_f24_1=np.load(ENERGY_SCALES/'escale_f24_1.npy')-shift_f24
escale_f24_2=np.load(ENERGY_SCALES/'escale_f24_2.npy')-shift_f24
escale_m24_1=np.load(ENERGY_SCALES/'escale_m24_1.npy')-shift_f24+0.4

# Not yet committed to the repo - load it if someone adds it to energy_scales/
_escale_d22_file=ENERGY_SCALES/'escale_ice_whole_detector.npy'
if _escale_d22_file.exists():
    escale_d22=np.load(_escale_d22_file)-2.4+1.62
else:
    escale_d22=None
    warnings.warn(f'{_escale_d22_file.name} not found in energy_scales/; d22 energy scale unavailable.')



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


class RIXS_run:
    def __init__(self,filename, run_number,index=np.nan, **kwargs):


        #this bit is so that you can determine what runinfo file to use with kwargs
        if 'runinfo' not in kwargs:
            raise ValueError(f'specify runinfo as one of {BEAMTIMES}')
        self.marker=''
        runinfo_name=kwargs['runinfo'].lower()
        if runinfo_name in BEAMTIMES:
            self.marker=runinfo_name
            self.runinfo={'d22':runinfo_d22,'m23':runinfo_m23,'f24':runinfo_f24,'m24':runinfo_m24}[runinfo_name]
        elif '24' in runinfo_name:
            if 'eb' in runinfo_name:
                self.runinfo=runinfo_f24
                self.marker='f24'
            else:
                self.runinfo=runinfo_m24
                self.marker='m24'
        elif '23' in runinfo_name:
            self.runinfo=runinfo_m23
            self.marker='m23'
        else:
            self.runinfo=runinfo_d22
            self.marker='d22'

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
        with h5py.File(filename) as file:
            try:
                self.arm_energy=float(np.array(file[run_number]['startmetadata']['veritasarm_energy']['position']))
                self.beamline_energy=float(np.array(file[run_number]['startmetadata']['beamline_energy']['position']))
            except KeyError:
                pass
    def print_information(self):
        file1 = rixsaqlib.dldfile.DLDFile(self.filename, self.run_number)
        with h5py.File(file1.hdf5filename, "r") as f:
            arme = f[self.run_number]["startmetadata/veritasarm_energy/position"][()]
            slit = f[self.run_number]["startmetadata/a_slit1_v/position"][()]
            beamline = f[self.run_number]["startmetadata/beamline_energy/position"][()]
            print("{},  Beamline energy: {:.2f} eV, Slit: {:.2f}, Arm energy: {:.2f} eV".format(self.run_number,beamline,slit,arme))
        
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
        timewindowsize = np.abs(calibration.time_window[1]-calibration.time_window[0])
        # Place the marker for dark
        darktimes = [500,500+timewindowsize]#[800,800+timewindowsize]
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

        if self.marker=='m23':
            self.energy_scale=escale_m23_1
            if self.index<113:
                self.energy_scale=escale_m23_1+1
            
        if self.marker=='f24':
            if self.index<56:
                self.energy_scale=escale_f24_1
            elif self.index>56:
                self.energy_scale=escale_f24_2

        if self.marker=='m24':
            self.energy_scale=escale_m24_1
        if self.marker=='d22':
            if escale_d22 is None:
                raise FileNotFoundError(f'{_escale_d22_file.name} not found in energy_scales/')
            self.energy_scale=escale_d22

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

    index=np.nan
    if kwargs.get('runinfo') not in BEAMTIMES:
        raise ValueError(f'specify runinfo as one of {BEAMTIMES}')
    if 'runinfo' in kwargs:
        if kwargs['runinfo']=='m23':
            sample_valence=deepcopy(runinfo_m23)
        if kwargs['runinfo']=='f24':
            sample_valence=deepcopy(runinfo_f24)
        if kwargs['runinfo']=='m24':
            sample_valence=deepcopy(runinfo_m24)
        if kwargs['runinfo']=='d22':
            sample_valence=deepcopy(runinfo_d22)
        
    if 'sample_name' in kwargs:
        sample_valence=sample_valence[sample_valence['Sample name']==kwargs['sample_name']]
    if 'type_of_ice' in kwargs:
        sample_valence=sample_valence[sample_valence['Type of ice']==kwargs['type_of_ice']]
    if 'temperature' in kwargs:
        if not kwargs['runinfo']=='m24':
            sample_valence=sample_valence[sample_valence['Temperature (K)']==kwargs['temperature']]
        else: sample_valence=sample_valence[sample_valence['Temperature (C, Thermo Connection)']==kwargs['temperature']]
    index=sample_valence[sample_valence['Veritas beamline energy (eV)']==excitation_energy].index.tolist()#[2:]
    index=np.array(index)
    if 'index_start' in kwargs:
        index=index[np.where(index>kwargs['index_start'])]
    if 'index_end' in kwargs:
        index=index[np.where(index<kwargs['index_end'])]
    return index


def load_from_index(index,**kwargs):
    """ choose d22, m23, f24 or m24 as runinfo,"""

    if kwargs.get('runinfo') not in BEAMTIMES:
        raise ValueError(f'specify runinfo as one of {BEAMTIMES}')

    if kwargs['runinfo']=='d22':
        basepath=raw_data_dir('d22')
        files=[basepath/'20221207_DLD_1.h5',basepath/'20221209_DLD_1.h5',
               basepath/'20221210_DLD_1.h5',basepath/'20221211_DLD_1.h5']
        index_cutoffs=[0,68,94,120,9999]
        runinfo_file=runinfo_d22
    
    if kwargs['runinfo']=='m23':
        basepath=raw_data_dir('m23')
        file1=basepath/'20230401_DLD_2.h5'
        file2=basepath/'20230404_DLD_1.h5'
        file3=basepath/'20230405_DLD_2.h5'
        file4=basepath/'20230407_DLD_1.h5'
        files=[file1,file2,file3,file4]
        index_cutoffs=[0,95,105,134,9999]
        runinfo_file=runinfo_m23

    if kwargs['runinfo']=='f24':
        basepath=raw_data_dir('f24')
        index_cutoffs=[0,56,9999]
        files=[basepath/'20240220_DLD1.h5',basepath/'testdld.h5']
        runinfo_file=runinfo_f24
    
    if kwargs['runinfo']=='m24':
        basepath=raw_data_dir('m24')
        # row 26 is the "filepath changed to 20240309_dld1.h5" note, so acq120 (row 25) is in the first file
        index_cutoffs=[0,26,9999]
        files=[basepath/'20240307_DLD1.h5',basepath/'20240309_DLD1.h5']
        runinfo_file=runinfo_m24

    # each cutoff is the first row of the next file (the "file changed" note row)
    filename=None
    for v,thing in enumerate(index_cutoffs[:-1]):
        if thing<=index<index_cutoffs[v+1]:
            filename=files[v]
    if filename is None:
        raise IndexError(f'row {index} is outside the {kwargs["runinfo"]} file ranges {index_cutoffs}')
    """
    if index:
        filename=file1
    if index>113:
        filename=file2
    if index>134:
        filename=file3
    """
    return RIXS_run(str(filename),runinfo_file.loc[index]['Run name'],index=index,runinfo=kwargs['runinfo'])

#I think I can just write a function that averages RIXS_runs and returns a new one

class average_RIXS(RIXS_run):
    def __init__(self,filename,runinfo=None):
        self.spectrum=np.array([])
        self.energy_scales=np.array([])
        self.dark=np.array([])
        self.corrected_spectra=([])
        self.binsize=1
        self.runs=[]
        self.run_numbers=[]
        self.filename=filename
        self.indices=[]
        self.runinfo_name=runinfo

    def load_indices(self,run_numbers):
        self.run_numbers=np.array(run_numbers)
    def load_runs(self):
        self.runs=[RIXS_run(self.filename,number,self.indices[v],runinfo=self.runinfo_name) for v,number in enumerate(self.run_numbers)]
    def average_runs(self):
        pass

    #def load_energy_scales(self):
    #    filename=load_frim






