from veritas_paths import (  # before h5py, see veritas_paths
    ENERGY_SCALES,
    RUNINFO,
    raw_data_dir,
)

import h5py
import matplotlib.pyplot as plt
import numpy as np
import rixsaqlib.dlddata
import rixsaqlib.dldfile

import beamtimes as bts
from spectrum import Spectrum

BEAMTIMES=tuple(bts.BEAMTIMES)

def rolling_avg(data,kernel_size):
    kernel = np.ones(kernel_size) / kernel_size
    return np.convolve(data, kernel, mode='same')

def event_interval(points, window=512):
    """Time between consecutive DLD events, box-averaged over `window` events (one shorter than points)."""
    return rolling_avg(np.diff(points[:, 3]), window)


def plot_count_rate_histogram(interval, label=None, bins=400, ax=None):
    ax = ax or plt.gca()
    counts, edges = np.histogram(1 / interval, bins=bins)
    ax.plot(edges[:-1], np.log(counts), label=label)
    ax.set_xlabel('countrate')
    ax.set_title('histogram over countrate (high countrate peak is probably crystallised)')
    ax.legend()
    ax.grid()
    return ax


def split_points(points, interval, cutoff):
    """(points below, points above) a count rate of `cutoff` events per second."""
    limit = 1 / float(cutoff)
    head = points[:-1]
    return head[interval > limit], head[interval < limit]


def _histogram(calibration, points, n=8192):
    channels = calibration.corrected_points(points)[:, 0].astype(int)
    return np.bincount(channels, minlength=n)[:n].astype(float)


def split_spectrum(filep,acnr,provide_cutoff=False,cutoff=None,**kwargs):
    """Spectra of the low- and high-count-rate parts of an acquisition, and its final beamline energy.

    Pass `cutoff` (events per second) to skip the histogram and the prompt.
    The spectra are not dark-corrected.
    """
    if provide_cutoff and cutoff is None:
        raise ValueError('provide_cutoff=True needs cutoff=<events per second>')
    with h5py.File(filep, 'r', locking=False) as file:
        beamline_energy=file[f'{acnr}/endmetadata/beamline_energy/position'][()]
    file1 = rixsaqlib.dldfile.DLDFile(filep, acnr)
    points = file1.loadPoints()
    interval = event_interval(points)
    if cutoff is None:
        plot_count_rate_histogram(interval, label=f'{int(beamline_energy)} eV')
        plt.show()
        cutoff = input('set countrate cutoff')
    low, high = split_points(points, interval, cutoff)
    calibration = file1.loadCalibration()
    return _histogram(calibration, low), _histogram(calibration, high), beamline_energy

# Old module-level names, now computed on first access (PEP 562).
_LEGACY = {
    **{f'runinfo_{k}': (lambda k=k: bts.run_table(k)) for k in bts.BEAMTIMES},
    '_RUNINFO': lambda: {k: bts.run_table(k) for k in bts.BEAMTIMES},
    '_DLD_FILES': lambda: {k: list(b.dld_files) for k, b in bts.BEAMTIMES.items()},
    'RUNINFO': lambda: RUNINFO,
    'raw_data_dir': lambda: raw_data_dir,
    '_escale_d22_file': lambda: ENERGY_SCALES / bts.energy_scale_entry('d22', 0)['file'],
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
        if 'runinfo' not in kwargs:
            raise ValueError(f'specify runinfo as one of {BEAMTIMES}')
        self.marker=bts.get(str(kwargs['runinfo']).lower()).key
        self.runinfo=bts.run_table(self.marker)
        self.run_number=run_number
        self.filename=filename
        self.spectrum=np.array([])
        self.energy_scale=np.array([])
        self.dark=np.array([])
        self.corrected_spectrum=([])
        self.binsize=1
        self.index=index
        self.excitation_energy=np.nan
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
        
    def loadspectrum(self):
        file1 = rixsaqlib.dldfile.DLDFile(self.filename, self.run_number)
        points = file1.loadPoints()

        calibration = file1.loadCalibration()
        calibration2 = file1.loadCalibration()
        calibration2.set_time_windows(_dark_time_window(self.filename, self.run_number, calibration))

        dark_points = calibration2.corrected_points(points)
        corrected_points = calibration.corrected_points(points)

        # Hack to get functions for creating spectrum
        data = rixsaqlib.dlddata.DLDData(calibration)
        escale = calibration.energy_scale

        spec = data.create_spectrum(corrected_points)
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
            energy_scale = self.energy_scale
            self.loadspectrum()
            if len(energy_scale):
                self.energy_scale = energy_scale
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
        assert not np.isnan(self.index), 'specify index first'

        self.energy_scale=bts.energy_scale(self.marker,int(self.index))
        self.energy_scale=self.energy_scale+offset

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
    specify runinfo as d22, m23, f24 or m24
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
