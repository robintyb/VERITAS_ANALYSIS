import numpy as np
import threading
from rixsaqlib.dynamichistogram import DynamicHistogram
from rixsaqlib.waterfalldatastructure import WaterfallData

NUMCOLUMNS = 12

class DLDData:
    """Stores and handles all relevant data for a measurement

    Attributes:
        raw_points         [x-coordinate, y-coordinate, t-coordinate, timestamp]
        x_spectrum         The current x-spectrum
        xy_image           The xy-image of the experiment
        calib_x_spectrum   Calibrated x spectrum
        calib_xy_imagage   Calibrated xy image
        total_points
        counts_per_second  Calculated in dldlib
        start_time         To measure the time elapsed for measurement
    """
    def __init__(self, calibration, only_pointdata=False, makecountratehistogram=False, maketimegatehistogram=False, makewaterfall=False):
        self.calibration = calibration
        self.makecountratehistogram = makecountratehistogram
        self.maketimegatehistogram = maketimegatehistogram
        self.makewaterfall = makewaterfall
        self.raw_points = np.zeros((0, NUMCOLUMNS), dtype=np.float64)
        self.lock = threading.RLock()

        if not only_pointdata:
            # Don't create images and spectra if not needed, they use lots of memory
            self.x_spectrum = np.zeros(self.calibration.x_size)
            self.xy_image = np.zeros((int(self.calibration.y_size/self.calibration.y_imgbinsize), int(self.calibration.x_size/self.calibration.x_imgbinsize)))

            self.calib_x_spectrum = np.zeros(self.calibration.x_size)
            self.calib_xy_image = np.zeros((int(self.calibration.y_size/self.calibration.y_imgbinsize), int(self.calibration.x_size/self.calibration.x_imgbinsize)))

        if self.makecountratehistogram:
            self.countratehistogram = DynamicHistogram(30, sumother=True)
        if self.maketimegatehistogram:
            self.timegatehistogram = DynamicHistogram(calibration.time_modulo_binsize)
        if self.makewaterfall:
            self.waterfall = WaterfallData(60, 1, 3, self.calibration.x_size)

        self.total_points = 0
        self.total_points_calib = 0
        self.start_time = 0

    def add_points(self, points):
        """
        Add points to raw_points and images and spectra
        :param points: to be added
        """
        if len(points) < 1:
            return
        with self.lock:
            numcolumnsnewpoints = points.shape[1]
            if numcolumnsnewpoints < NUMCOLUMNS: # This is old points, add 0s in the new columns
                extracolumns = np.zeros((points.shape[0],NUMCOLUMNS-numcolumnsnewpoints))
                points = np.hstack([points, extracolumns])
            # Hack because we cannot currently send all columns in tango.
            # Currently we only send columns 0,1,2,3,9
            # So move idx 4 to idx 9
            if numcolumnsnewpoints == 5:
                points[:,9] = points[:,4]
                points[:,4] = np.zeros_like(points[:,4])
            self.raw_points = np.vstack(([self.raw_points, points]))  # Performance?
            self._add_points_to_image(points, self.xy_image, self.calibration.x_imgbinsize, self.calibration.y_imgbinsize)
            self._add_points_to_spectrum(points, self.x_spectrum)

            c_points = self.calibration.corrected_points(points)
            self._add_points_to_image(c_points, self.calib_xy_image, self.calibration.x_imgbinsize, self.calibration.y_imgbinsize)
            self._add_points_to_spectrum(c_points, self.calib_x_spectrum)

            if self.makecountratehistogram:
                self.countratehistogram.add_points(c_points[:,3],sumpoints=c_points[:,9])
            if self.maketimegatehistogram:
                c_points_notimewindow = self.calibration.corrected_points(points, skip_timewindow=True)
                self.timegatehistogram.add_points(c_points_notimewindow[:,2])
            if self.makewaterfall:
                self.waterfall.add_points(c_points, self.calibration.energy_scale)

            self.total_points += len(points)
            self.total_points_calib += len(c_points)

    @staticmethod
    def _add_points_to_image(points, image, x_binsize, y_binsize):
        """
        Adds the points to image, note this works in same memory and doesn't copy
        When we get numpy > 1.8.0 use
        np.add.at(image, (points[:,1],points[:,0]),1)
        """
        y = points[:,1]/y_binsize
        x = points[:,0]/x_binsize
        np.add.at(image, (y.astype(int),x.astype(int)),1)

    @staticmethod
    def _add_points_to_spectrum(points, spectrum):
        """
        Adds the points to image, note this works in same memory and doesn't copy
        When we get numpy > 1.8.0 use
        np.add.at(spectrum, points[:,0],1)
        """
        #for p in points:
        #    spectrum[p[0]] += 1
        np.add.at(spectrum, points[:,0].astype(int),1)

    def apply_calibration(self, newcalibration):
        """
        Updates the data with a new calibration
        """
        with self.lock:
            self.calibration = newcalibration
            self.refresh_calibration()

    def create_spectrum(self, points):
        """
        Creates and empty spectra the size of calibration.x_size
        by creating an empty spectra and then putting all the points in the correct place
        """
        spectrum = np.zeros(self.calibration.x_size)
        self._add_points_to_spectrum(points, spectrum)
        return spectrum

    def create_image(self, points, unbinned=False):
        """
        Creates and image by creating an empty 2d array the shape of x_size and y_size and then
        adds the points to the correct place
        """
        if unbinned: #used by curvaturetab
            image = np.zeros((self.calibration.y_size, self.calibration.x_size))
            self._add_points_to_image(points, image, 1, 1)
        else:
            image = np.zeros((int(self.calibration.y_size/self.calibration.y_imgbinsize), int(self.calibration.x_size/self.calibration.x_imgbinsize)))
            self._add_points_to_image(points, image, self.calibration.x_imgbinsize, self.calibration.y_imgbinsize)
        return image


    def set_new_imgbinsize(self, xbinsize, ybinsize):
        """
        Set the new img binsizes and recalculate the images
        """
        with self.lock:
            self.calibration.x_imgbinsize = xbinsize
            self.calibration.y_imgbinsize = ybinsize
            self.xy_image = self.create_image(self.raw_points)
            self.refresh_calibration()

    def refresh_calibration(self):
        """
        Refreshes the calibrated images and spectra data
        """
        if self.raw_points.size == 0:
            return
        with self.lock:
            c_points = self.calibration.corrected_points(self.raw_points)
            self.total_points_calib = c_points.shape[0]
            calib_xy_image = self.create_image(c_points)
            calib_x_spectrum = self.create_spectrum(c_points)
            if self.makecountratehistogram:
                self.countratehistogram.recalculate(c_points[:,3],sumpoints=c_points[:,9])
            if self.maketimegatehistogram:
                c_points_notimewindow = self.calibration.corrected_points(self.raw_points, skip_timewindow=True)
                self.timegatehistogram.recalculate(c_points_notimewindow[:,2])
            if self.makewaterfall:
                self.waterfall.recalculate(c_points, self.calibration.energy_scale)
            self.calib_xy_image = calib_xy_image
            self.calib_x_spectrum = calib_x_spectrum
