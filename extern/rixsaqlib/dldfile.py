import h5py
import datetime
import os
import numpy as np
import rixsaqlib.dldcalibration as dldcalibration
import rixsaqlib.dlddata as dlddata
import rixsaqlib.dldrixsmap as dldrixsmap

DLDFILEVERSION = "1.2"

# 1.0 Had x,y,t,t as points
# 1.1 Has 10 columns x, y, t, timestamp, milliseconds, start_counter, time_tag, master_rst_counter, signal1bit, extratango
#     This version also moved from mask regions [2 values], to mask squares[4 values]
# 1.2 Adding 2 more colums for 2 more extra tango data columns
NUMCOLUMNS = 12

class DLDFile:
    """
    Handles all saving and loading from hdffile.
    """
    def __init__(self, hdf5filename, datasetname, saveImages=False):
        self.hdf5filename = hdf5filename
        self.hdf5datasetname = datasetname
        self.saveimages = saveImages

    def initH5file(self, data):
        """
        Initialize a new dataset in the hdf5 file.
        First iterates the name to make sure it is unique, then creates the structure and fills it with
        data from argument data.
        """
        self.hdf5datasetname = self.iterateDatasetname(self.hdf5datasetname)
        directorypath = os.path.dirname(self.hdf5filename)
        if not os.path.exists(directorypath):
            # If there is no directory then we create the directory
            os.mkdir(directorypath)
        with h5py.File(self.hdf5filename, mode='a') as f: #This outputs error but works anyway
            nxentry = f.create_group(self.hdf5datasetname)
            nxentry.attrs['DLDFileVersion'] = DLDFILEVERSION
            nxentry.attrs["NX_class"] = 'NXentry'

            nxinstrument = nxentry.create_group('Instrument')
            nxinstrument.attrs["NX_class"] = 'NXinstrument'

            dld8080 = nxinstrument.create_group('DLD8080')
            dld8080.attrs["NX_class"] = 'NXcollection'

            dld8080data = dld8080.create_group('data')
            dld8080data.attrs["NX_class"] = 'NXcollection'

            dld8080calibdata = dld8080.create_group('calibrated_data')
            dld8080calibdata.attrs["NX_class"] = 'NXcollection'

            calib_x_spectrum = dld8080calibdata.create_dataset('calib_x_spectrum', data=data.calib_x_spectrum)
            if self.saveimages:
                calib_xy_image = dld8080calibdata.create_dataset('calib_xy_image', data=data.calib_xy_image, compression="lzf")
            points = dld8080data.create_dataset("points", (0, NUMCOLUMNS), maxshape=(None, NUMCOLUMNS), dtype='float64')
            x_spectrum = dld8080data.create_dataset("x_spectrum", data=data.x_spectrum)
            if self.saveimages:
                xy_image = dld8080data.create_dataset("xy_image", data=data.xy_image, compression="lzf")

            calibration = dld8080.create_group('calibration')
            calibration.attrs["NX_class"] = 'NXcollection'
            calibration.create_dataset("curvature_center", data=data.calibration.curvature_center)
            curvatures = calibration.create_dataset('curvatures', (0, 100), maxshape=(None, 100), dtype='float64')
            curvatures.resize(np.shape(data.calibration.curvatures_asarray()))
            curvatures[...] = data.calibration.curvatures_asarray()
            energy_scale = calibration.create_dataset('energy_scale', dtype='float64',
                                                          data=data.calibration.energy_scale)

            energy_scale_lines = calibration.create_dataset('energy_scale_lines', (0, 2), maxshape=(None, 2), dtype='float64')
            energy_scale_lines.resize(np.shape(data.calibration.energy_scale_lines_asarray()))
            energy_scale_lines[...] = data.calibration.energy_scale_lines_asarray()

            clipping_masks = calibration.create_dataset('clipping_masks', (0, 4), maxshape=(None, 4), dtype='int')
            clipping_masks.resize(np.shape(data.calibration.masks))
            clipping_masks[...] = data.calibration.masks

            time_window = calibration.create_dataset('time_window', data=data.calibration.time_window, dtype='int64')

            calibration.create_dataset("x_size", data=data.calibration.x_size)
            calibration.create_dataset("y_size", data=data.calibration.y_size)
            calibration.create_dataset("t_size", data=data.calibration.t_size)
            calibration.create_dataset("time_modulo", data=data.calibration.time_modulo)
            calibration.create_dataset("time_modulo_offset", data=data.calibration.time_modulo_offset)
            calibration.create_dataset("energy_scale_order", data=data.calibration.energy_scale_order)

            #Creating hard links for auto plotting
            nxdata = nxentry.create_group('data')
            nxdata.attrs["NX_class"] = 'NXdata'
            nxdata['calib_x_spectrum'] = calib_x_spectrum
            nxdata['calib_x_spectrum'].attrs['signal'] = 1
            nxdata['calib_x_spectrum'].attrs['axes'] = "energy_scale"
            nxdata['energy_scale'] = energy_scale

    def iterateDatasetname(self, datasetname):
        """Check if datasetname exists, generates a new name with leading zeroes datasetname000

        Args:
            datasetname (str): beginning of datasetname

        Returns
            str: uniquedatasetname with leading zeroes

        """
        leading_zeroes = 3
        iterator = 0
        newname = datasetname + str(iterator).zfill(leading_zeroes)
        if os.path.exists(self.hdf5filename):  # If file exists, open and iterate
            with h5py.File(self.hdf5filename, "r") as f:
                e = newname in f
                while e:
                    iterator += 1
                    newname = datasetname + str(iterator).zfill(leading_zeroes)
                    e = newname in f
        return newname

    def savePoints(self, newPoints):
        """This appends the points to the hdf5 file

        Args:
            newPoints (n x 3 array): points to be saved, e.g. np.array([[1, 2, 3]])
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            dld8080data = dld8080['data']
            points = dld8080data['points']
            #Resizing
            lenPoints = points.shape[0]
            lenNewPoints = newPoints.shape[0]
            newLen = lenPoints + lenNewPoints
            newShape = (newLen, NUMCOLUMNS)
            points.resize(newShape)
            #Appending
            points[lenPoints:newLen, :] = newPoints



    def saveSpectra(self, data):
        """This overwrites the spectra in the hdf5 file

        data: data where the image and spectra to be saved are
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            dld8080data = dld8080['data']

            x_spectrum = dld8080data['x_spectrum']
            x_spectrum[...] = data.x_spectrum

            dld8080calibdata = dld8080['calibrated_data']
            calib_x_spectrum = dld8080calibdata['calib_x_spectrum']
            calib_x_spectrum[...] = data.calib_x_spectrum




    def saveImages(self, data):
        """This overwrites the images in the hdf5 file

        This is slower than saving points so this is not called as often

        data: data where the image and spectra to be saved are
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            dld8080data = dld8080['data']

            xy_image = dld8080data['xy_image']
            xy_image[...] = data.xy_image

            dld8080calibdata = dld8080['calibrated_data']
            calib_xy_image = dld8080calibdata['calib_xy_image']
            calib_xy_image[...] = data.calib_xy_image


    def saveData(self, data):
        """This saves points, x_spectrum, xy_image, calib_xspectrum and calib_xyimage
           It overwrites the old data

        Args:
            data (tdcdata): the tdcdata instance to be saved

        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            dld8080data = dld8080['data']

            points = dld8080data['points']
            points.resize(np.shape(data.raw_points))
            points[...] = data.raw_points

            x_spectrum = dld8080data['x_spectrum']
            x_spectrum[...] = data.x_spectrum

            if self.saveimages:
                xy_image = dld8080data['xy_image']
                xy_image[...] = data.xy_image

            # Future: we don't need to save all data if we run uncalibrated
            dld8080calibdata = dld8080['calibrated_data']
            calib_x_spectrum = dld8080calibdata['calib_x_spectrum']
            calib_x_spectrum[...] = data.calib_x_spectrum
            if self.saveimages:
                calib_xy_image = dld8080calibdata['calib_xy_image']
                calib_xy_image[...] = data.calib_xy_image

    def apply_calibration(self, data):
        """
        Update the calibration in the dataset with the calibration used in data.
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            calibration = dld8080['calibration']
            curvature_center = calibration['curvature_center']
            curvatures = calibration['curvatures']
            energy_scale = calibration['energy_scale']
            energy_scale_lines = calibration['energy_scale_lines']
            energy_scale_order = calibration['energy_scale_order']
            clipping_masks = calibration['clipping_masks']
            time_window = calibration['time_window']
            time_modulo = calibration['time_modulo']
            time_modulo_offset = calibration['time_modulo_offset']

            curvature_center[...] = data.calibration.curvature_center
            energy_scale_order[...] = data.calibration.energy_scale_order
            energy_scale[...] = data.calibration.energy_scale
            time_window[...] = data.calibration.time_window
            time_modulo[...] = data.calibration.time_modulo
            time_modulo_offset[...] = data.calibration.time_modulo_offset

            curvatures.resize(np.shape(data.calibration.curvatures_asarray()))
            curvatures[...] = data.calibration.curvatures_asarray()

            energy_scale_lines.resize(np.shape(data.calibration.energy_scale_lines_asarray()))
            energy_scale_lines[...] = data.calibration.energy_scale_lines_asarray()

            clipping_masks.resize(np.shape(data.calibration.masks))
            clipping_masks[...] = data.calibration.masks

    def loadCalibration(self):
        """
        Loads the data file and returns a dldcalibration object

        Return:
        calibration dldcalibration
        """
        with h5py.File(self.hdf5filename, "r") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            calibration = dld8080['calibration']
            curvarray = calibration['curvatures'][:]
            energy_scale = calibration['energy_scale'][:]
            energy_linearray = calibration['energy_scale_lines'][:]
            clipping_masks = calibration['clipping_masks'][:]
            curvature_center = calibration['curvature_center'][()]
            x_size = calibration['x_size'][()]
            y_size = calibration['y_size'][()]
            try:
                t_size = calibration['t_size'][()]
            except:
                print("No t_size in dataset")
                t_size = 2**31
            try:
                energy_scale_order = calibration['energy_scale_order'][()]
            except:
                print("No energy scale order in dataset")
                energy_scale_order = 0
            try:
                time_windows = calibration['time_window'][()]
            except:
                print("No time_window in dataset")
                time_windows = np.array([0, t_size])
            try:
                time_modulo = calibration['time_modulo'][()]
                time_modulo_offset = calibration['time_modulo_offset'][()]
            except:
                print("No time_modulo in dataset")
                time_modulo = 0
                time_modulo_offset = 0

            if time_windows.ndim == 2:
                # Transitional, some old save files have a higher dimension
                # This was removed.
                print("Warning! Time windows clipped to first window")
                time_windows = time_windows[0,:]

            loaded_calibration = dldcalibration.DLDCalibration(x_size, y_size, t_size)
            loaded_calibration.energy_scale = energy_scale
            loaded_calibration.set_curvature_center(curvature_center)
            loaded_calibration.set_curvatures_fromarray(curvarray)
            loaded_calibration.set_energylist_fromarray(energy_linearray)
            loaded_calibration.set_energy_scale_order(energy_scale_order)
            loaded_calibration.set_time_windows(time_windows)
            loaded_calibration.masks = clipping_masks
            loaded_calibration.set_time_modulo(time_modulo)
            loaded_calibration.set_time_modulo_offset(time_modulo_offset)
        return loaded_calibration

    def loadPoints(self):
        """
        Load points from file

        Returns: Numpy array with x,y,t,timestamp per point

        """
        with h5py.File(self.hdf5filename, "r") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            dld8080data = dld8080['data']
            points = dld8080data['points'][:]
            if points.shape[1] < 4:
                # Old savefile without timestamp
                print("No timestamp in points, using zeroes")
                points = np.hstack((points, np.zeros((points.shape[0], 1), dtype=points.dtype)))
        return points

    def loadSpectrum(self):
        """
        Loads calibration and spectrum and calibrated spectrum

        Returns: dlddata with spectra

        """
        calibration = self.loadCalibration()
        with h5py.File(self.hdf5filename, "r") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            dld8080data = dld8080['data']

            x_spectrum = dld8080data['x_spectrum'][:]
            dld8080calibdata = dld8080['calibrated_data']
            calib_x_spectrum = dld8080calibdata['calib_x_spectrum'][:]

            data = dlddata.DLDData(calibration)
            data.x_spectrum = x_spectrum
            data.calib_x_spectrum = calib_x_spectrum
        return data


    def loadRixsmap(self, startdataset):
        """
        A simple load of rixsmap
        """
        rixsmap = dldrixsmap.DLDRixsMap()
        entrynumber = int(startdataset.lstrip("acq"))
        stepnumber = 0
        loading = True
        self.hdf5datasetname = startdataset
        while loading:
            print("Loading Entry: ", self.hdf5datasetname, entrynumber)
            with h5py.File(self.hdf5filename, "r") as f:
                incidentenergy = f["{}/Instrument/DLD8080/data/points".format(self.hdf5datasetname)][0,-1]
            data = self.loadSpectrum()
            rixsmap.add_spectrum(data.calib_x_spectrum, stepnumber, incidentenergy, data.calibration.energy_scale)
            entrynumber += 1
            stepnumber += 1
            leading_zeroes = 3
            newdataset = "acq" + str(entrynumber).zfill(leading_zeroes)
            self.hdf5datasetname = newdataset
            newstep = -1
            with h5py.File(self.hdf5filename, "r") as f:
                if "{}/rixsmap_step".format(newdataset) in f:
                    newstep = f["{}/rixsmap_step".format(newdataset)][()]
            if newstep != stepnumber:
                loading = False
        return rixsmap


    def loadRixsmapAsPoints(self, startdataset):
        """
        Load all the points from a rixsmap
        """
        entrynumber = int(startdataset.lstrip("acq"))
        stepnumber = 0
        loading = True
        self.hdf5datasetname = startdataset
        points = self.loadPoints()
        while loading:
            entrynumber += 1
            stepnumber += 1
            newstep = -1
            newdataset = "acq" + str(entrynumber).zfill(3)
            self.hdf5datasetname = newdataset
            with h5py.File(self.hdf5filename, "r") as f:
                if "{}/rixsmap_step".format(newdataset) in f:
                    newstep = f["{}/rixsmap_step".format(newdataset)][()]
            if newstep != stepnumber:
                loading = False
                return points
            print("Loading Entry: ", self.hdf5datasetname)
            morepoints = self.loadPoints()
            points = np.vstack(([points, morepoints]))
        # This is not reachable
        return points


    def listdatasets(self, onlyRixsmaps=False):
        """
        List datasets in current file
        Returns: list of dictionary of datasets
                 the dictionary contains name and some info

        """
        with h5py.File(self.hdf5filename, "r") as f:
            datasets = []
            for dataset in f:
                infodict = {"name": dataset}
                try:
                    if "{}/rixsmap_step".format(dataset) in f:
                        infodict["rixsmap_step"] = f["{}/rixsmap_step".format(dataset)][()]
                    if "{}/start_time".format(dataset) in f:
                        infodict["start_time"] = f["{}/start_time".format(dataset)][()]#.value
                    if "{}/end_time".format(dataset) in f:
                        infodict["end_time"] = f["{}/end_time".format(dataset)][()]#.value
                    if "{}/Instrument/DLD8080/data/points".format(dataset) in f:
                        infodict["numpoints"] = f["{}/Instrument/DLD8080/data/points".format(dataset)].shape[0]
                    if "{}/Instrument/DLD8080/calibrated_data/calib_x_spectrum".format(dataset) in f:
                        infodict["calib_x_spectrum"] = f["{}/Instrument/DLD8080/calibrated_data/calib_x_spectrum".format(dataset)][()]
                    if "{}/Instrument/DLD8080/calibration/energy_scale".format(dataset) in f:
                        infodict["energy_scale"] = f["{}/Instrument/DLD8080/calibration/energy_scale".format(dataset)][()]
                    if "{}/startmetadata/veritas_beamline_energy/Position".format(dataset) in f:
                        infodict["veritas_beamline_energy"] = f["{}/startmetadata/veritas_beamline_energy/Position".format(dataset)][()]
                except Exception as e:
                    print(str(e))
                if not onlyRixsmaps:
                    datasets.append(infodict)
                else:
                    if "rixsmap_step" in infodict:
                        if infodict["rixsmap_step"] == 0:
                            # only list the first step in a rixsmap
                            datasets.append(infodict)
        return datasets

    def writeExposureTime(self, exposuretime):
        """
        Write the exposure time to the dataset in hdf5 file
        If there already is an exposure time, iterate the name.
        This shows that multiple exposures has been in the dataset.
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            nxinstrument = nxentry['Instrument']
            dld8080 = nxinstrument['DLD8080']
            if not 'exposure_time' in dld8080:
                dld8080.create_dataset("exposure_time", data=exposuretime)
            else:   #It exists then iterate and add another
                iterator = 2
                newname = 'exposure_time' + str(iterator)
                e = newname in dld8080
                while e:
                    iterator += 1
                    newname = 'exposure_time' + str(iterator)
                    e = newname in dld8080
                dld8080.create_dataset(newname, data=exposuretime)

    def writeStartTime(self):
        """
        Write start time (now) into dataset
        """
        now = datetime.datetime.now()
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            if not 'start_time' in nxentry:
                nxentry.create_dataset("start_time", data=str(now))

    def writeEndTime(self):
        """
        Write end time (now) into dataset
        """
        now = datetime.datetime.now()
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            if 'end_time' in nxentry:  # Exists, update
                end_time = nxentry['end_time']
                end_time[...] = str(now)
            else:
                nxentry.create_dataset("end_time", data=str(now))


    def writeRixsMapStep(self, stepnr):
        """
        Write the stepnumber of the rixsmap in the h5file
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            if not 'rixsmap_step' in nxentry:
                nxentry.create_dataset("rixsmap_step", data=int(stepnr))


    def writeStartMetaData(self, metadataobjects):
        """
        Write starting metadata into dataset.
        Input parameter metadata is a list of DLDMetaData objects
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            for metadata in metadataobjects:
                tangoattr = metadata.tangoattribute
                value = metadata.value
                nxpath = metadata.nxpath
                if value is None:
                    value = "None"
                if nxpath is not None:
                    metadataentry = nxentry.create_dataset(nxpath, data=value)
                    metadataentry.attrs["TangoAttribute"] = tangoattr
                if nxpath is None:
                    # If the is no path, then put it in a non-organized pile
                    if not "startmetadata" in nxentry:
                        startmetadata = nxentry.create_group("startmetadata")
                    nxentry["startmetadata"].create_dataset(tangoattr, data=value)

    def writeEndMetaData(self, metadataobjects):
        """
        Write starting metadata into dataset.
        Input parameter metadata is a list of DLDMetaData objects
        # TODO Right now this is ignoring any proper structure, and just putting
        # it in endmetadata, anyway thats how we are working with the data
        # Not very future proof though
        """
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f[self.hdf5datasetname]
            for metadata in metadataobjects:
                tangoattr = metadata.tangoattribute
                value = metadata.value
                nxpath = metadata.nxpath
                if value is None:
                    value = "None"
                # TODO fix
                #if nxpath is not None:
                #    metadataentry = nxentry.create_dataset(nxpath, data=value)
                #    metadataentry.attrs["TangoAttribute"] = tangoattr
                #if nxpath is None:
                #    # If the is no path, then put it in a non-organized pile
                if not "endmetadata" in nxentry:
                    endmetadata = nxentry.create_group("endmetadata")
                nxentry["endmetadata"].create_dataset(tangoattr, data=value)


    def exportCalibration(self, data):
        """
        Export the calibration used in data to a new dataset
        """
        self.hdf5datasetname = self.iterateDatasetname(self.hdf5datasetname)
        with h5py.File(self.hdf5filename, "a") as f:
            nxentry = f.create_group(self.hdf5datasetname)
            nxentry.attrs["NX_class"] = 'NXentry'

            nxinstrument = nxentry.create_group('Instrument')
            nxinstrument.attrs["NX_class"] = 'NXinstrument'

            dld8080 = nxinstrument.create_group('DLD8080')
            dld8080.attrs["NX_class"] = 'NXcollection'

            calibration = dld8080.create_group('calibration')
            calibration.attrs["NX_class"] = 'NXcollection'
            curvatures = calibration.create_dataset('curvatures', (0, 100), maxshape=(None, 100))
            calibration.create_dataset("curvature_center", data=data.calibration.curvature_center)
            curvatures.resize(np.shape(data.calibration.curvatures_asarray()))
            curvatures[...] = data.calibration.curvatures_asarray()
            energy_scale = calibration.create_dataset('energy_scale', dtype='float64',
                                                          data=data.calibration.energy_scale)
            calibration.create_dataset("x_size", data=data.calibration.x_size)
            calibration.create_dataset("y_size", data=data.calibration.y_size)
            calibration.create_dataset("t_size", data=data.calibration.t_size)
            calibration.create_dataset("energy_scale_order", data=data.calibration.energy_scale_order)

            energy_scale_lines = calibration.create_dataset('energy_scale_lines', (0, 2), maxshape=(None, 2), dtype='float64')
            energy_scale_lines.resize(np.shape(data.calibration.energy_scale_lines_asarray()))
            energy_scale_lines[...] = data.calibration.energy_scale_lines_asarray()

            clipping_masks = calibration.create_dataset('clipping_masks', (0, 4), maxshape=(None, 4), dtype='int')
            clipping_masks.resize(np.shape(data.calibration.masks))
            clipping_masks[...] = data.calibration.masks

            time_window = calibration.create_dataset('time_window', data=data.calibration.time_window, dtype='int64')

            calibration.create_dataset("time_modulo", data=data.calibration.time_modulo)
            calibration.create_dataset("time_modulo_offset", data=data.calibration.time_modulo_offset)


class DLDMetaData(object):
    """
    Small helper class to keep track of metadataobjects
    """
    def __init__(self, tangoattribute, value, nxpath = None):
        self.tangoattribute = tangoattribute
        self.value = value
        self.nxpath = nxpath


def ExportAsciiPoints(filename, data, datasetname):
    """
    Export the points as a raw ascii txt file

    Args:
        filename: output filename
        data: data to save
        datasetname: name in txt header

    Returns:

    """
    np.savetxt(filename, data.raw_points, delimiter="\t", header="RAW POINTS FROM "+str(datasetname)+". X\tY\tT\tTimestamp")


def ExportAsciiPointsCalib(filename, data, datasetname):
    """
    Export the calibrated points as a raw ascii txt file

    Args:
        filename: output filename
        data: data to save
        datasetname: name in txt header

    Returns:

    """
    c_points = data.calibration.corrected_points(data.raw_points)
    np.savetxt(filename, c_points, delimiter="\t", header="POINTS FROM "+str(datasetname)+". X\tY\tT\tTimestamp")



def ExportAsciiSpectra(filename, data, datasetname):
    """
    Export the energyscale, calibrated spectrum and raw spectrum to an ascii txt file

    Args:
        filename: output filename
        data: data to save
        datasetname: name in txt header

    Returns:

    """
    out = np.vstack([data.calibration.energy_scale, data.calib_x_spectrum, data.x_spectrum]).T
    np.savetxt(filename, out, delimiter="\t", header=str(datasetname)+" ENERGYSCALE\tCALIBRATED-SPECTRUM\tRAWSPECTRUM")
