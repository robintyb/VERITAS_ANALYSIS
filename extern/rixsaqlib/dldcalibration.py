import numpy as np
import json
import logging

class DLDCalibration:
    """Stores and finds all calibration curves and energy scale

    Attributes:
        curvatures         The polynomials for curve correction list of np polynomials
        curvature_center   Center line of curvature, this defines how far the points will be moved
        energy_scale       The calibrated energy_scale, np array with len x_size
        masks              Clipping masks, 2 element arrays of min max y-values. The points in the masks will be removed.
        time_modulo        The modulo that will be applied to the t-value
        time_modulo_offset Offset that will be added after doing the time_modulo
        time_windows       Windows that is used to select a part of t-values


    """

    def __init__(self, x_size, y_size, t_size, x_imgbinsize=1, y_imgbinsize=1):
        self.x_size = x_size
        self.y_size = y_size
        self.t_size = t_size
        # Try to only use binsize other than 1 in the client
        self.x_imgbinsize = x_imgbinsize # Make it possible to bin the images
        self.y_imgbinsize = y_imgbinsize # so that they don't use unecessary space in memory

        self.curvatures = []
        self.curvature_center = y_size/2
        self.energy_scale_order = 1
        self.energy_scale_lines = []
        self.energy_scale = np.arange(0, x_size)
        self.masks = np.array([[]], dtype=np.int32)
        self.time_modulo = 0
        self.time_modulo_offset = 0
        self.time_window = [0, t_size]

        # Only for display currently
        self.time_modulo_binsize = 1000



    def to_json(self):
        class MyEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                else:
                    return super(MyEncoder, self).default(obj)

        calibrationdict = {"x_size": self.x_size,
                           "y_size": self.y_size,
                           "t_size": self.t_size,
                           "curvatures": self.curvatures_asarray().tolist(),
                           "curvature_center": self.curvature_center,
                           "energy_scale_order": self.energy_scale_order,
                           "energy_scale_lines": self.energy_scale_lines_asarray().tolist(),
                           #"energy_scale": self.energy_scale.tolist(),
                           "masks": self.masks.tolist(),
                           "time_modulo": self.time_modulo,
                           "time_modulo_offset": self.time_modulo_offset,
                           "time_window": self.time_window,
                           "time_modulo_binsize": self.time_modulo_binsize}
        jsonstr = json.dumps(calibrationdict, cls=MyEncoder)
        return jsonstr

    def from_json(self, jsnstr):
        """
        Convert calibration json to object.
        """
        if jsnstr is None or jsnstr == "":
            logging.info("Got NONE as a jsonstr")
            return
        calibrationdict = json.loads(jsnstr)
        x_size = calibrationdict["x_size"]
        y_size = calibrationdict["y_size"]
        t_size = calibrationdict["t_size"]
        curvatures = np.asarray(calibrationdict["curvatures"])
        curvature_center = calibrationdict["curvature_center"]
        energy_scale_order = calibrationdict["energy_scale_order"]
        energy_scale_lines = np.asarray(calibrationdict["energy_scale_lines"])
        #energy_scale = np.asarray(calibrationdict["energy_scale"])
        masks = np.asarray(calibrationdict["masks"])
        time_modulo = calibrationdict["time_modulo"]
        time_modulo_offset = calibrationdict["time_modulo_offset"]
        time_window = calibrationdict["time_window"]
        time_modulo_binsize = calibrationdict["time_modulo_binsize"]
        self.x_size = x_size
        self.x_size = x_size
        self.y_size = y_size
        self.t_size = t_size
        self.set_curvatures_fromarray(curvatures)
        self.curvature_center = curvature_center
        self.energy_scale_order = energy_scale_order
        self.set_energylist_fromarray(energy_scale_lines)
        self.energy_scale = self.calculate_energy_scale(energy_scale_lines, self.energy_scale_order)
        self.masks = masks
        self.time_modulo = time_modulo
        self.time_modulo_offset = time_modulo_offset
        self.time_modulo_binsize = time_modulo_binsize
        self.time_window = time_window

    def add_curv_poly(self, poly):
        """
        Add curvature poly to curvature list
        poly is a np array.
        """
        self.curvatures.append(np.copy(poly))

    def remove_curv_poly(self, idx):
        """
        Remove curvature with idx
        """
        del self.curvatures[idx]

    def clear_curvatures(self):
        """
        Clear all curvatures
        """
        self.curvatures = []

    def curvatures_asarray(self):
        """
        Takes the list self.curvatures, fills with zeroes to make the equal length and
        :return: self.curvatures as an ndarray
        """
        if len(self.curvatures) < 1: return np.array([[]])
        length = len(sorted(self.curvatures,key=len, reverse=True)[0])
        return np.array([[0]*(length-len(xi)) + xi.tolist() for xi in self.curvatures])

    def set_curvatures(self, curvlist):
        """
        Set the curvatures from list
        """
        self.curvatures = curvlist

    def set_time_modulo(self, newmodulo):
        """
        Set the time modulo to newmodulo
        """
        self.time_modulo = newmodulo

    def set_time_modulo_binsize(self, newbinsize):
        """
        Set the binsize to timegating
        """
        self.time_modulo_binsize = newbinsize

    def set_time_modulo_offset(self, newmodulooffset):
        """
        Set the time modulo offset to newmodulooffset
        """
        self.time_modulo_offset = newmodulooffset

    def set_curvatures_fromarray(self, curvarray):
        """
        Convert array of curvatures into list
        """
        curvatures = []
        if curvarray is None:
            self.curvatures = curvatures
            return
        if curvarray.size == 0: #ugly hack when loading empty hdf5
            self.curvatures = curvatures
            return
        for curv in curvarray:
            curvatures.append(np.trim_zeros(curv, trim='f'))
        self.curvatures = curvatures

    def set_curvature_center(self, val):
        """
        Set the center line of the curvatures to value
        """
        self.curvature_center = val

    def set_energy_scale_order(self, neworder):
        """
        Sets the order of the energy scale used
        """
        self.energy_scale_order = neworder

    def set_energy_scale(self, energy_scale):
        """
        Set the energy scale to energy scale.
        energy scale is a numpy array with the same len as x-spectrum
        """
        self.energy_scale = np.copy(energy_scale)

    def set_energylist(self, linelist):
        """
        Set energy scale lines from list
        """
        self.energy_scale_lines = linelist

    def set_energylist_fromarray(self, linearray):
        """
        Convert the energy scale lines from array to list
        """
        energy_scale_lines = []
        if linearray is None:
            self.energy_scale_lines = energy_scale_lines
            return
        if linearray.size == 0: #ugly hack when loading empty hdf5
            self.energy_scale_lines = energy_scale_lines
            return
        if len(linearray) == 0:
            self.energy_scale_lines = energy_scale_lines
            return
        self.energy_scale_lines = linearray.tolist()

    def energy_scale_lines_asarray(self):
        """
        Returns the energyscale lines as an array
        """
        if len(self.energy_scale_lines) < 1: return np.array([[]])
        return np.array(self.energy_scale_lines)

    def add_energy_line(self, pos, energy):
        """
        Add energy scale line at position with energy
        """
        self.energy_scale_lines.append(np.array([pos, energy]))

    def remove_energy_line(self, idx):
        """
        Remove energy scale leine with index
        """
        del self.energy_scale_lines[idx]

    def clear_energy_scale_lines(self):
        """
        Clear all energy scale lines
        """
        self.energy_scale_lines = []

    def set_masks(self, masks):
        """
        Set clipping masks to the masks in masks
        """
        if masks is None:
            self.clear_masks()
        else:
            self.masks = masks

    def add_mask(self, mask):
        """
        Add clipping mask. mask is a 4 element array containing min and max in x and y of clipping mask
        y1,y2,x1,x2
        """
        if self.masks.size < 1:
            self.masks = np.empty([0,4], dtype=np.int32)
        if len(mask) == 2: # Backwards compatbility for masks that where only in y
            mask = np.array([mask[0],mask[1],0,self.x_size])
        self.masks = np.vstack([self.masks, mask])

    def remove_mask(self, rowidx):
        """
        Remove clipping mask with index
        """
        self.masks = np.delete(self.masks, (rowidx), axis=0)
        if self.masks.size < 1:
            self.clear_masks()

    def clear_masks(self):
        """
        Clear the clipping masks in array
        """
        self.masks = np.array([[]])

    def masked_points(self, points):
        """
        Removes points in the clipping mask.
        Beware! This doesn't copy the array
        """
        if self.masks.size < 1: return points
        for mask in self.masks.tolist():
            if len(mask) == 2: # Backwards compatbility for masks that where only in y
                mask = np.array([mask[0],mask[1],0,self.x_size])
            maskidxx = (points[:,1] >= mask[0]) & (points[:,1] < mask[1])
            maskidxy = (points[:,0] >= mask[2]) & (points[:,0] < mask[3])
            maskidx = maskidxx & maskidxy
            points = points[~maskidx]
        return points

    def set_time_windows(self, newwindows):
        """
        Set the time window to timewindows
        """
        if isinstance(newwindows, (np.ndarray, np.generic) ):
            if newwindows.ndim > 1:
                raise Exception("Only one window is allwed")
            newwindows = newwindows.tolist()
        self.time_window = newwindows

    def time_windowed_points(self, points):
        """
        Removes points outside timewindow.
        Beware! This doesn't copy the array
        """
        if len(points) < 1:
             return points
        expr = np.zeros(len(points[:,2]), dtype=bool)
        expr = np.logical_or(expr, (points[:,2] >= self.time_window[0]) & (points[:,2] < self.time_window[1]))
        points = points[expr]
        return points

    def time_modulo_points(self, points):
        """
        Folds the time coordinates, use this before the time window,
        If zero use no modulo
        Beware! This doesn't copy the array
        """
        if self.time_modulo > 0:
            points[:,2] = np.mod(points[:,2]+self.time_modulo_offset, self.time_modulo)
        return points

    def timestamp_windowed_points(self, points, timestamp_window):
        """
        Removes points outside timestamp window
        Beware! This doesn't copy the array
        """
        points = points[(points[:,3] >= timestamp_window[0]) & (points[:,3] < timestamp_window[1])]
        return points

    def curvature_corrected_points(self, points):
        """
        Curvature correction. Make sure that lines on polynomial curvature becomes straight.
        When using mulitple curvatures interpolate the correction between them.
                +         XX
                |       XX
                |     XX
                |   XX
                | XX
                |X
                |X
                |X
                | XX
                |   XX
                |     XX
                |       XX
                +         XX
                +---xcorr----+
        Curvxcenter is the centerpoint for the curvature, points on the curvature will be corrected
        to this line. The offset xcorr is the amount the points on the curvature needs to be moved.
        Points between curvatures will use an interpolation of two xcorr:s.
        """
        if len(points) < 1:
            return points
        if len(self.curvatures) < 1:
            return points
        curvatures = self.curvatures
        x = np.transpose(points[:,0])
        y = np.transpose(points[:,1])

        curvx = np.empty([len(curvatures),len(x)])
        curvxcorr = np.empty([len(curvatures),len(x)])

        for i in range(0,len(curvatures)):
            curv = curvatures[i]
            curvcenterx = np.polyval(curv, self.curvature_center)
            thiscurvx   = np.polyval(curv, y)
            curvx[i,:]  = thiscurvx
            xcorr       = np.copy(thiscurvx)-curvcenterx
            curvxcorr[i,:] = (xcorr)

        # Sort to avoid nonsense in interpolation
        # The lines must not cross
        sortidx   = np.argsort(curvx[:,0])
        curvx     = curvx[sortidx]
        curvxcorr = curvxcorr[sortidx]

        # Make linear interpolation for all curvatures
        # Interpolation index as shown, where the lines are the curvatures
        #      +       +       +
        #   0  |   1   |   2   |  3
        #      +       +       +
        #
        # Curv 0       1       2
        interpxcorr = np.empty([len(curvatures)+1,len(x)])
        interpxcorr[0] = curvxcorr[0]
        for i in np.arange(len(curvatures)-1):
            interp = curvxcorr[i] + (x-curvx[i])/(curvx[i+1]-curvx[i])*(curvxcorr[i+1]-curvxcorr[i])
            interpxcorr[i+1] = interp
        interpxcorr[-1] = curvxcorr[-1]

        # Find which interpolation to use
        # This method finds the nearest curvature for every point
        # It then checks if the point is to the right side of that curve
        mindist = None
        idx = np.zeros(len(x), dtype=int)
        for i in np.arange(len(curvatures)):
            dist = np.square(curvx[i]-x)
            if mindist is None: mindist = dist
            cmp = (dist<=mindist)
            mindist[cmp] = dist[cmp]
            idx[cmp]  = i
            cmp2 = np.logical_and((x-curvx[i]) > 0,(cmp)) #See if higher Place Right
            idx[cmp2] = i+1

        x_corr = idx.choose(interpxcorr).astype(int)

        # Apply the curvature correction, points are moved according to x_corr
        points[:,0] -= x_corr
        points[(points[:,0] >= self.x_size),0] = self.x_size - 1
        points[(points[:,0] < 0),0] = 0
        return points

    def corrected_points(self, points, skip_timewindow=False, skip_curvature=False):
        """
        Takes raw points as input and corrects the according to the curvatures.
        Note that timestamp window is not used here

        :param points:
        :return: corrected points
        """
        # Remember to copy, to not destroy raw points
        correct_points = np.copy(points)

        # Fold with time modulo
        correct_points = self.time_modulo_points(correct_points)

        # Apply time window
        if not skip_timewindow:
            correct_points = self.time_windowed_points(correct_points)

        # Correct for curvature
        if not skip_curvature:
            correct_points = self.curvature_corrected_points(correct_points)

        # Apply clipping mask
        correct_points = self.masked_points(correct_points)

        return correct_points

    def calculate_energy_scale(self, positions, order=1):
        """
        Calculates an energy scale from the input positions.
        Positions is channel numbers and scale value
        :param positions:
        :return: an energy scale from 0 to x_size fitted to the positions
        """
        energy_scale = np.arange(0, self.x_size)
        if positions.size > 0:
            poly = np.polyfit(positions[:, 0], positions[:, 1], order)
            energy_scale = np.polyval(poly, energy_scale)
        return energy_scale

    def channel_to_energy(self, channel):
        """
        Takes channel and checks it with energy_scale, if it exists return
        the energy, otherwise return 0
        Return: energy
        """
        if channel > self.x_size-1 or channel < 0:
            return 0
        else:
            return self.energy_scale[int(channel)]

    def energy_to_channel(self, energy):
        """
        Takes energy and return channel number.
        Will return idx of nearest value, which will correspond to the channel number.
        """
        return self.find_nearest(self.energy_scale, energy)


    @staticmethod
    def find_nearest(array,value):
        """
        Returns idx of nearest value to values in array
        """
        idx = (np.abs(array-value)).argmin()
        return idx
