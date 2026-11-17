import threading
import numpy as np


class DynamicHistogram(object):
    """
    For the rixsaq systems, we need histograms to which we can add new points as they come in,
    for the lab-time/humantime/count-rate the number of bins is not known from start, and the bins has to expand
    dynamically.

    Also can keep track of an external value, of which the sum in each bin will be taken
    """
    binsize = None
    sumother = False

    def __init__(self, binsize, sumother=False):
        self.binsize = binsize
        self.sumother = sumother
        self.lock = threading.RLock()
        self.clear_histogram()

    def clear_histogram(self):
        """
        Clear and reset histogram
        """
        self.bins = np.array([])
        self.histogram = np.array([])
        if self.sumother:
            self.sumhistogram = np.array([], dtype=np.float64)

    def new_binsize(self, binsize, allpoints, sumpoints=None):
        """
        Set the binsize and recalculate the histogram
        """
        with self.lock:
            self.binsize = binsize
            self.recalculate(allpoints, sumpoints)

    def recalculate(self, allpoints, sumpoints=None):
        """
        Recalculate the histogram from the start
        """
        self.clear_histogram()
        self.add_points(allpoints, sumpoints)

    def add_points(self, newpoints, sumpoints=None):
        """
        Add new points to the histogram
        This will first add bins if needed

        If we are summing points, then do a summation of
        those points into the same bins as the points
        """
        with self.lock:
            if len(newpoints) < 1:
                return
            self._expand_bins(np.min(newpoints), np.max(newpoints))
            # Lets first sort the new points so that searchsorted doesn't
            # screw us over in the odd case that the points are not in order
            sortidxes = np.argsort(newpoints)
            newpoints = newpoints[sortidxes]
            newhist = np.zeros_like(self.histogram)
            # Now find the bin to which each poins belong
            binidxs = np.searchsorted(self.bins, newpoints, side='right')
            # Add one point at each of these bins
            np.add.at(newhist, binidxs, 1)
            self.histogram += newhist
            if self.sumother:
                # Sum the points into the correct bin
                sumpoints = sumpoints[sortidxes]
                newsumhist = np.zeros_like(self.sumhistogram)
                np.add.at(newsumhist, binidxs, sumpoints)
                self.sumhistogram += newsumhist

    def _expand_bins(self, minimum, maximum):
        """
        Expand the bins so that everything between minimum and maximum is
        also included. The max binedge is 1 step larger than the maximum,
        this is so that we have the end bin also. 2* comes from that
        arange excludes the endpoint
        """
        if len(self.bins) == 0: # We cannot take min,max if we don't already have bins
            newbins = np.arange(minimum, maximum+2*self.binsize, self.binsize)
            self.__prepend_bins(newbins)
        else:
            if minimum < np.min(self.bins):
                newbins = np.arange(minimum, np.min(self.bins), self.binsize)
                self.__prepend_bins(newbins)
            if maximum >= np.max(self.bins)-self.binsize:
                newbins = np.arange(np.max(self.bins)+self.binsize, maximum+2*self.binsize, self.binsize)
                self.__append_bins(newbins)

    def __prepend_bins(self, newbins):
        """
        Helper function that add bins to the histogram.
        Inserts the newbins, at position 0
        """
        newhist = np.zeros_like(newbins)
        self.bins = np.insert(self.bins, 0, newbins)
        self.histogram = np.insert(self.histogram, 0, newhist)
        if self.sumother:
            self.sumhistogram = np.insert(self.sumhistogram, 0, newhist)

    def __append_bins(self, newbins):
        """
        Helper function that add bins to the histogram.
        """
        newhist = np.zeros_like(newbins)
        self.bins = np.append(self.bins, newbins)
        self.histogram = np.append(self.histogram, newhist)
        if self.sumother:
            self.sumhistogram = np.append(self.sumhistogram, newhist)


if __name__=="__main__":
    p = np.array([1,2,2,3,3,3,-1,0,4,4,4,4,5,5,5,5,5])
    histogram = DynamicHistogram(1)
    histogram.add_points(p)
    print(histogram.histogram) # Should be [1. 1. 1. 2. 3. 4. 5. 0.]
    histogram.add_points(np.array([-1,0]))
    print(histogram.histogram) # Should be [2. 2. 1. 2. 3. 4. 5. 0.]
    histogram.add_points(np.array([-2]))
    print(histogram.histogram) # Should be [1, 2. 2. 1. 2. 3. 4. 5. 0.]
    histogram.add_points(np.array([6]))
    print(histogram.histogram) # Should be [1, 2. 2. 1. 2. 3. 4. 5. 1. 0]
    histogram.add_points(np.array([6]))
    print(histogram.histogram) # Should be [1, 2. 2. 1. 2. 3. 4. 5. 2. 0]
    histogram.add_points(np.array([8]))
    print(histogram.histogram) # Should be [1, 2. 2. 1. 2. 3. 4. 5. 2. 0 1 0]
    histogram.add_points(np.array([-4]))
    print(histogram.histogram) # Should be [1 0 1, 2. 2. 1. 2. 3. 4. 5. 2. 0 1 0]
    histogram.new_binsize(2, p)
    print(histogram.histogram)
    print(np.histogram(p, bins=np.arange(-1,10,2)))
    print("+++++++++++ Test for sumpoints ++++++++++++")
    histogram = DynamicHistogram(1,sumother=True)
    p = np.array([1,1,1,2,3])
    v = np.array([8,9,10,5,5])
    histogram.add_points(p,sumpoints=v)
    print(histogram.histogram)
    print(histogram.sumhistogram) # Should be [0,27,5,5]
