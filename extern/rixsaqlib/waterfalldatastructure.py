import threading
import numpy as np

class WaterfallData(object):
    """
    Contains the datastructure used to create the waterfall plots.
    This will be much like the dynamic histogram, however we will instead be 
    keeping a spectrum in each bin.
    We will keep all possible y-axes/binlabels as a property,
    Parameters: t_binsize, size in time of each line in the waterfall
                binidx, the index of the parameter from which we create the bins, most used would be the unixtimestamp
                            but this could of course be other things.
                x_size, the size of the spectrum, i.e. 8192 or 4096
    """
    binlabeltitles = ["Time","RixsmapP."]
    binlabelpointsidx = [3, 9] # The idx of the parameter in the pointslist.
    numbinlabels = len(binlabeltitles)

    def __init__(self, t_binsize, e_binsize, binidx, x_size):
        self.t_binsize = t_binsize
        self.e_binsize = e_binsize
        self.elossmap = False
        self.binidx = binidx
        self.elossbinidx = 9 # The column to use when creating an elossmap
        self.x_size = x_size
        self.lock = threading.RLock()
        self.clear_waterfall()

    def clear_waterfall(self):
        """
        Clear and reset the waterfall
        """
        with self.lock:
            self.bins = np.array([])
            self.binlabels = np.empty((0,self.numbinlabels))
            # We want the label to be the mean in each bin, so
            # we need to keep track of the sum and the number of points
            # In each bin and, and then we need to update it as needed...
            self.binlabelsums = np.empty((0,self.numbinlabels))
            self.binlabelnums = np.empty((0,self.numbinlabels))
            self.waterfall = np.empty((0,int(self.x_size/self.e_binsize)))
            self.fitresults = np.array([], dtype=object)
            self.fitplots = None
            self.fitplotsstderr = None
            self.fitparamlabels = []

    def new_binsize(self, binsize, e_binsize, allpoints, eloss, energyscale):
        """
        Set the binsize and recalculate the waterfall
        """
        with self.lock:
            self.elossmap = eloss
            self.t_binsize = binsize
            self.e_binsize = e_binsize
            self.recalculate(allpoints, energyscale)

    def recalculate(self, allpoints, energyscale):
        """
        Recalculate the waterfall from the start
        """
        self.clear_waterfall()
        self.add_points(allpoints, energyscale)

    def add_points(self, newpoints, energyscale):
        """
        Add new points to the waterfall
        This will first add bins if needed
        """
        with self.lock:
            if len(newpoints) < 1:
                return
            t_min, t_max = np.min(newpoints[:,self.binidx]), np.max(newpoints[:,self.binidx])
            self._expand_bins(t_min, t_max)
            ###import time
            ###starttime = time.time()
            """
            # Works but naive and slow, optimized below
            for p in newpoints:
                bin = np.searchsorted(self.bins, p[self.binidx], side='right')
                np.add.at(self.waterfall[bin], p[0].astype(int),1)
            """
            # Lets first sort the new points so that searchsorted doesn't
            # screw us over in the odd case that the points are not in order
            sortidxes = np.argsort(newpoints[:,self.binidx])
            sortedpoints = newpoints[sortidxes]
            #print("Sorted points:", sortedpoints)
            ## Now find the bin to which the points belong
            binidxs = np.searchsorted(self.bins, sortedpoints[:,self.binidx], side='right')
            # print("binidxs:", binidxs)
            # Now lets add all these into the appropriate spectrum
            # This is just x-channel but binned, howto to get energyloss?
            # oh we need to bring in the energyscale to here,
            # my oh my, this is gonna be messy.....
            emissionchannel = sortedpoints[:,0]
            if self.elossmap:
                emissionenergy = energyscale[emissionchannel.astype(int)]
                incidentenergy = sortedpoints[:,self.elossbinidx]
                #elosscenter = np.mean(energyscale)
                # Lets try to do it naive, AND superslow!!!
                eshifts = []
                for e_in in incidentenergy:
                    # Find channelshift
                    ch_in = (np.abs(energyscale - e_in)).argmin()
                    eshifts.append(ch_in)
                emissionchannel += self.x_size/2-np.array(eshifts)
                # Todo, somehow make a wider waterfall when measuring eloss.
                emissionchannel[emissionchannel < 0] = 0
                emissionchannel[emissionchannel > self.x_size-1] = self.x_size-1
                #print("Damn, it's an elossmap", eshifts)
            binnedemissionchannel = emissionchannel/self.e_binsize
            np.add.at(self.waterfall, (binidxs, binnedemissionchannel.astype(int)), 1)
            # Now we also need to add to the binlabels sums, and binlabel nums, and the recalculate the binlabels.
            for labelidx,labelpointidx in enumerate(self.binlabelpointsidx):
                np.add.at(self.binlabelsums, (binidxs, labelidx), sortedpoints[:,labelpointidx])
                np.add.at(self.binlabelnums, (binidxs, labelidx), 1)
            uniquebins = np.unique(binidxs)
            # Now lets refresh the binlabels that we touched!
            self.binlabels[uniquebins] = self.binlabelsums[uniquebins]/self.binlabelnums[uniquebins]
            # Mark the bins as not fitted!
            self.fitresults[uniquebins] = None
            #self.binlabelsums = np.empty((0,self.numbinlabels))
            #self.binlabelnums = np.empty((0,self.numbinlabels))
            ###print(time.time()-starttime)


    def _expand_bins(self, minimum, maximum):
        """
        Expand the bins so that everything between minimum and maximum is
        also included. The max binedge is 1 step larger than the maximum,
        this is so that we have the end bin also. 2* comes from that
        arange excludes the endpoint
        """
        if len(self.bins) == 0: # We cannot take min,max if we don't already have bins
            newbins = np.arange(minimum, maximum+2*self.t_binsize, self.t_binsize)
            self.__prepend_bins(newbins)
        else:
            if minimum < np.min(self.bins):
                newbins = np.arange(minimum, np.min(self.bins), self.t_binsize)
                self.__prepend_bins(newbins)
            if maximum >= np.max(self.bins)-self.t_binsize:
                newbins = np.arange(np.max(self.bins)+self.t_binsize, maximum+2*self.t_binsize, self.t_binsize)
                self.__append_bins(newbins)

    def __prepend_bins(self, newbins):
        """
        Helper function that add bins to the waterfall.
        Inserts the newbins, at position 0
        """
        self.bins = np.insert(self.bins, 0, newbins)
        newspectrum = np.zeros(int(self.x_size/self.e_binsize))
        newbinlabels = np.array([None]*self.numbinlabels)
        newbinlabelnumums = np.array([0]*self.numbinlabels)
        for bin in newbins:
            self.fitresults = np.insert(self.fitresults, 0, None)
            self.waterfall = np.vstack([newspectrum, self.waterfall])
            self.binlabels = np.vstack([newbinlabels, self.binlabels])
            self.binlabelnums = np.vstack([newbinlabelnumums, self.binlabelnums])
            self.binlabelsums = np.vstack([newbinlabelnumums, self.binlabelsums])

    def __append_bins(self, newbins):
        """
        Helper function that add bins to the waterfall.
        """
        newspectrum = np.zeros(int(self.x_size/self.e_binsize))
        self.bins = np.append(self.bins, newbins)
        newbinlabels = np.array([None]*self.numbinlabels)
        newbinlabelnumums = np.array([0]*self.numbinlabels)
        for bin in newbins:
            self.fitresults = np.append(self.fitresults, None)
            self.waterfall = np.vstack([self.waterfall, newspectrum])
            self.binlabels = np.vstack([self.binlabels, newbinlabels])
            self.binlabelnums = np.vstack([self.binlabelnums, newbinlabelnumums])
            self.binlabelsums = np.vstack([self.binlabelsums, newbinlabelnumums])

    def ClearFits(self):
        """
        Mark all fits as unfitted
        """
        with self.lock:
            for i in range(len(self.bins)):
                self.fitresults[i] = None
            self.fitplots = None
            self.fitplotsstderr = None


    def FitUnittedSpectra(self, energyscale, fitter):
        """
        TO be able to output progress, this is now a generator!
        Go to all bins where the labels are None, and fit them

        # Fitter is from rixsaq_fittingbackend, use e.g.
        myModel = RixsAqFitModelLinear()
        out = myModel.Fit(datax, datay)
        # But in this case we will probably use the one in the gui
        self.fitBackModel = FitBack.from_json(FitBack.to_json(self.peakfittab.fitBackModel))
        fitoutput = self.fitBackModel.Fit(x, y)
        """
        #with self.lock: # probably needed, but it slows down the tango reading :(
        numfitted = 0
        numtofit = 0 # To be able to give a progress bar

        numbins = min(len(self.bins), len(self.waterfall)) # Hack when multithreading
        #if len(self.bins) != len(self.waterfall):
        #    print("Warning, we are a bit out of sync")
        for i in range(numbins):
            if self.fitresults[i] is None:
                numtofit += 1
                yield numfitted,numtofit
        for i in range(numbins):
            if self.fitresults[i] is None:
                spectrum = self.waterfall[i,:]
                if sum(spectrum) > 10: # don't fit empty spectra please
                    fitoutput = fitter.Fit(energyscale, spectrum) # the magic happens here!
                    self.fitresults[i] = fitoutput
                    numfitted += 1
                    yield numfitted,numtofit
        ########### fitting is done here, the rest is so that we can have each parameter plotted easily
        if numfitted > 0:
            # Lets also keep array of all fitted results, we can use that to quickly make a plot of 
            # The different parameters in the bins
            params = list(fitoutput.params.keys())
            self.fitparamlabels = params
            fitplots = [[] for i in params]
            fitplotsstderr = [[] for i in params]
            for i,key in enumerate(params):
                for z in range(len(self.fitresults)):
                    if self.fitresults[z] is None:
                        res = np.nan
                        res2 = np.nan
                    else:
                        res = self.fitresults[z].params[key].value
                        res2 = self.fitresults[z].params[key].stderr
                    fitplots[i].append(res)
                    fitplotsstderr[i].append(res2)
            self.fitplots = fitplots # TODO HACK, this updated all of this every time, potentiolly very slow!
            self.fitplotsstderr = fitplotsstderr
        yield numfitted,numtofit


if __name__=="__main__":
    testpoints = np.array([[1, 1, 3, 1337],
                           [1, 1, 3, 1337],
                           [1, 2, 3, 1347],
                           [1, 2, 3, 1347],
                           [1, 4, 1, 1337],
                           [1, 4, 1, 1357]])
    for i in range(15):
        testpoints = np.vstack([testpoints, testpoints])
    print(testpoints.shape)
    wf = WaterfallData(10, 3, 8)
    print(wf.bins)
    wf.add_points(testpoints)
    print(wf.bins)
    print(wf.waterfall)
    wf.add_points(testpoints)
    print(wf.bins)
    print(wf.waterfall)
