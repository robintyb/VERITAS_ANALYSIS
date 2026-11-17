import json
import numpy as np


class DLDStatistics:
    """
    Container for a dld statistics readout.
    """
    def __init__(self,
                 counts_read     = None,
                 counts_received = None,
                 events_found    = None,
                 events_in_roi   = None,
                 events_received = None,
                 counters        = None,
                 jsonstr         = None
                 ):
        if counters is None:
            self.counters = np.zeros(16)
        else:
            self.counters = np.array(counters)

        # counts_read = Tdc data per channel transferred from fpga to cpu
        if counts_read is None:
            self.counts_read  = np.zeros(16)
        else:
            self.counts_read = np.array(counts_read)

        #Dld events recognized in fpga
        if events_found is None:
            self.events_found = 0
        else:
            self.events_found = events_found

        # Dld events recognized in fpga which fits in hardware roi
        if events_in_roi is None:
            self.events_in_roi = 0
        else:
            self.events_in_roi = events_in_roi

        if events_received is None:
            self.events_received = 0
        else:
            self.events_received = events_received

        # Tdc data per channel transferred from fpga to cpu
        if counts_received is None:
            self.counts_received = np.zeros(16)
        else:
            self.counts_received = np.array(counts_received)

        # Load from json
        if jsonstr is not None:
            self._from_JSON(jsonstr)

    def __repr__(self):
        """
        String representation of dldstatistics, used for getting a nice print
        """
        return ("""
        Counters %s
        Counts_read %s
        Events_found %s
        Events_in_roi %s
        Events_received %s
        Counts_received %s""") %(str(self.counters), str(self.counts_read), str(self.events_found),
                                 str(self.events_in_roi), str(self.events_received), str(self.counts_received))

    def __add__(self, other):
        """
        Sum the statistics of two exposures together, useful for accumulation
        """
        counts_read     = self.counts_read + other.counts_read
        counts_received = self.counts_received + other.counts_received
        events_found    = self.events_found + other.events_found
        events_in_roi   = self.events_in_roi + other.events_in_roi
        events_received     = self.events_received + other.events_received
        counters        = self.counters + other.counters
        return DLDStatistics(counts_read, counts_received, events_found, events_in_roi, events_received, counters)

    def to_JSON(self):
        class NumPyArangeEncoder(json.JSONEncoder):
            def default(self, obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                return json.JSONEncoder.default(self, obj)
        return json.dumps(self.__dict__, cls=NumPyArangeEncoder)

    def _from_JSON(self, jstr):
        jdict = json.loads(jstr)
        self.counts_read     = np.array(jdict['counts_read'])
        self.counts_received = np.array(jdict['counts_received'])
        self.events_found    = jdict['events_found']
        self.events_in_roi   = jdict['events_in_roi']
        self.events_received = jdict['events_received']
        self.counters        = np.array(jdict['counters'])