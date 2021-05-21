"""
Tools for working with sampled data
"""

import numpy as np
from scipy.signal import hilbert, firwin, filtfilt, butter

class Time:
    """
    Time when working with sampled data (including video). INTEGER IMPLIES SAMPLE NUMBER, FLOAT IMPLIES TIME.
    Use this to encapsulate sampling rate (sr), sample number (sample), and time (s).
    When the sampling rate is changed, the sample number is updated, but the time is held constant.
    When the time is changed, sample number is updated.
    When the sample number is changed, the time is updated
    When working in Premiere Pro, use 29.97 fps drop-frame timecode to show the actual time in video.
    You should see semicolons instead of colons
        inp 
            (str)   hh;mm;ss;frame#
            (float) assumes provided input is time in seconds!
            (int)   assumes the provided input is the sample number
            (tuple) assumes (timestamp/time/sample, sampling rate)
        sr 
            sampling rate, in Hz. casted into a float.

    Examples:
        t = Time('00;09;53;29', 30)
        t = Time(9.32, 180)
        t = Time(12531, 180)
        t = Time((9.32, sr=180))
        t = Time((9.32, 180), 30) # DO NOT DO THIS, sampling rate will be 180
        t.time
        t.sample
    """
    def __init__(self, inp, sr=30.):
        # set the sampling rate
        if isinstance(inp, tuple):
            assert len(inp) == 2
            self._sr = float(inp[1])
            inp = inp[0] # input is now either a string, float, or int!
        else:
            self._sr = float(sr)

        # set the sample number before setting the time
        assert isinstance(inp, (str, float, int))
        if isinstance(inp, str):
            inp = [int(x) for x in inp.split(';')]
            self._sample = int((inp[0]*60*60 + inp[1]*60 + inp[2])*self.sr + inp[3])
        if isinstance(inp, float): # time to sample
            self._sample = int(inp*self.sr)
        if isinstance(inp, int):
            self._sample = inp
        
        # set the time based on the sample number
        self._time = float(self._sample)/self._sr

    @property
    def sr(self):
        return self._sr

    @sr.setter
    def sr(self, sr_val):
        """When changing the sampling rate, time is kept the same, and the sample number is NOT"""
        sr_val = float(sr_val)
        self._sr = sr_val
        self._sample = int(self._time*self._sr)
    
    def change_sr(self, new_sr):
        self.sr = new_sr
        return self

    @property
    def sample(self):
        return self._sample
    
    @sample.setter
    def sample(self, sample_val):
        self._sample = int(sample_val)
        self._time  = float(self._sample)/self._sr
    
    @property
    def time(self):
        """Return time in seconds"""
        return self._time

    @time.setter
    def time(self, s_val):
        """If time is changed, then the sample number should be reset as well"""
        self._time = float(s_val)
        self._sample = int(self._time*self._sr)

    def __add__(self, other):
        x = self._arithmetic(other)
        return Time(x[2].__add__(x[0], x[1]), self.sr)

    def __sub__(self, other):
        x = self._arithmetic(other)
        return Time(x[2].__sub__(x[0], x[1]), self.sr)

    def _arithmetic(self, other):
        if isinstance(other, self.__class__):
            assert other.sr == self.sr
            return (self.sample, other.sample, int)
        elif isinstance(other, int):
            # integer implies sample, float implies time
            return (self.sample, other, int)
        elif isinstance(other, float):
            return (self.time, other, float)
        else:
            raise TypeError(other, "Unexpected input type! Input either a float for time, integer for sample, or time object")

    def to_interval(self, zero=None, iter_rate=None):
        """Return an interval object with start and end times being the same"""
        return Interval(self, self, zero, self.sr, iter_rate)

class Interval:
    """
    Interval object with start and stop times. Implements the iterator protocol.
    INCLUDES BOTH START AND END SAMPLES
    Pictoral understanding:
    start           -> |                                           | <-
    frames          -> |   |   |   |   |   |   |   |   |   |   |   | <- [self.sr, len(self)=12, self.t_data, self.t]
    animation times -> |        |        |        |        |         <- [self.iter_rate, self._index, self.t_iter]
    Frame sampling is used to pick the nearest frame corresponding to the animation times
    Example:
        intvl = ot.Interval(('00;09;51;03', 30), ('00;09;54;11', 30), sr=180, iter_rate=env.Key().fps)
        intvl.iter_rate = 24 # say 24 fps for animation
        for nearest_sample, time, index in intvl:
            print((nearest_sample, time, index))
    """
    def __init__(self, start, end, zero=None, sr=30., iter_rate=None):
        # if isinstance(start, (int, float)) and sr is not None:
        self.start = self._process_inp(start, sr)
        self.end = self._process_inp(end, sr)
        if zero is None:
            self.zero = self.start
        else:
            self.zero = self._process_inp(zero, sr)

        assert self.start.sr == self.end.sr == self.zero.sr # interval is defined for a specific sampled dataset
        
        self._index = 0
        if iter_rate is None:
            self.iter_rate = self.sr # this will be the animation fps when animating data at a different rate
        else:
            self.iter_rate = float(iter_rate)

    @staticmethod
    def _process_inp(inp, sr):
        if isinstance(inp, Time):
            return inp # sr is ignored, superseded by input's sampling rate
        return Time(inp, sr) # string, float, int or tuple. sr is ignored if tuple.

    @property
    def sr(self):
        return self.start.sr
    
    @sr.setter
    def sr(self, sr_val):
        sr_val = float(sr_val)
        self.start.sr = sr_val
        self.end.sr = sr_val
        self.zero.sr = sr_val
        
    def change_sr(self, new_sr):
        self.sr = new_sr
        return self

    @property
    def dur_time(self):
        """Duration in seconds"""
        return self.end.time - self.start.time
    
    @property
    def dur_sample(self):
        """Duration in number of samples"""
        return self.end.sample - self.start.sample + 1 # includes both start and end samples
    
    def __len__(self):
        return self.dur_sample

    # iterator protocol - you can do: for sample, time, index in interval
    def __iter__(self):
        """Iterate from start sample to end sample"""
        return self
    
    def __next__(self):
        index_interval = 1./self.iter_rate
        if self._index <= int(self.dur_time*self.iter_rate)+1:
            time = self.start.time + self._index*index_interval
            nearest_sample = self.start.sample + int(self._index*index_interval*self.sr)
            result = (nearest_sample, time, self._index)
        else:
            self._index = 0
            raise StopIteration
        self._index += 1
        return result
    
    # time vectors
    @property
    def t_iter(self):
        """Time Vector for the interval at iteration frame rate"""
        return self._t(self.iter_rate)

    @property
    def t_data(self):
        """Time vector at the data sampling rate"""
        return self._t(self.sr)

    @property
    def t(self):
        """Time Vector relative to t_zero"""
        tzero = self.zero.time
        return [t - tzero for t in self.t_data]
        
    def _t(self, rate):
        _t = [self.start.time]
        while _t[-1] <= self.end.time:
            _t.append(_t[-1] + 1./rate)
        return _t

    def __add__(self, other):
        return Interval(self.start+other, self.end+other, zero=self.zero+other, sr=self.sr, iter_rate=self.iter_rate)

    def __sub__(self, other):
        return Interval(self.start-other, self.end-other, zero=self.zero-other, sr=self.sr, iter_rate=self.iter_rate)

    def add(self, other):
        """Add to object, rather than returning a new object"""
        self.start = self.start + other
        self.end = self.end + other
        self.zero = self.zero + other

    def sub(self, other):
        self.start = self.start - other
        self.end = self.end - other
        self.zero = self.zero - other


class Data: # Signal processing
    def __init__(self, sig, sr, axis=None, history=None, t0=0.):
        """
        axis (int) time axis
        t0 (float) time at start sample
        """
        self._sig = sig # assumes sig is uniformly resampled
        self.sr = sr
        if axis is None:
            self.axis = np.argmax(np.shape(self._sig))
        else:
            self.axis = axis
        if history is None:
            self._history = [('initialized', None)]
        else:
            assert isinstance(history, list)
            self._history = history
        self._t0 = t0
    
    def __call__(self):
        return self._sig

    def _clone(self, proc_sig, his_append=None):
        if his_append is None:
            his = self._history # only useful when cloning without manipulating the data, e.g. returning a subset of columns
        else:
            his = self._history + [his_append]
        return self.__class__(proc_sig, self.sr, self.axis, his, self._t0)

    def analytic(self):
        proc_sig = hilbert(self._sig, axis=self.axis)
        return self._clone(proc_sig, ('analytic', None))

    def envelope(self, type='upper', lowpass=True):
        # analytic envelope, optionally low-passed
        assert type in ('upper', 'lower')
        if type == 'upper':
            proc_sig = np.abs(hilbert(self._sig, axis=self.axis))
        else:
            proc_sig = -np.abs(hilbert(-self._sig, axis=self.axis))

        if lowpass:
            if lowpass is True: # set cutoff frequency to lower end of bandpass filter
                assert 'bandpass' in [h[0] for h in self._history]
                lowpass = [h[1]['low'] for h in self._history if h[0] == 'bandpass'][0]
            assert isinstance(lowpass, (int, float)) # cutoff frequency
            return self._clone(proc_sig, ('envelope_'+type)).lowpass(lowpass)
        return self._clone(proc_sig, ('envelope_'+type, None))
    
    def phase(self):
        proc_sig = np.unwrap(np.angle(hilbert(self._sig, axis=self.axis)))
        return self._clone(proc_sig, ('instantaneous_phase', None))
    
    def instantaneous_frequency(self):
        proc_sig = np.diff(self.phase()._sig) / (2.0*np.pi) * self.sr
        return self._clone(proc_sig, ('instantaneous_frequency', None))

    def bandpass(self, low, high, order=None):
        if order is None:
            order = int(self.sr/2) + 1
        filt_pts = firwin(order, (low, high), fs=self.sr, pass_zero='bandpass')
        proc_sig = filtfilt(filt_pts, 1, self._sig, axis=self.axis)
        return self._clone(proc_sig, ('bandpass', {'filter':'firwin', 'low':low, 'high':high, 'order':order}))

    def _butterfilt(self, cutoff, order, btype):
        assert btype in ('low', 'high')
        if order is None:
            order = 6
        b, a = butter(order, cutoff/(0.5*self.sr), btype=btype, analog=False)
        proc_sig = filtfilt(b, a, self._sig, axis=self.axis)
        return self._clone(proc_sig, (btype+'pass', {'filter':'butter', 'cutoff':cutoff, 'order':order}))

    def lowpass(self, cutoff, order=None):
        return self._butterfilt(cutoff, order, 'low')
    
    def highpass(self, cutoff, order=None):
        return self._butterfilt(cutoff, order, 'high')

    def __len__(self):
        return np.shape(self._sig)[self.axis]

    @property
    def t(self):
        n_samples = len(self)
        return np.linspace(self._t0, self._t0 + (n_samples-1)/self.sr, n_samples)


class Event(Interval):
    def __init__(self, start, end, **kwargs):
        """
        kwargs:
        labels (list of strings) - hastags defining the event
        """
        self.labels = kwargs.pop('labels', [])
        super().__init__(start, end, **kwargs)


class Events(list):
    def append(self, key):
        assert isinstance(key, Event)
        super().append(key)
    
    def get(self, label):
        return Events([e for e in self if label in e.labels])
