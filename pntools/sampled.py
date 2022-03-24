"""
Tools for working with sampled data
"""

import collections
import numpy as np
from scipy.signal import hilbert, firwin, filtfilt, butter
from scipy.fft import fft, fftfreq
from scipy.interpolate import interp1d

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
    
    def __repr__(self):
        return "time={:.3f} s, sample={}, sr={} Hz ".format(self.time, self.sample, self.sr) + super().__repr__()


class Sequence:
    """
    Create a sequence of named time objects (collection).
    Created for working with motion sequences / periodic data collected
    from multiple modalities/devices where data is sampled at different
    rates. 
    Seqeuences of events in the real-world are observed through
    different modalities, but there is only one 'base' time in the real
    world, and each data acquistion modality is going to have its own
    clock and sampling rate. I want to be able to refer to the
    event/sequence in the real world.

    Inputs:
        marker_names - string of words separated by spaces, like input to namedtuple
        input_sr     - sampling rate at which time will be specified
    Example:
        normal_pitching = Sequence('start emg_start acc_start foot_off release end', input_sr=30.)
        normal_pitching.append('00;05;57;26', '00;05;58;18', '00;06;00;20', '00;06;00;26', '00;06;00;29', '00;06;01;29')
        normal_pitching[0].emg_start

        n_zoom = normal_pitching.change_sr(25.) # when retrieving frames from zoom
        n_canon = normal_pitching.change_sr(30.) # when retrieving frames from canon DSLR
        n_motive = normal_pitching.change_sr(180.) # when working with motion capture videos
        n_delsys = normal_pitching.change_sr(2000.) # when dealing with EMG data sampled at 2000 Hz
        n_delsys[0].emg_start

        n_zoom.all_labels()
    """
    def __init__(self, marker_names, input_sr=30., output_sr=180.):
        self._marker_names = marker_names
        self._template = collections.namedtuple('Sequence', marker_names)
        self._data = [] # each entry is a dictionary with ev and labels
        self._input_sr = input_sr # sampling rate of timestamps that will be input
        self._output_sr = output_sr
    
    def append(self, *args, **kwargs):
        """Add a sequence to this collection."""
        labels = kwargs.pop('labels', [])
        if isinstance(labels, str):
            labels = [labels]
        processed_args = []
        for arg in args:
            processed_args.append(self._process_inp(arg).change_sr(self._output_sr))
        processed_kwargs = {}
        for kwarg_name, kwarg in kwargs.items():
            processed_kwargs[kwarg_name] = self._process_inp(kwarg).change_sr(self._output_sr)
        self._data.append({'ev': self._template(*processed_args, **processed_kwargs), 'labels': labels})

    def __getitem__(self, key):
        """Retrieve event from the _data list. This hides the labels."""
        if isinstance(key, int):
            return self._data[key]['ev']
        elif isinstance(key, str):
            return [d['ev'] for d in self._data if key in d['labels']]
    
    def change_sr(self, new_sr): # rename to change_modality?
        """Create a new sequence object where output sampling rate is new_sr"""
        s = Sequence(self._marker_names, self._input_sr, new_sr)
        for d in self._data:
            s.append(**d['ev']._asdict(), labels=d['labels'])
        return s

    def all_labels(self):
        ret = []
        for d in self._data:
            ret += d['labels']
        return set(ret)

    def _process_inp(self, inp):
        if isinstance(inp, Time):
            return inp # sr is ignored, superseded by input's sampling rate
        return Time(inp, self._input_sr) # string, float, int or tuple. sr is ignored if tuple.


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
        intvl = Interval(('00;09;51;03', 30), ('00;09;54;11', 30), sr=180, iter_rate=env.Key().fps)
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
        """Used to shift an interval, use union to find a union"""
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

    def union(self, other):
        """ 
        Merge intervals to make an interval from minimum start time to
        maximum end time. Other can be an interval, or a tuple of intervals.

        iter_rate, sr, and zero are inherited from the original
        event. Therefore, e1.union(e2) doesn't have to be the same as
        e2.union(e1)
        """
        assert self.sr == other.sr
        this_start = (self.start, other.start)[np.argmin((self.start.time, other.start.time))]
        this_end = (self.end, other.end)[np.argmax((self.end.time, other.end.time))]
        return Interval(this_start, this_end, zero=self.zero, sr=self.sr, iter_rate=self.iter_rate)


class Data: # Signal processing
    def __init__(self, sig, sr, axis=None, history=None, t0=0.):
        """
        axis (int) time axis
        t0 (float) time at start sample
        NOTE: When inheriting from this class, if the parameters of the
        __init__ method change, then make sure to rewrite the _clone method
        """
        self._sig = np.asarray(sig) # assumes sig is uniformly resampled
        assert self._sig.ndim in (1, 2)
        if not hasattr(self, 'sr'): # in case of multiple inheritance - see ot.Marker
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
    
    def __call__(self, col=None):
        """Return either a specific column or the entire set 2D signal"""
        if col is not None:
            assert isinstance(col, int) and col < len(self)
            slc = [slice(None)]*self._sig.ndim
            slc[(self.axis+1)%self._sig.ndim] = col
            return self._sig[tuple(slc)] # not converting slc to tuple threw a FutureWarning
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
            return self._clone(proc_sig, ('envelope_'+type, None)).lowpass(lowpass)
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

        nan_manip = False
        if (nan_bool := np.isnan(self._sig)).any():
            nan_manip = True
            self = self.interpnan() # interpolate missing values before applying an IIR filter

        proc_sig = filtfilt(b, a, self._sig, axis=self.axis)
        if nan_manip:
            proc_sig[nan_bool] = np.NaN # put back the NaNs in the same place

        return self._clone(proc_sig, (btype+'pass', {'filter':'butter', 'cutoff':cutoff, 'order':order, 'NaN manipulation': nan_manip}))

    def lowpass(self, cutoff, order=None):
        return self._butterfilt(cutoff, order, 'low')
    
    def highpass(self, cutoff, order=None):
        return self._butterfilt(cutoff, order, 'high')

    def medfilt(self, order=11):
        """
        Median filter the signal
        
        order is the number of samples in the kernel if it is an int, and treated as time if it is a float
        """
        sw = np.lib.stride_tricks.sliding_window_view # this should be much faster than using running window
        if isinstance(order, float):
            order = int(order*self.sr)
        assert isinstance(order, int)
        order = (order // 2)*2 + 1 # ensure order is odd for simpler handling of time
        proc_sig_middle = np.median(sw(self._sig, order, axis=self.axis), axis=-1)
        pre_fill = np.take(self._sig, np.r_[:order//2], axis=self.axis)
        post_fill = np.take(self._sig, np.r_[-order//2+1:0], axis=self.axis)
        proc_sig = np.concatenate((pre_fill, proc_sig_middle, post_fill)) # ends of the signal will not be filtered
        return self._clone(proc_sig, ('median_filter', {'order': order, 'kernel_size_s': order/self.sr}))
    
    def interpnan(self, maxgap=None, **kwargs):
        """
        Only interpolate values within the mask
        kwargs will be passed to scipy.interpolate.interp1d
        """
        proc_sig = np.apply_along_axis(interpnan, self.axis, self._sig, maxgap, **kwargs)
        return self._clone(proc_sig, ('instantaneous_phase', None))

    def shift_baseline(self, offset): 
        # you can use numpy broadcasting to shift each signal if multi-dimensional
        return self._clone(self._sig - offset, ('shift_baseline', offset))
    
    def scale(self, scale_factor):
        return self._clone(self._sig/scale_factor, ('scale', scale_factor))
        
    def __len__(self):
        return np.shape(self._sig)[self.axis]

    @property
    def t(self):
        n_samples = len(self)
        return np.linspace(self._t0, self._t0 + (n_samples-1)/self.sr, n_samples)
    
    @property
    def dur(self):
        return (len(self)-1)/self.sr
    
    def __getitem__(self, key): 
        # NOTE: Generalize for multi-dimensional signals!
        assert isinstance(key, Interval)
        his = self._history + [('slice', key)]
        offset = round(self._t0*self.sr)
        proc_sig = self._sig.take(indices=range(key.start.sample-offset, key.end.sample-offset), axis=self.axis)
        return self.__class__(proc_sig, self.sr, self.axis, his, key.start.time)

    def make_running_win(self, win_size=0.25, win_inc=0.1):
        win_size_samples = (round(win_size*self.sr)//2)*2 + 1 # ensure odd number of samples
        win_inc_samples = round(win_inc*self.sr)
        n_samples = len(self)
        return RunningWin(n_samples, win_size_samples, win_inc_samples)

    def apply_running_win(self, func, win_size=0.25, win_inc=0.1):
        """
        Process the signal using a running window by applying func to each window.
        Returns:
            Sampled data 
        Example:
            Extract RMS envelope
            self.apply_running_win(lambda x: np.sqrt(np.mean(x**2)), win_size, win_inc)
        """
        rw = self.make_running_win(win_size, win_inc)
        ret_sig = np.array([func(self._sig[r_win], self.axis) for r_win in rw()])
        ret_sr = self.sr/round(win_inc*self.sr)
        return Data(ret_sig, ret_sr, axis=self.axis, t0=self.t[rw.center_idx[0]])
    
    def __le__(self, other): return self._comparison('__le__', other)
    def __ge__(self, other): return self._comparison('__ge__', other)
    def __lt__(self, other): return self._comparison('__lt__', other)
    def __gt__(self, other): return self._comparison('__gt__', other)
    def __eq__(self, other): return self._comparison('__eq__', other)
    def __ne__(self, other): return self._comparison('__ne__', other)

    def _comparison(self, dunder, other):
        cmp_dunder_dict = {'__le__':'<=', '__ge__':'>=', '__lt__':'<', '__gt__':'>', '__eq__':'==', '__ne__':'!='}
        assert dunder in cmp_dunder_dict
        assert isinstance(other, (int, float))
        return self._clone(getattr(self._sig, dunder)(other), (cmp_dunder_dict[dunder], other))
    
    def onoff_times(self):
        """Onset and offset times of a thresholded 1D sampled.Data object"""
        onset_samples, offset_samples = onoff_samples(self._sig)
        return [self.t[x] for x in onset_samples], [self.t[x] for x in offset_samples]

    def fft(self):
        N = len(self)
        T = 1/self.sr
        f = fftfreq(N, T)[:N//2]
        amp = 2.0/N * np.abs(fft(self._sig)[0:N//2])
        return amp, f
    
    def diff(self):
        if self._sig.ndim == 2:
            if self.axis == 1:
                pp_value = (self._sig[:, 1] - self._sig[:, 0])[:, None]
                fn = np.hstack
            else: # self.axis == 0
                pp_value = self._sig[1] - self._sig[0]
                fn = np.vstack
        else: # self._sig.ndim == 1
            pp_value = self._sig[1] - self._sig[0]
            fn = np.hstack

        # returning a marker type even though this is technically not true
        return self._clone(fn( (pp_value, np.diff(self._sig, axis=self.axis)*self.sr) ), ('diff', None))
    
    def magnitude(self):
        assert self._sig.ndim == 2 # magnitude does not make sense for a 1D signal (in that case, use np.linalg.norm directly)
        return Data(np.linalg.norm(self._sig, axis=(self.axis+1)%2), self.sr, history=self._history+[('magnitude', 'None')])

    def apply(self, func, *args, **kwargs):
        try:
            kwargs['axis'] = self.axis
            proc_sig = func(self._sig, *args, **kwargs)
        except TypeError:
            kwargs.pop('axis')
            proc_sig = func(self._sig, *args, **kwargs)
        return self._clone(proc_sig, ('apply', {'func': func, 'args': args, 'kwargs': kwargs}))
        

class Event(Interval):
    def __init__(self, start, end, **kwargs):
        """
        Interval with labels.

        kwargs:
        labels (list of strings) - hastags defining the event
        """
        self.labels = kwargs.pop('labels', [])
        super().__init__(start, end, **kwargs)


class Events(list):
    """List of event objects that can be selected by labels using the 'get' method."""
    def append(self, key):
        assert isinstance(key, Event)
        super().append(key)
    
    def get(self, label):
        return Events([e for e in self if label in e.labels])


class RunningWin:
    def __init__(self, n_samples, win_size, win_inc=1, step=None, offset=0):
        """
        n_samples, win_size, and win_inc are integers (not enforced, but expected!)
        offset (int) offsets all running windows by offset number of samples.
            This is useful when the object you're slicing has an inherent offset that you need to consider.
            For example, consider creating running windows on a sliced optitrack marker
            Think of offset as start_sample
        Attributes of interest:
            run_win (array of slice objects)
            center_idx (indices of center samples)
        """
        self.n_samples = int(n_samples)
        self.win_size = int(win_size)
        self.win_inc = int(win_inc)
        self.n_win = int(np.floor((n_samples-win_size)/win_inc) + 1)
        self.start_index = int(offset)

        run_win = []
        center_idx = []
        for win_count in range(0, self.n_win):
            win_start = (win_count * win_inc) + offset
            win_end = win_start + win_size
            center_idx.append(win_start + win_size//2)
            run_win.append(slice(win_start, win_end, step))
        
        self._run_win = run_win
        self.center_idx = center_idx
        
    def __call__(self, data=None):
        if data is None: # return slice objects
            return self._run_win
        # if data is supplied, apply slice objects to the data
        assert len(data) == self.n_samples
        return [data[x] for x in self._run_win]
    
    def __len__(self):
        return self.n_win


def interpnan(sig, maxgap=None, **kwargs):
    """
    Interpolate NaNs in a 1D signal
        sig - 1D numpy array
        maxgap - 
            - (NoneType) all NaN values will be interpolated
            - (int) stretches of NaN values smaller than or equal to maxgap will be interpolated
            - (boolean array) will be used as a mask where interpolation will only happen where maxgap is True
        kwargs - 
            these get passed to scipy.interpolate.interp1d function
            commonly used: kind='cubic'
    """
    assert np.ndim(sig) == 1
    if 'fill_value' not in kwargs:
        kwargs['fill_value'] = 'extrapolate'
        
    def nan_helper(y):
        return np.isnan(y), lambda z: z.nonzero()[0]
    proc_sig = sig.copy()
    nans, x = nan_helper(proc_sig)
    if maxgap is None:
        mask = np.ones_like(nans)
    elif isinstance(maxgap, int):
        nans = np.isnan(sig)
        mask = np.zeros_like(nans)
        onset_samples, offset_samples = onoff_samples(nans)
        for on_s, off_s in zip(onset_samples, offset_samples):
            assert on_s < off_s
            if off_s - on_s <= maxgap: # interpolate this
                mask[on_s:off_s] = True
    else:
        mask = maxgap
    assert len(mask) == len(sig)
    proc_sig[nans & mask]= interp1d(x(~nans), proc_sig[~nans], **kwargs)(x(nans & mask)) # np.interp(x(nans & mask), x(~nans), proc_sig[~nans])
    return proc_sig

def onoff_samples(tfsig):
    """
    Find onset and offset samples of a 1D boolean signal (e.g. Thresholded TTL pulse)
    Currently works only on 1D signals!
    tfsig is shorthand for true/false signal
    """
    assert tfsig.dtype == bool
    assert np.sum(np.asarray(np.shape(tfsig)) > 1) == 1
    x = np.squeeze(tfsig).astype(int)
    onset_samples = list(np.where(np.diff(x) == 1)[0] + 1)
    offset_samples = list(np.where(np.diff(x) == -1)[0] + 1)
    if tfsig[0]: # is True
        onset_samples = [0] + onset_samples
    if tfsig[-1]:
        offset_samples = offset_samples + [len(tfsig)]
    return onset_samples, offset_samples

def uniform_resample(time, sig, sr, t_min=None, t_max=None):
    """
    Uniformly resample a signal at a given sampling rate sr.
    Ideally the sampling rate is determined by the smallest spacing of
    time points.
    Inputs:
        time (list, 1d numpy array) is a non-decreasing array
        sig (list, 1d numpy array)
        sr (float) sampling rate in Hz
        t_min (float) start time for the output array
        t_max (float) end time for the output array
    Returns:
        pn.sampled.Data
    """
    assert len(time) == len(sig)
    assert np.ndim(sig) == 1

    if t_min is None: t_min = time[0]
    if t_max is None: t_max = time[-1]

    n_samples = int((t_max - t_min)*sr) + 1
    t_max = t_min + (n_samples-1)/sr

    t_proc = np.linspace(t_min, t_max, n_samples)
    sig_proc = np.interp(t_proc, time, sig)
    return Data(sig_proc, sr, t0=t_min)
