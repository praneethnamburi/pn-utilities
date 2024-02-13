"""
Simple Graphical User Interface elements for browsing data

Classes:
    GenericBrowser - Generic class to browse data. Meant to be extended.
    SignalBrowser - Browse an array of pntools.Sampled.Data elements, or 2D arrays
    PlotBrowser - Scroll through an array of complex data where a plotting function is defined for each element
    VideoBrowser - Scroll through images of a video
    VideoPlotBrowser - Browse through video and 1D signals synced to the video side by side

    Future:
        Extend VideoBrowser to play, pause and extract clips using hotkeys. Show timeline in VideoBrowser.
        Add clickable navigation.
"""
import functools
import inspect
import io
import json
import os
from datetime import datetime, timedelta
from pathlib import Path

import matplotlib as mpl
import numpy as np
import pandas as pd
from matplotlib import axes as maxes
from matplotlib import lines as mlines
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.path import Path as mPath
from matplotlib.widgets import Button as ButtonWidget
from matplotlib.widgets import LassoSelector as LassoSelectorWidget

import pntools as pn
from pntools import sampled, video

CLIP_FOLDER = 'C:\\data\\_clipcollection'

### Helper functions
def _parse_fax(fax, ax_pos=(0.01, 0.01, 0.98, 0.98)):
    assert isinstance(fax, (type(None), plt.Figure, maxes.Axes))
    if fax is None:
        f = plt.figure()
        ax = f.add_axes(ax_pos)
    elif isinstance(fax, plt.Figure):
        f = fax
        ax = f.add_axes(ax_pos)
    else:
        f = fax.figure
        ax = fax
    return f, ax

def _parse_pos(pos):
    if isinstance(pos, str):
        updown, leftright = pos.replace('middle', 'center').split(' ')
        assert updown in ('top', 'center', 'bottom')
        assert leftright in ('left', 'center', 'right')
        y = {'top': 1, 'center': 0.5, 'bottom': 0}[updown]
        x = {'left': 0, 'center': 0.5, 'right': 1}[leftright]
        pos = (x, y, updown, leftright)
    assert len(pos) == 4
    return pos


### Extended widget classes
class Button(ButtonWidget):
    """Add a 'name' state to a matplotlib widget button"""
    def __init__(self, ax, name:str, **kwargs) -> None:
        super().__init__(ax, name, **kwargs)
        self.name = name

class StateButton(Button): # store a number/coordinate
    def __init__(self, ax, name: str, start_state, **kwargs) -> None:
        super().__init__(ax, name, **kwargs)
        self.state = start_state # stores something in the state

class ToggleButton(StateButton):
    """
    Add a toggle button to a matplotlib figure

    For example usage, see plot browser
    """
    def __init__(self, ax, name:str, start_state:bool=True, **kwargs) -> None:
        super().__init__(ax, name, start_state, **kwargs)
        self.on_clicked(self.toggle)
        self.set_text()
    
    def set_text(self):
        self.label._text = f'{self.name}={self.state}'

    def toggle(self, event=None):
        self.state = not self.state
        self.set_text()
    
    def set_state(self, state:bool):
        assert isinstance(state, bool)
        self.state = state
        self.set_text()


class Selector:
    """
    Select points in a plot using the lasso selection widget
    Indices of selected points are stored in self.sel

    Example:
        f, ax = plt.subplots(1, 1)
        ph, = ax.plot(np.random.rand(20))
        plt.show(block=False)
        ls = gui.Lasso(ph)
        ls.start()
        -- play around with selecting points --
        ls.stop() -> disconnects the events
    """
    def __init__(self, plot_handle) -> None:
        """plot_handle - matplotlib.lines.Line2D object returned by plt.plot function"""
        assert isinstance(plot_handle, mlines.Line2D)
        self.plot_handle = plot_handle
        self.xdata, self.ydata = plot_handle.get_data()
        self.ax = plot_handle.axes
        self.overlay_handle, = self.ax.plot([], [], ".")
        self.sel = np.zeros(self.xdata.shape, dtype=bool)
        self.is_active = False

    def get_data(self):
        return np.vstack((self.xdata, self.ydata)).T

    def onselect(self, verts):
        """Select if not previously selected; Unselect if previously selected"""
        selected_ind = mPath(verts).contains_points(self.get_data())
        self.sel = np.logical_xor(selected_ind, self.sel)
        sel_x = list(self.xdata[self.sel])
        sel_y = list(self.ydata[self.sel])
        self.overlay_handle.set_data(sel_x, sel_y)
        plt.draw()
    
    def start(self, event=None): # split callbacks when using start and stop buttons
        self.lasso = LassoSelectorWidget(self.plot_handle.axes, self.onselect)
        self.is_active = True

    def stop(self, event=None):
        self.lasso.disconnect_events()
        self.is_active = False
    
    def toggle(self, event=None): # one callback when activated using a toggle button
        if self.is_active:
            self.stop(event)
        else:
            self.start(event)


### Managers for extended widget classes defined here (used by Generic browser)
class Buttons:
    """Manager for buttons in a matplotlib figure or GUI (see GenericBrowser for example)"""
    def __init__(self, parent):
        self._button_list : list[Button] = []
        self.parent = parent # matplotlib figure, or something that has a 'figure' attribute that is a figure

    def __len__(self):
        return len(self())

    def __getitem__(self, key):
        d = self.asdict()
        if isinstance(key, int) and key not in d:
            return self()[key]
        return d[key]
    
    def __call__(self):
        return self._button_list

    def asdict(self):
        return {b.name: b for b in self()}

    def add(self, text='Button', action_func=None, pos=None, w=0.25, h=0.05, buf=0.01, type_='Push', **kwargs):
        """
        Add a button to the parent figure / object
        If pos is provided, then w, h, and buf will be ignored
        """
        assert type_ in ('Push', 'Toggle')
        nbtn = len(self)
        if pos is None: # start adding at the top left corner
            parent_fig = self.parent.figure
            mul_factor = 6.4/parent_fig.get_size_inches()[0]
            
            btn_w = w*mul_factor
            btn_h = h*mul_factor
            btn_buf = buf
            pos = (btn_buf, (1-btn_buf)-((btn_buf+btn_h)*(nbtn+1)), btn_w, btn_h)
        
        if type_ == 'Toggle':
            b = ToggleButton(plt.axes(pos), text, **kwargs)
        else:
            b = Button(plt.axes(pos), text, **kwargs)

        if action_func is not None: # more than one can be attached
            if isinstance(action_func, (list, tuple)):
                for af in action_func:
                    b.on_clicked(af)
            else:
                b.on_clicked(action_func)
        
        self().append(b)
        return b


class Selectors:
    """Manager for selector objects - for picking points on line2D objects"""
    def __init__(self, parent):
        self._lasso_list : list[Selector] = []
        self.parent = parent
    
    def __call__(self):
        return self._lasso_list
    
    def __len__(self):
        return len(self())

    def add(self, plot_handle):
        ls = Selector(plot_handle)
        self._lasso_list.append(ls)
        return ls
        
class MemorySlots:
    def __init__(self, parent):
        self._idx = self.initialize()
        self._memtext = None
        self.parent = parent
    
    def __len__(self):
        return len(self._idx)

    @staticmethod
    def initialize():
        return {str(k):None for k in range(1, 10)}

    def disable(self):
        self._idx = {}
    
    def enable(self):
        self._idx = self.initialize()

    def show(self, pos='bottom left'):
        self._memtext = TextView(self._idx, fax=self.parent.figure, pos=pos)
    
    def update(self, key):
        """
        memory slot handling - Initiate when None, Go to the slot if it exists, frees slot if pressed when it exists
        key is the event.key triggered by a callback
        """
        if self._idx[key] is None:
            self._idx[key] = self.parent._current_idx
            self.update_display()
        elif self._idx[key] == self.parent._current_idx:
            self._idx[key] = None
            self.update_display()
        else:
            self.parent._current_idx = self._idx[key]
            self.parent.update()
    
    def update_display(self):
        """Refresh memory slot text if it is not hidden"""
        if self._memtext is not None:
            self._memtext.update(self._idx)

    def hide(self):
        """Hide the memory slot text"""
        if self._memtext is not None:
            self._memtext._text.remove()
        self._memtext = None
    
    def is_enabled(self):
        return bool(self._idx)

class StateVariable:
    def __init__(self, name:str, states:list):
        self.name = name
        self.states = list(states)
        self._current_state_idx = 0
    
    @property
    def current_state(self):
        return self.states[self._current_state_idx]
    
    def n_states(self):
        return len(self.states)
    
    def cycle(self):
        self._current_state_idx = (self._current_state_idx+1)%self.n_states()
    
    def set_state(self, state):
        if isinstance(state, int):
            assert 0 <= state < self.n_states()
            self._current_state_idx = state
        if isinstance(state, str):
            assert state in self.states
            self._current_state_idx = self.states.index(state)

class StateVariables:
    def __init__(self, parent):
        self._list : list[StateVariable] = []
        self.parent = parent
        self._text = None

    def __len__(self):
        return len(self._list)
    
    def __getitem__(self, key):
        """return the state variable by name key"""
        assert key in self.names
        return {x.name:x for x in self._list}[key]
    
    @property
    def names(self):
        return [x.name for x in self._list]
    
    def asdict(self):
        return {x.name:x.states for x in self._list}
    
    def add(self, name:str, states:list):
        assert name not in self.names
        self._list.append(StateVariable(name, states))
    
    def _get_display_text(self):
        return ['State variables:'] + [f'{x.name} - {x.current_state}' for x in self._list]
    
    def show(self, pos='bottom right'):
        self._text = TextView(self._get_display_text(), fax=self.parent.figure, pos=pos)

    def update_display(self):
        self._text.update(self._get_display_text())

class EventData:
    """
    Manage the data from one event type in one trial.
    """
    def __init__(self, default=None, added=None, removed=None, tags=None, algorithm_name:str='', params:dict=None) -> None:
        _to_list = lambda x: [] if x is None else x
        self.default = _to_list(default) # e.g. created by an algorithm
        self.added = _to_list(added)     # Manually added events, e.g. through a UI. if an 'added' point is removed, then it will simply be deleted. There will be no record of it.
        self.removed = _to_list(removed) # anything that is removed from default will be stored here
        self.tags = _to_list(tags)
        self.algorithm_name = algorithm_name
        self.params = params if params is not None else {} # params used to generate the default list 

    def asdict(self):
        return dict(
            default         = self.default,
            added           = self.added,
            removed         = self.removed,
            tags            = self.tags,
            algorithm_name  = self.algorithm_name,
            params          = self.params,
        )
    
    def __len__(self): # number of events
        return len(self.get_times())
    
    def get_times(self):
        x = self.default + self.added
        x.sort()
        return x

class Event:
    """
    Manage selection of a sequence of events (of length >= 1)
    """
    def __init__(self, name, size, fname, data_id_func=None, color='random', pick_action='overwrite', ax_list=None, win_remove=(-0.1, 0.1), win_add=(-0.25, 0.25), **plot_kwargs):
        self.name = name
        assert isinstance(size, int) and size > 0
        self.size = size # length of the sequence
        self.fname = fname # load and save events to this file
        self.data_id_func = data_id_func # gets the current data_id from the parent ui when executed
        if isinstance(color, int):
            color = pn.PLOT_COLORS[color]
        elif color == 'random':
            color = np.random.choice(pn.PLOT_COLORS)
        self.color = color
        assert pick_action in ('overwrite', 'append') # overwrite if there can only be one sequence per 'signal'. For multiple, use 'append'
        self.pick_action = pick_action

        self._buffer = []
        _, self._data = self.load()

        self.ax_list = ax_list # list of axes on which to show the event
        self.plot_handles = []

        # self.win_add = win_add # seconds, search to add peak within this window, in the peak or valley modes
        self.win_remove = win_remove # seconds, search to remove an event within this window
        self.win_add = win_add # seconds, search to add an event within this window in peak or valley mode
        self.plot_kwargs = plot_kwargs # tune the style of the plot using this
        self._hide = False

    def initialize_event_data(self, data_id_list):
        """Useful for initializing an event"""
        for data_id in data_id_list:
            if data_id not in self._data:
                self._data[data_id] = EventData()
    
    @classmethod
    def _from_existing_file(cls, fname, data_id_func=None):
        """Create an Event object by reading an existing json file."""
        h, _ = cls._read_json_file(fname)
        return cls(h['name'], h['size'], fname, data_id_func, h['color'], h['pick_action'], None, h['win_remove'], h['win_add'], **h['plot_kwargs'])
    
    @classmethod
    def from_file(cls, fname, **kwargs):
        """
        Create an empty events file with the given file name (fname) and any parameters.
        Assigns best-guess defaults
        """
        if not os.path.exists(fname):
            kwargs['name'] = kwargs.get('name', Path(fname).stem)
            kwargs['size'] = kwargs.get('size', 1)
            kwargs['data_id_func'] = kwargs.get('data_id_func', None) # this is irrelevant

            ret = cls(fname=fname, **kwargs)
            ret.save() # this has a print message
            return ret
        return cls._from_existing_file(fname, kwargs.get('data_id_func', None))

    @classmethod
    def from_data(cls, data:dict, name:str='Event', fname:str='', overwrite:bool=False, **kwargs):
        """Create an event file by filling in the 'default' events extracted by an algorithm.
        kwargs 
            - tags, algorithm_name, and params will be passed to gui.EventData
            - all other kwargs will be passed to gui.Event
        """
        algorithm_info = dict(
            tags = kwargs.pop('tags', []),
            algorithm_name = kwargs.pop('algorithm_name', ''),
            params = kwargs.pop('params', {})
        )
        
        size = []
        for key, val in data.items():
            if isinstance(val, EventData):
                continue
            v = np.asarray(val)
            if v.ndim == 1: # passing in a list events of size 1
                v = v[:, np.newaxis]
                val = [list(x) for x in v]
            data[key] = EventData(default=val, **algorithm_info)
            if len(data[key]) > 0: # when there are no events, size cannot be inferred for that trial. Note that this process will fail if there are no events in ANY of the trials. size has to be passed in with kwargs.
                size.append(v.shape[-1])
        
        if not size: # if there were no events in the data that was passed!
            assert 'size' in kwargs
            size = kwargs['size']
        else:
            size = list(set(size))
            assert len(size) == 1 # make sure we have the same type of events
            size = size[0]
        
        if 'size' in kwargs:
            assert kwargs['size'] == size
            del kwargs['size']
        
        ret = cls(name, size, fname, **kwargs)
        ret._data = data
        if pn.is_path_exists_or_creatable(fname):
            if (not os.path.exists(fname)) or overwrite:
                ret.save()
            else: # don't overwrite and exists, then append new data to the file if it exists
                assert os.path.exists(fname) and (not overwrite)
                ret_existing = cls.from_file(fname, **kwargs)
                new_keys = set(ret._data.keys()) - set(ret_existing._data.keys())
                if len(new_keys) > 0: # if there is new data, then add it to the event file
                    print(f'Appending new data to the event file {fname}:')
                    print(new_keys)
                    ret_existing._data = ret._data | ret_existing._data
                    ret_existing.save()
        return ret
    
    def all_keys_are_tuples(self) -> bool:
        return all([type(x) == tuple for x in self._data.keys()])

    def get_header(self):
        return dict(
            name  = self.name,
            size  = self.size,
            fname = self.fname,
            color = self.color,
            pick_action = self.pick_action,
            win_remove = self.win_remove,
            win_add = self.win_add,
            plot_kwargs = self.plot_kwargs,
            all_keys_are_tuples = self.all_keys_are_tuples(),
        )

    @staticmethod
    def _read_json_file(fname):
        with open(fname, 'r') as f:
            header, data = json.load(f)
        if header['all_keys_are_tuples']:
            data = {eval(k):EventData(**v) for k,v in data.items()}
        else:
            data = {k:EventData(**v) for k,v in data.items()}
        return header, data

    def load(self):
        if os.path.exists(self.fname):
            header, data = self._read_json_file(self.fname)
            return header, data
        return {}, {}

    def save(self):
        action_str = 'Updated' if os.path.exists(self.fname) else 'Created'
        with open(self.fname, 'w') as f:
            header = self.get_header()
            if header['all_keys_are_tuples']:
                data = {str(k):v.asdict() for k,v in self._data.items()}
            else:
                data = {k:v.asdict() for k,v in self._data.items()}
            json.dump((header, data), f, indent=4)
        print(action_str + ' ' + self.fname)
    
    def add(self, event): # the parent UI would invoke this
        """
        Pick the time points of an interval and associate it with a supplied ID
        If the first selection is outside the axis, then select the first available time point.
        If the last selection is outside the axis, then select the last available time point.
        If the selections are not monotonically increasing, then empty the buffer.
        If any of the 'middle' picks (i.e. not first or last in the sequence) are outside the axes, then empty the buffer.
        """
        def strictly_increasing(_list):
            return all(x<y for x, y in zip(_list, _list[1:]))

        def _get_lines():
            """Return non-empty lines in the axis where event was invoked, or else in all lines in the figure"""
            if event.inaxes is not None:
                return [line for line in event.inaxes.get_lines() if len(line.get_xdata()) > 0]
            return [line for ax in event.canvas.figure.axes for line in ax.get_lines() if len(line.get_xdata()) > 0] # return ALL lines in the figure
        
        def _get_first_available_timestamp():
            return min([np.nanmin(l.get_xdata()) for l in _get_lines() if len(l.get_xdata()) > 0])
            # return self.parent.data[self._current_idx].t[0]
        
        def _get_last_available_timestamp():
            return max([np.nanmax(l.get_xdata()) for l in _get_lines() if len(l.get_xdata()) > 0])
            # return self.parent.data[self._current_idx].t[-1]
        
        def clamp(n): 
            smallest = _get_first_available_timestamp()
            largest = _get_last_available_timestamp()
            return max(smallest, min(n, largest))
        
        # add picks to the buffer until the length is equal to the size
        if event.xdata is None: # pick is outside the axes
            if not self._buffer: # first in the sequence
                inferred_timestamp = _get_first_available_timestamp()
            else:
                assert len(self._buffer) == self.size-1 # last in the sequence
                inferred_timestamp = _get_last_available_timestamp()
        else:
            inferred_timestamp = clamp(float(event.xdata))

        self._buffer.append(inferred_timestamp)

        if not strictly_increasing(self._buffer):
            self._buffer = [] # reset buffer
        
        if len(self._buffer) < self.size:
            return
        
        assert len(self._buffer) == self.size

        sequence = self._buffer.copy()

        # add data to store
        data_id = self.data_id_func()
        if data_id not in self._data:
            self._data[data_id] = EventData()
        if self.pick_action == 'append':
            self._data[data_id].added.append(sequence)
        else: # overwrite => one event per trial
            self._data[data_id].added = [sequence]

        print(self.name, 'add', data_id, sequence)
        self._buffer = []
        self.update_display()
    
    def remove(self, event):
        """
        For events of length > 1, remove by removing the first element in that sequence.
        """
        if event.xdata is None:
            return
        t_marked = float(event.xdata)
        data_id = self.data_id_func()
        if data_id not in self._data:
            return
        ev = self._data[data_id]
        
        added_start_times = [x[0] for x in ev.added]
        default_start_times = [x[0] for x in ev.default]
        sequence = None # data that was removed
        _removed = False
        _deleted = False
        if len(ev.added) > 0 and len(ev.default) > 0:
            idx_add, val_add = pn.find_nearest(added_start_times, t_marked)
            idx_def, val_def = pn.find_nearest(default_start_times, t_marked)
            
            add_dist = np.abs(val_add-t_marked)
            def_dist = np.abs(val_def-t_marked)
            if (add_dist <= def_dist) and (self.win_remove[0] < add_dist < self.win_remove[1]):
                sequence = ev.added.pop(idx_add)
                _deleted = True
            if (def_dist < add_dist) and (self.win_remove[0] < def_dist < self.win_remove[1]):
                ev.removed.append(sequence := ev.default.pop(idx_def))
                _removed = True
        elif len(ev.added) > 0 and len(ev.default) == 0:
            idx_add, val_add = pn.find_nearest(added_start_times, t_marked)
            add_dist = np.abs(val_add-t_marked)
            if self.win_remove[0] < add_dist < self.win_remove[1]:
                sequence = ev.added.pop(idx_add)
                _deleted = True
        elif len(ev.added) == 0 and len(ev.default) > 0:
            idx_def, val_def = pn.find_nearest(default_start_times, t_marked)
            def_dist = np.abs(val_def-t_marked)
            if self.win_remove[0] < def_dist < self.win_remove[1]:
                ev.removed.append(sequence := ev.default.pop(idx_def))
                _removed = True
        else:
            return
        
        if sequence is None:
            return
        
        assert _removed is not _deleted # removed moves data from default (i.e. auto-detected) to removed, and delete expunges a manually added event
        print(self.name, {True: 'remove', False: 'delete'}[_removed], data_id, sequence)
        self.update_display()
    
    def get_current_event_times(self):
        return list(np.array(self._data.get(self.data_id_func(), EventData()).get_times()).flatten())
    
    def _get_display_funcs(self):
        display_type = self.plot_kwargs.get('display_type', 'line')
        assert display_type in ('line', 'fill')
        if display_type == 'fill':
            assert self.size == 2
        if display_type == 'line':
            return self._setup_display_line, self._update_display_line
        return self._setup_display_fill, self._update_display_fill

    def setup_display(self): # setup event display this event on one or more axes
        setup_func, _ = self._get_display_funcs()
        setup_func()

    def _setup_display_line(self):
        plot_kwargs = {'label':f'event:{self.name}'} | self.plot_kwargs
        plot_kwargs.pop('display_type', None)
        for ax in self.ax_list:
            this_plot, = ax.plot([], [], color=self.color, **plot_kwargs)
            self.plot_handles.append(this_plot)

    def _setup_display_fill(self):
        return # everything is redrawm currently for fill display. So, don't do setup.

    def update_display(self, draw=True):
        _, update_func = self._get_display_funcs()
        update_func(draw)
    
    def _get_ylim(self, this_ax, type='data'):
        if type == 'data':
            try:
                x = np.asarray([(np.nanmin(line.get_ydata()), np.nanmax(line.get_ydata())) for line in this_ax.get_lines() if not line.get_label().startswith('event:')])
                return np.min(x[:, 0]), np.max(x[:, 1])
            except ValueError:
                return this_ax.get_ylim()
        return this_ax.get_ylim()

    def _update_display_line(self, draw):
        for ax, plot_handle in zip(self.ax_list, self.plot_handles):
            yl = self._get_ylim(ax)
            plot_handle.set_data(*pn.ticks_from_times(self.get_current_event_times(), yl))
        if draw:
            plt.draw()
    
    def _update_display_fill(self, draw):
        if self._hide:
            return
        for plot_handle in self.plot_handles:
            plot_handle.remove()
        self.plot_handles = []
        plot_kwargs = dict(alpha=0.2, edgecolor=None) | self.plot_kwargs
        plot_kwargs.pop('display_type', None)
        for ax in self.ax_list:
            yl = self._get_ylim(ax)
            x = np.asarray([this_times + [np.nan] for this_times in self._data.get(self.data_id_func(), EventData()).get_times()]).flatten()
            y1 = np.asarray([[yl[0]]*2 + [np.nan] for _ in range(int(len(x)/3))]).flatten()
            y2 = np.asarray([[yl[1]]*2 + [np.nan] for _ in range(int(len(x)/3))]).flatten()
            this_collection = ax.fill_between(x, y1, y2, color=self.color, **plot_kwargs)
            self.plot_handles.append(this_collection)
        if draw:
            plt.draw()
    
    def to_dict(self):
        event_data = self._data
        if self.pick_action == 'overwrite':
            ret = {k:v.get_times()[0] for k,v in event_data.items()}
        else:
            ret = {k:v.get_times() for k,v in event_data.items()}
        return ret
    
    def to_portions(self):
        assert self.size == 2
        P = pn.portion
        ret = {}
        for signal_id, signal_events in self.to_dict().items():
            ret[signal_id] = functools.reduce(lambda a,b: a|b, [P.closed(*interval_limits) for interval_limits in signal_events])
        return ret
    

class Events:
    def __init__(self, parent):
        self._list : list = [] # list of some type of event
        self.parent = parent
        self._text = None

    def __len__(self):
        return len(self._list)
    
    def __getitem__(self, key):
        """return the state variable by name key"""
        assert key in self.names
        return {x.name:x for x in self._list}[key]
    
    @property
    def names(self):
        return [x.name for x in self._list]
    
    def add(self, 
            name,
            size, 
            fname, 
            data_id_func, 
            color, 
            pick_action='overwrite', 
            ax_list=None, 
            win_remove=(-0.1, 0.1),
            win_add=(-0.25, 0.25),
            add_key=None,
            remove_key=None,
            save_key=None,
            show=True,
            **plot_kwargs):
        assert name not in self.names
        this_ev = Event(name, size, fname, data_id_func, color, pick_action, ax_list, win_remove, win_add, **plot_kwargs)
        self._list.append(this_ev)
        if add_key is not None:
            self.parent.add_key_binding(add_key, this_ev.add, f'Add {name}')
        if remove_key is not None:
            self.parent.add_key_binding(remove_key, this_ev.remove, f'Remove {name}')
        if save_key is not None:
            self.parent.add_key_binding(save_key, this_ev.save, f'Save {name}')
        if show:
            this_ev.setup_display()
        else:
            this_ev._hide = True # This is for fill displays
    
    def add_from_file(self, fname, data_id_func, ax_list=None, add_key=None, remove_key=None, save_key=None, show=True, **plot_kwargs):
        """Easier than using add for adding events that are created by another algorithm, and meant to be edited using the gui module."""
        assert os.path.exists(fname)
        ev = Event._from_existing_file(fname)
        hdr = ev.get_header()
        del hdr['all_keys_are_tuples']
        plot_kwargs = hdr['plot_kwargs'] | plot_kwargs
        del hdr['plot_kwargs']
        self.add(data_id_func=data_id_func, ax_list=ax_list, add_key=add_key, remove_key=remove_key, save_key=save_key, show=show, **(hdr | plot_kwargs))
    
    def setup_display(self):
        for ev in self._list:
            ev.setup_display()

    def update_display(self, draw=True):
        for ev in self._list:
            ev.update_display(draw=False)
        if draw:
            plt.draw()

class GenericBrowser:
    """
    Generic class to browse data. Meant to be extended before use.
    Features:
        Navigate using arrow keys.
        Store positions in memory using number keys (e.g. for flipping between positions when browsing a video).
        Quickly add toggle and push buttons.
        Design custom functions and assign hotkeys to them (add_key_binding)

    Default Navigation (arrow keys):
        ctrl+k      - show all keybindings
        right       - forward one frame
        left        - back one frame
        up          - forward 10 frames
        down        - back 10 frames
        shift+left  - first frame
        shift+right - last frame
        shift+up    - forward nframes/20 frames
        shift+down  - back nframes/20 frames
    """ 
    def __init__(self, figure_handle=None):
        if figure_handle is None:
            figure_handle = plt.figure()
        assert isinstance(figure_handle, plt.Figure)
        self.figure = figure_handle
        self._keypressdict = {} # managed by add_key_binding
        self._bindings_removed = {}
        
        # tracking variable
        self._current_idx = 0

        self._keybindingtext = None
        self.buttons = Buttons(parent=self)
        self.selectors = Selectors(parent=self)
        self.memoryslots = MemorySlots(parent=self)
        self.statevariables = StateVariables(parent=self)
        self.events = Events(parent=self)

        # for cleanup
        self.cid = []
        self.cid.append(self.figure.canvas.mpl_connect('key_press_event', self))
        self.cid.append(self.figure.canvas.mpl_connect('close_event', self))
    
    def update_assets(self):
        if self.has('memoryslots'):
            self.memoryslots.update_display()
        if self.has('events'):
            self.events.update_display()
        if self.has('statevariables'):
            self.statevariables.update_display()

    def update(self, event=None): # extended classes are expected to implement their update function!
        self.update_assets()
    
    def update_without_clear(self, event=None):
        self.update_assets()
        # I did this for browsers that clear the axis each time! Those classes will need to re-implement this method

    def mpl_remove_bindings(self, key_list):
        """If the existing key is bound to something in matplotlib, then remove it"""
        for key in key_list:
            this_param_name = [k for k, v in mpl.rcParams.items() if isinstance(v, list) and key in v]
            if this_param_name: # not an empty list
                assert len(this_param_name) == 1
                this_param_name = this_param_name[0]
                mpl.rcParams[this_param_name].remove(key)
                self._bindings_removed[this_param_name] = key
    
    def __call__(self, event):
        # print(event.__dict__) # for debugging
        if event.name == 'key_press_event':
            if event.key in self._keypressdict:
                f = self._keypressdict[event.key][0]
                argspec = inspect.getfullargspec(f)[0]
                if len(argspec) == 2 and argspec[1] == 'event':
                    f(event)
                else:
                    f() # this may or may not redraw everything
            if event.key in self.memoryslots._idx:
                self.memoryslots.update(event.key)
        elif event.name == 'close_event': # perform cleanup
            self.cleanup()
    
    def cleanup(self):
        """Perform cleanup, for example, when the figure is closed."""
        for this_cid in self.cid:
            self.figure.canvas.mpl_disconnect(this_cid)
        self.mpl_restore_bindings()

    def mpl_restore_bindings(self):
        """Restore any modified default keybindings in matplotlib"""
        for param_name, key in self._bindings_removed.items():
            if key not in mpl.rcParams[param_name]:
                mpl.rcParams[param_name].append(key) # param names: keymap.back, keymap.forward)
            self._bindings_removed[param_name] = {}

    def __len__(self):
        if hasattr(self, 'data'): # otherwise returns None
            return len(self.data)

    def reset_axes(self, event=None): # event in case it is used as a callback function
        """Reframe data within matplotlib axes."""
        for ax in self.figure.axes:
            if isinstance(ax, maxes.SubplotBase):
                ax.relim()
                ax.autoscale()
        plt.draw()

    ## select plots

    # Event responses - useful to pair with add_key_binding
    # These capabilities can be assigned to different key bindings
    def add_key_binding(self, key_name, on_press_function, description=None):
        """
        This is useful to add key-bindings in classes that inherit from this one, or on the command line.
        See usage in the __init__ function
        """
        if description is None:
            description = on_press_function.__name__
        self.mpl_remove_bindings([key_name])
        self._keypressdict[key_name] = (on_press_function, description)

    def set_default_keybindings(self):
        self.add_key_binding('left', self.decrement)
        self.add_key_binding('right', self.increment)
        self.add_key_binding('up', (lambda s: s.increment(step=10)).__get__(self), description='increment by 10')
        self.add_key_binding('down', (lambda s: s.decrement(step=10)).__get__(self), description='decrement by 10')
        self.add_key_binding('shift+left', self.decrement_frac, description='step forward by 1/20 of the timeline')
        self.add_key_binding('shift+right', self.increment_frac, description='step backward by 1/20 of the timeline')
        self.add_key_binding('shift+up', self.go_to_start)
        self.add_key_binding('shift+down', self.go_to_end)
        self.add_key_binding('ctrl+c', self.copy_to_clipboard)
        self.add_key_binding('ctrl+k', (lambda s: s.show_key_bindings(f='new', pos='center left')).__get__(self), description='show key bindings')
        self.add_key_binding('/', (lambda s: s.pan(direction='right')).__get__(self), description='pan right')
        self.add_key_binding(',', (lambda s: s.pan(direction='left')).__get__(self), description='pan left')
        self.add_key_binding('l', (lambda s: s.pan(direction='up')).__get__(self), description='pan up')
        self.add_key_binding('.', (lambda s: s.pan(direction='down')).__get__(self), description='pan down')
        self.add_key_binding('r', self.reset_axes)
    
    def increment(self, step=1):
        self._current_idx = min(self._current_idx+step, len(self)-1)
        self.update()

    def decrement(self, step=1):
        self._current_idx = max(self._current_idx-step, 0)
        self.update()
    
    def go_to_start(self): # default: shift+left
        self._current_idx = 0
        self.update()
    
    def go_to_end(self):
        self._current_idx = len(self)-1
        self.update()
    
    def increment_frac(self, n_steps=20):
        # browse entire dataset in n_steps
        self._current_idx = min(self._current_idx+int(len(self)/n_steps), len(self)-1)
        self.update()
    
    def decrement_frac(self, n_steps=20):
        self._current_idx = max(self._current_idx-int(len(self)/n_steps), 0)
        self.update()
    
    def copy_to_clipboard(self):
        from PySide2.QtGui import QClipboard, QImage
        buf = io.BytesIO()
        self.figure.savefig(buf)
        QClipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    def show_key_bindings(self, f=None, pos='bottom right'):
        f = {None: self.figure, 'new': plt.figure()}[f]
        text = []
        for shortcut, (_, description) in self._keypressdict.items():
            text.append(f'{shortcut:<12} - {description}')
        self._keybindingtext = TextView(text, f, pos=pos)
    
    @staticmethod
    def _filter_sibling_axes(ax, share='x', get_bool=False):
        """Given a list of matplotlib axes, it will return axes to manipulate by picking one from a set of siblings"""
        assert share in ('x', 'y')
        if isinstance(ax, maxes.Axes): # only one axis
            return [ax]
        ax = [tax for tax in ax if isinstance(tax, maxes.SubplotBase)]
        if not ax: # no subplots in figure
            return
        pan_ax = [True]*len(ax)
        get_siblings = {'x': ax[0].get_shared_x_axes, 'y': ax[0].get_shared_y_axes}[share]().get_siblings
        for i, ax_row in enumerate(ax):
            sib = get_siblings(ax_row)
            for j, ax_col in enumerate(ax[i+1:]):
                if ax_col in sib:
                    pan_ax[j+i+1] = False

        if get_bool:
            return pan_ax
        return [this_ax for this_ax, this_tf in zip(ax, pan_ax) if this_tf]
        
    def pan(self, direction='left', frac=0.2):
        assert direction in ('left', 'right', 'up', 'down')
        if direction in ('left', 'right'):
            pan_ax='x'
        else:
            pan_ax='y'
        ax = self._filter_sibling_axes(self.figure.axes, share=pan_ax, get_bool=False)
        if ax is None:
            return
        for this_ax in ax:
            lim1, lim2 = {'x': this_ax.get_xlim, 'y': this_ax.get_ylim}[pan_ax]()
            inc = (lim2-lim1)*frac
            if direction in ('down', 'right'):
                new_lim = (lim1+inc, lim2+inc)
            else:
                new_lim = (lim1-inc, lim2-inc)
            {'x': this_ax.set_xlim, 'y': this_ax.set_ylim}[pan_ax](new_lim)
        plt.draw()
        self.update_without_clear() # panning is pointless if update clears the axis!!
    
    def has(self, asset_type): # e.g. has('events')
        assert asset_type in ('buttons', 'selectors', 'memoryslots', 'statevariables', 'events')
        return len(getattr(self, asset_type)) != 0


class PlotBrowser(GenericBrowser):
    """
    Takes a list of data, and a plotting function (or a pair of setup
    and update functions) that parses each of the elements in the array.
    Assumes that the plotting function is going to make one figure.
    """
    def __init__(self, plot_data, plot_func, figure_handle=None, **plot_kwargs):
        """
            plot_data - list of data objects to browse

            plot_func can be a tuple (setup_func, update_func), or just one plotting function - update_func
                If only one function is supplied, figure axes will be cleared on each update.
                setup_func takes:
                    the first element in plot_data list as its first input
                    keyword arguments (same as plot_func)
                setup_func outputs:
                    **dictionary** of plot handles that goes as the second input to update_func

                update_func is a plot-refreshing function that accepts 3 inputs:
                    an element in the plot_data list as its first input
                    output of the setup_func if it exists, or a figure handle on which to plot
                    keyword arguments
            
            figure_handle - (default: None) matplotlib figure handle within which to instantiate the browser
                Ideally, the setup function will handle this

            plot_kwargs - these keyword arguments will be passed to plot_function after data and figure
        """
        self.data = plot_data # list where each element serves as input to plot_func
        self.plot_kwargs = plot_kwargs

        if isinstance(plot_func, tuple):
            assert len(plot_func) == 2
            self.setup_func, self.plot_func = plot_func            
            self.plot_handles = self.setup_func(self.data[0], **self.plot_kwargs)
            plot_handle = list(self.plot_handles.values())[0]
            if 'figure' in self.plot_handles:
                figure_handle = self.plot_handles['figure']
            elif isinstance(plot_handle, list):
                figure_handle = plot_handle[0].figure
            else:
                figure_handle = plot_handle.figure # figure_handle passed as input will be ignored
        else:
            self.setup_func, self.plot_func = None, plot_func
            self.plot_handles = None
            figure_handle = figure_handle

        # setup
        super().__init__(figure_handle)

        # initialize
        self.set_default_keybindings()
        self.buttons.add(text='Auto limits', type_='Toggle', action_func=self.update, start_state=False)
        self.memoryslots.show()
        if self.__class__.__name__ == 'PlotBrowser': # if an inherited class is accessing this, then don't run the update function here
            self.update() # draw the first instance
            self.reset_axes()
            plt.show(block=False)
        # add selectors after drawing!
        try:
            s0 = self.selectors.add(list(self.plot_handles.values())[0])
            self.buttons.add(text='Selector 0', type_='Toggle', action_func=s0.toggle, start_state=s0.is_active)
        except AssertionError:
            print('Unable to add selectors')

    def get_current_data(self):
        return self.data[self._current_idx]

    def update(self, event=None): # event = None lets this function be attached as a callback
        if self.setup_func is None:
            self.figure.clear() # redraw the entire figure contents each time, NOT recommended
            self.memoryslots.show()
            self.plot_func(self.get_current_data(), self.figure, **self.plot_kwargs)
        else:
            self.memoryslots.update_display()
            self.plot_func(self.get_current_data(), self.plot_handles, **self.plot_kwargs)
        if self.buttons['Auto limits'].state: # is True
            self.reset_axes()
        super().update(event)
        plt.draw()
    
    def udpate_without_clear(self):
        self.memoryslots.update_display()
        plt.draw()


class SignalBrowser(GenericBrowser):
    """
    Browse an array of pntools.Sampled.Data elements, or 2D arrays
    """
    def __init__(self, plot_data, titlefunc=None, figure_handle=None, reset_on_change=False):
        super().__init__(figure_handle)

        self._ax = self.figure.subplots(1, 1)
        this_data = plot_data[0]
        if isinstance(this_data, sampled.Data):
            self._plot = self._ax.plot(this_data.t, this_data())
        else:
            self._plot = self._ax.plot(this_data)

        self.data = plot_data
        if titlefunc is None:
            if hasattr(self.data[0], 'name'):
                self.titlefunc=lambda s: f'{s.data[s._current_idx].name}'
            else:
                self.titlefunc = lambda s: f'Plot number {s._current_idx}'
        else:
            self.titlefunc = titlefunc

        self.reset_on_change = reset_on_change
        # initialize
        self.set_default_keybindings()
        self.buttons.add(text='Auto limits', type_='Toggle', action_func=self.update, start_state=False)
        plt.show(block=False)
        self.update()
    
    def update(self, event=None):
        this_data = self.data[self._current_idx]
        if isinstance(this_data, sampled.Data):
            data_to_plot = this_data.split_to_1d()
            for plot_handle, this_data_to_plot in zip(self._plot, data_to_plot):
                plot_handle.set_data(this_data_to_plot.t, this_data_to_plot())
        else:
            self._plot.set_ydata()
        self._ax.set_title(self.titlefunc(self))
        if self.buttons['Auto limits'].state: # is True
            self.reset_axes()
        plt.draw()

class TestIntervalEvents(SignalBrowser):
    def __init__(self):
        plot_data = [sampled.Data(np.random.rand(100), sr=10, meta={'id': f'sig{sig_count:02d}'}) for sig_count in range(10)]
        super().__init__(plot_data)
        self.memoryslots.disable()
        data_id_func = (lambda s: s.data[s._current_idx].meta['id']).__get__(self)
        self.events.add(
            name='pick1',
            size=1,
            fname=r'C:\data\_cache\_pick1.json',
            data_id_func = data_id_func,
            color = 'tab:red',
            pick_action = 'append',
            ax_list = [self._ax],
            add_key='1',
            remove_key='4',
            save_key='ctrl+1',
            linewidth=1.5,
        )
        self.events.add(
            name='pick2',
            size=2,
            fname=r'C:\data\_cache\_pick2.json',
            data_id_func = data_id_func,
            color = 'tab:green',
            pick_action = 'append',
            ax_list = [self._ax],
            add_key='2',
            remove_key='5',
            save_key='ctrl+2',
            linewidth=1.5,
        )
        self.events.add(
            name='pick3',
            size=3,
            fname=r'C:\data\_cache\_pick3.json',
            data_id_func = data_id_func,
            color = 'tab:blue',
            pick_action = 'overwrite',
            ax_list = [self._ax],
            add_key='3',
            remove_key='6',
            save_key='ctrl+3',
            linewidth=1.5,
        )
        self.update()

    def update(self, event=None):
        self.events.update_display()
        return super().update(event)

class VideoBrowser(GenericBrowser):
    """Scroll through images of a video

        If figure_handle is an axis handle, the video will be plotted in that axis.
    """
    def __init__(self, vid_name, titlefunc=None, figure_or_ax_handle=None):
        from decord import VideoReader
        assert isinstance(figure_or_ax_handle, (plt.Axes, plt.Figure, type(None)))
        if isinstance(figure_or_ax_handle, plt.Axes):
            figure_handle = figure_or_ax_handle.figure
            ax_handle = figure_or_ax_handle
        else: # this is the same if figure_or_ax_handle is none or a figure handle
            figure_handle = figure_or_ax_handle
            ax_handle = None
        super().__init__(figure_handle)

        if not os.path.exists(vid_name): # try looking in the CLIP FOLDER
            vid_name = os.path.join(CLIP_FOLDER, os.path.split(vid_name)[-1])        
        assert os.path.exists(vid_name)
        self.fname = vid_name
        self.name = os.path.splitext(os.path.split(vid_name)[1])[0]
        with open(vid_name, 'rb') as f:
            self.data = VideoReader(f)
        
        if ax_handle is None:
            self._ax = self.figure.subplots(1, 1)
        else:
            assert isinstance(ax_handle, plt.Axes)
            self._ax = ax_handle
        this_data = self.data[0]
        self._im = self._ax.imshow(this_data.asnumpy())
        self._ax.axis('off')

        self.fps = self.data.get_avg_fps()
        if titlefunc is None:
            self.titlefunc = lambda s: f'Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}'
        
        self.set_default_keybindings()
        self.add_key_binding('e', self.extract_clip)
        self.memoryslots.show(pos='bottom left')

        if self.__class__.__name__ == 'VideoBrowser': # if an inherited class is accessing this, then don't run the update function here
            plt.show(block=False)
            self.update()

    def update(self):
        self._im.set_data(self.data[self._current_idx].asnumpy())
        self._ax.set_title(self.titlefunc(self))
        super().update() # updates memory slots
        plt.draw()

    def extract_clip(self, start_frame=None, end_frame=None, fname_out=None, out_rate=None):
        #TODO: For musicrunning, grab the corresponding audio and add the audio track to the video clip?
        try:
            import ffmpeg
            use_subprocess = False
        except ModuleNotFoundError:
            import subprocess
            use_subprocess = True

        if start_frame is None:
            start_frame = self.memoryslots._idx['1']
        if end_frame is None:
            end_frame = self.memoryslots._idx['2']
        assert end_frame > start_frame
        start_time = float(start_frame)/self.fps
        end_time = float(end_frame)/self.fps
        dur = end_time - start_time
        if out_rate is None:
            out_rate = self.fps
        if fname_out is None:
            fname_out = os.path.join(CLIP_FOLDER, os.path.splitext(self.name)[0] + '_s{:.3f}_e{:.3f}.mp4'.format(start_time, end_time))
        if use_subprocess:
            subprocess.getoutput(f'ffmpeg -ss {start_time} -i "{self.fname}" -r {out_rate} -t {dur} -vcodec h264_nvenc "{fname_out}"')
        else:
            ffmpeg.input(self.fname, ss=start_time).output(fname_out, vcodec='h264_nvenc', t=dur, r=out_rate).run()
        return fname_out


class VideoPlotBrowser(GenericBrowser):
    def __init__(self, vid_name:str, signals:dict, titlefunc=None, figure_handle=None, event_win=None):
        """
        Browse a video and an array of sampled.Data side by side. 
        Assuming that the time vectors are synchronized across the video and the signals, there will be a black tick at the video frame being viewed.
        Originally created to browse montage videos from optitrack alongside physiological signals from delsys. 
        For example, see projects.fencing.snapshots.browse_trial

        signals is a {signal_name<str> : signal<pntools.sampled.Data>}
        """
        from decord import VideoReader
        figure_handle = plt.figure(figsize=(20, 12))
        super().__init__(figure_handle)

        self.event_win = event_win
        
        self.vid_name = vid_name
        assert os.path.exists(vid_name)
        self.name = os.path.splitext(os.path.split(vid_name)[1])[0]
        with open(vid_name, 'rb') as f:
            self.video_data = VideoReader(f)
        self.fps = self.video_data.get_avg_fps()
        
        self.signals = signals
        if titlefunc is None:
            self.titlefunc = lambda s: f'Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}'

        self.plot_handles = self._setup()
        self.plot_handles['ax']['montage'].set_axis_off()

        self.set_default_keybindings()
        self.add_key_binding('e', self.extract_clip)
        self._len = len(self.video_data)
        self.memoryslots.show(pos='bottom left')

        self.figure.canvas.mpl_connect('button_press_event', self.onclick)
        
        plt.show(block=False)
        self.update()
    
    def __len__(self):
        return self._len

    def _setup(self):
        fig = self.figure
        gs = GridSpec(nrows=len(self.signals), ncols=2, width_ratios=[2, 3])
        ax = {}
        plot_handles = {}
        for signal_count, (signal_name, this_signal) in enumerate(self.signals.items()):
            this_ax = fig.add_subplot(gs[signal_count, 1])
            plot_handles[f'signal{signal_count}'] = this_ax.plot(this_signal.t, this_signal())
            ylim = this_ax.get_ylim()
            plot_handles[f'signal{signal_count}_tick'], = this_ax.plot([0, 0], ylim, 'k')
            this_ax.set_title(signal_name)
            if signal_count < len(self.signals)-1:
                this_ax.get_xaxis().set_ticks([])
            else:
                this_ax.set_xlabel('Time (s)')
            ax[f'signal{signal_count}'] = this_ax

        ax['montage'] = fig.add_subplot(gs[:, 0])
        plot_handles['montage'] = ax['montage'].imshow(self.video_data[0].asnumpy())
        plot_handles['ax'] = ax
        plot_handles['fig'] = fig
        signal_ax = [v for k,v in plot_handles['ax'].items() if 'signal' in k]
        signal_ax[0].get_shared_x_axes().join(*signal_ax)
        plot_handles['signal_ax'] = signal_ax
        return plot_handles
    
    def update(self):
        self.plot_handles['montage'].set_data(self.video_data[self._current_idx].asnumpy())
        self.plot_handles['ax']['montage'].set_title(self.titlefunc(self))
        for signal_count, this_signal in enumerate(self.signals.items()):
            # ylim = self.plot_handles['ax'][f'signal{signal_count}'].get_ylim()
            self.plot_handles[f'signal{signal_count}_tick'].set_xdata([self._current_idx/self.fps]*2)
        if self.event_win is not None:
            curr_t = self._current_idx/self.fps
            self.plot_handles['signal_ax'][0].set_xlim(curr_t+self.event_win[0], curr_t+self.event_win[1])
        super().update()
        plt.draw()

    def onclick(self, event):
        """Right click mouse to seek to that frame."""
        this_frame = round(event.xdata*self.fps)

        # Right click to seek
        if isinstance(this_frame, (int, float)) and (0 <= this_frame < self._len) and (str(event.button) == 'MouseButton.RIGHT'):
            self._current_idx = this_frame
            self.update()
    
    def extract_clip(self, start_frame=None, end_frame=None, sav_dir=None, out_rate=30):
        """Save a video of screengrabs"""
        import shutil
        import subprocess
        if start_frame is None:
            start_frame = self.memoryslots._idx['1']
        if end_frame is None:
            end_frame = self.memoryslots._idx['2']
        assert end_frame > start_frame
        if sav_dir is None:
            sav_dir = os.path.join(CLIP_FOLDER, datetime.now().strftime("%Y%m%d_%H%M%S"))
        if not os.path.exists(sav_dir):
            os.mkdir(sav_dir)
        print(f"Saving image sequence to {sav_dir}...")
        for frame_count in range(start_frame, end_frame+1):
            self._current_idx = frame_count
            self.update()
            self.figure.savefig(os.path.join(sav_dir, f'{frame_count:08d}.png'))
        print("Creating video from image sequence...")
        cmd = f'cd "{sav_dir}" && ffmpeg -framerate {self.fps} -start_number 0 -i %08d.png -c:v h264_nvenc -b:v 10M -maxrate 12M -bufsize 24M -vf scale="-1:1080" -an "{sav_dir}.mp4"'
        subprocess.getoutput(cmd)

        print("Removing temporary folder...")
        shutil.rmtree(sav_dir)
        
        print("Done")
        # vid_name = os.path.join(Path(sav_dir).parent)
        # f"ffmpeg -framerate {out_rate} -start_number {start_frame} -i "


class VideoPointAnnotator(VideoBrowser):
    """Add point annotations to videos.
    Use arrow keys to navigate frames.
    Select a 'label category' from 0 to 9 by pressing the corresponding number key.
    Point your mouse at a desired location in the video and press the forward slash / button to add a point annotation.
    When you're done, press 's' to save your work, which will create a '<video name>_annotatoins.json' file in the same folder as the video file.
    These annotations will be automagically loaded when you try to annotate this file again.

    If you're doing one label at a time, then pick the frames for the first label arbitrarily.
    For the second label onwards, 
    """
    def __init__(self, vid_name, titlefunc=None):
        figure_handle, _ax = plt.subplots(1, 1) # height_ratiors=[4,1,1]
        super().__init__(vid_name, titlefunc, _ax)
        self.memoryslots.hide()
        self.memoryslots.disable()

        self.fname_annotations = os.path.join(Path(self.fname).parent, Path(self.fname).stem + '_annotations.json')
        self.annotations = VideoAnnotation(fname=self.fname_annotations, vname=self.fname)
        self.add_key_binding('s', on_press_function=self.annotations.save)

        self._current_label = '0'
        self._current_label_text = TextView([f'Current label: {self._current_label}'], self.figure, pos='bottom center')

        self.plot_handles = {}
        self.palette = self.get_default_color_palette(len(self.annotations))
        for label, color in zip(self.annotations.labels, self.palette):
            self.plot_handles[f'label_{label}'], = self._ax.plot([], [], 'o', color=color)
            self.plot_handles[f'trace_{label}'], = self._ax.plot([], [], color=color)

        self.add_key_binding('/', self.add_annotation)
        self.add_key_binding('t', self.add_annotation)
        self.add_key_binding('.', self.remove_annotation)
        self.add_key_binding('y', self.remove_annotation)

        self.add_key_binding('n', self.next_annotation)
        self.add_key_binding('p', self.previous_annotation)
        self.add_key_binding('f', self.increment_if_unannotated)
        self.add_key_binding('g', self.increment)
        self.add_key_binding('d', self.decrement_if_unannotated)

        plt.show(block=False)
        self.update()
    
    @staticmethod
    def get_default_color_palette(n_colors=10):
        return [ # seaborn set 2
            (0.40, 0.76, 0.65),
            (0.99, 0.55, 0.38),
            (0.55, 0.63, 0.79),
            (0.91, 0.54, 0.76),
            (0.65, 0.85, 0.33),
            (1.00, 0.85, 0.18),
            (0.90, 0.77, 0.58),
            (0.70, 0.70, 0.70),
            (0.40, 0.76, 0.65),
            (0.99, 0.55, 0.38)
            ][:n_colors]
    
    def _parse_labels_pos(self, labels_pos):
        if labels_pos is None:
            labels_pos = {}
        
        ret = {}
        for this_label in self.annotations.labels:
            if this_label in labels_pos:
                ret[this_label] = labels_pos[this_label]
            else:
                ret[this_label] = np.array([[np.nan, np.nan]])
        return ret
    
    def __call__(self, event):
        super().__call__(event)
        if event.name == 'key_press_event' and event.key in self.annotations.labels:
            self._current_label = event.key
            self.update_current_label_text(draw=True)
    
    def update(self):
        self.update_annotation_display(draw=False)
        self.update_current_label_text(draw=False)
        super().update()
        self.reset_axes()
    
    def update_annotation_display(self, draw=False):
        for label, annot_dict in self.annotations.data.items():
            this_data = ([], [])
            frame_number = self._current_idx
            if frame_number in annot_dict: 
                this_data = annot_dict[frame_number]
            self.plot_handles[f'label_{label}'].set_data(*this_data)
        if draw:
            plt.draw()
    
    def update_current_label_text(self, draw=False):
        self._current_label_text.update([f'Current label: {self._current_label}'])
        if draw:
            plt.draw()

    def add_annotation(self, event):
        # Add annotation at frame. If it exists, it'll get overwritten.
        self.annotations.data[self._current_label][self._current_idx] = [float(event.xdata), float(event.ydata)]
        self.update()
    
    def remove_annotation(self, event=None):
        # remove annotation at the current frame if it exists
        self.annotations.data[self._current_label].pop(self._current_idx, None)
        self.update()
    
    def next_annotation(self, event=None):
        try:
            current_frame = self._current_idx
            self._current_idx = min([x for x in self.annotations.get_frames(self._current_label) if x > current_frame])
            self.update()
        except ValueError:
            return
    
    def previous_annotation(self, event=None):
        try:
            current_frame = self._current_idx
            self._current_idx = max([x for x in self.annotations.get_frames(self._current_label) if x < current_frame])
            self.update()
        except ValueError:
            return

    def increment_if_unannotated(self, event=None):
        if self._current_idx not in self.annotations.frames:
            self.increment()
    
    def decrement_if_unannotated(self, event=None):
        if self._current_idx not in self.annotations.frames:
            self.decrement()


class VideoAnnotation:
    """Manage point annotations in video.

    Args:
        fname (str, optional): File name of the annotations (.json) file. If it
            doesn't exist, it will be created when save method is used. If this is a
            video file, fname will default to <video_name>_annotations.json.
            Defaults to None.
        vname (str, optional): Name of the video being annotated. Defaults to None.
    
    Methods:
        from_dlc : Load data from a deeplabcut .h5 file
        to_dlc: Convert from json file format into a deeplabcut dataframe format, and optionally save the file.
    
    """
    def __init__(self, fname: str=None, vname: str=None):
        self.fname, vname = self._parse_inp(fname, vname)

        if self.fname is not None:
            self.name = Path(fname).stem
        else:
            self.name = None

        if video.is_video(vname):
            self.video = video.Video(vname)
        else:
            self.video = None
        
        self.data = self.load()

    @staticmethod
    def _parse_inp(fname_inp, vname_inp):
        if fname_inp is None and vname_inp is None:
            fname, vname = fname_inp, vname_inp # do nothing, empty annotation
        elif fname_inp is not None and vname_inp is None:
            if video.is_video(fname_inp):
                vname = fname_inp
                fname = os.path.join(Path(vname_inp).parent, Path(vname_inp).stem + '_annotations.json')
            else:
                fname, vname = fname_inp, vname_inp # do nothing, just for code readability
        elif fname_inp is None and vname_inp is not None:
            assert video.is_video(vname_inp)
            vname = vname_inp
            fname = os.path.join(Path(vname_inp).parent, Path(vname_inp).stem + '_annotations.json')
        elif fname_inp is not None and vname_inp is not None:
            assert video.is_video(vname_inp)
            fname, vname = fname_inp, vname_inp # do nothing
        return fname, vname

    @classmethod
    def from_dlc(cls, dlc_fname, vname=None, remove_label_prefix='point', img_prefix='img', img_suffix='.png'):
        """Load annotations from a deeplabcut h5 file."""
        if isinstance(dlc_fname, pd.DataFrame):
            df = dlc_fname
            fname = None
        else:
            assert os.path.exists(dlc_fname)
            assert Path(dlc_fname).suffix == '.h5'
            df = pd.read_hdf(dlc_fname)
            fname = Path(dlc_fname).with_suffix('.json')
            if os.path.exists(fname):
                fname = None
        
        obj = cls(fname, vname)
        
        obj.data = obj._dlc_df_to_annotation_dict(df, remove_label_prefix, img_prefix, img_suffix)
        return obj
    
    @staticmethod
    def _dlc_df_to_annotation_dict(df, remove_label_prefix='point', img_prefix='img', img_suffix='.png'):
        labels = [x.removeprefix(remove_label_prefix) for x in df.columns.levels[1]]
        frames_str = [x.removeprefix(img_prefix).removesuffix(img_suffix) for x in df.index.levels[-1]]

        data = {label: {} for label in labels}
        video_stem = df.index.levels[1].values[0]
        scorer = df.columns.levels[0].values[0]
        for label in labels:
            for frame_str in frames_str:
                coord_val = [
                    df.loc['labeled-data', video_stem, f'{img_prefix}{frame_str}{img_suffix}']
                        [scorer, f'{remove_label_prefix}{label}', coord_name] 
                    for coord_name in ('x', 'y')
                    ]
                if np.all(np.isnan(coord_val)):
                    continue
                data[label][int(frame_str)] = coord_val
        
        return data
    
    def __len__(self):
        """Number of annotations"""
        return len(self.data)
    
    @property
    def n_frames(self):
        """Number of frames in the video being annotated"""
        if self.video is None:
            return None
        return len(self.video)
    
    @property
    def n_annotations(self):
        """Number of points being annotated in the video."""
        return len(self)
    
    @property
    def labels(self):
        """Labels of the annotations."""
        return list(self.data.keys())
    
    @property
    def frames(self):
        """Frame numbers in the video that have annotations."""
        ret = list(set([frame for label in self.labels for frame in self.get_frames(label)]))
        ret.sort()
        return ret

    @property
    def frames_overlapping(self):
        """List of frames in the video where all the labels are annotated."""
        ret = list(functools.reduce(set.intersection, [set(self.get_frames(label)) for label in self.labels]))
        ret.sort()
        return ret
    
    def get_frames(self, label):
        """Return a list of frames that are annotated with the current label."""
        assert label in self.labels
        return list(self.data[label].keys())
    
    def load(self, n_annotations=10):
        """Load annotations from a json file, or initialize an annotation dictionary if a file doesn't exist."""
        if self.fname is not None and os.path.exists(self.fname):
            with open(self.fname, 'r') as f:
                ret = {}
                for k,v in json.load(f).items():
                    if v:
                        ret[k] = {int(frame_num):loc for frame_num, loc in v.items()}
                return ret
        return {str(label):{} for label in range(n_annotations)}
        
    def save(self, fname=None):
        """Save the annotations json file. self.fname should be a valid file path."""
        if fname is None:
            assert self.fname is not None
            fname = self.fname
        self.sort_data()
        with open(fname, 'w') as f:
            json.dump(self.data, f, indent=4)
    
    def sort_data(self):
        """Sort annotations by the frame numbers."""
        self.data = {label:dict(sorted(self.data[label].items())) for label in self.labels}
    
    def get_values_cv(self, frame_num: int):
        """Return annotations at frame_num in a format for openCV's optical flow algorithms"""
        return np.array(
            self.get_at_frame(frame_num), 
            dtype=np.float32).reshape((self.n_annotations, 1, 2)
            )
    
    def _n_digits_in_frame_num(self):
        """Number of digits to use when constructing a string from the frame number."""
        if self.n_frames is None:
            return '6'
        return str(len(str(self.n_frames)))

    def _frame_num_as_str(self, frame_num: int):
        """Return the frame umber as a formatted string."""
        return f'{frame_num:0{self._n_digits_in_frame_num()}}'
    
    def add_at_frame(self, frame_num: int, values: np.ndarray):
        """Add annotations at a frame, given the annotation values."""
        assert isinstance(frame_num, int)
        values = np.array(values)
        assert values.shape == (self.n_annotations, 2)
        for label, value in zip(self.labels, values):
            self.data[label][frame_num] = list(value)
    
    def get_at_frame(self, frame_num: int):
        """Retrieve annotations at a given frame number. If an annotation is not present, nan values will be used."""
        ret = []
        for label in self.labels:
            if frame_num in self.data[label]:
                ret.append(self.data[label][frame_num])
            else:
                ret.append([np.nan, np.nan])
        return ret
    
    def __getitem__(self, key):
        """Easy access to specific annotation, or data from a frame number."""
        if key in self.labels:
            return self.data[key]
        if key in self.frames:
            return self.get_at_frame(key)
        raise ValueError(f'{key} is neither an annotation nor a frame with annotation.')
    
    def to_dlc(self, scorer='praneeth', output_path=None, file_prefix=None, img_prefix='img', img_suffix='.png', label_prefix='point', save=True):
        """Save annotations in deeplabcut format."""
        annotations = self.data
        
        if output_path is None:
            output_path = Path(self.fname).parent
        output_path = Path(output_path)

        index_length = self._n_digits_in_frame_num()
        img_stems = [f'{img_prefix}{x:0{index_length}}{img_suffix}' for x in self.frames]
        
        row_idx = pd.MultiIndex.from_tuples([('labeled-data', self.video.name, img_stem) for img_stem in img_stems])
        col_idx = pd.MultiIndex.from_product([[scorer], [f'{label_prefix}{x}' for x in annotations], ['x', 'y']], names = ['scorer', 'bodyparts', 'coords'])
        df = pd.DataFrame([], index=row_idx, columns=col_idx)
        for annotation_label, annotation_dict in annotations.items():
            for frame, xy in annotation_dict.items():
                for coord_name, coord_val in zip(('x', 'y'), xy):
                    df.loc['labeled-data', self.video.name, f'{img_prefix}{frame:0{index_length}}{img_suffix}'][scorer, f'{label_prefix}{annotation_label}', coord_name] = coord_val
        df = df.apply(lambda col:pd.to_numeric(col, errors='coerce'))

        if file_prefix is None:
            file_prefix = self.name
        elif file_prefix == 'dlc': # usual dlc name
            file_prefix = f"CollectedData_{scorer}"
        else:
            assert isinstance(file_prefix, str)

        if save:
            labeled_data_file_prefix = str(output_path / file_prefix)
            df.to_csv(labeled_data_file_prefix + '.csv')
            df.to_hdf(labeled_data_file_prefix + '.h5', key="df_with_missing", mode="w")
        return df


class TextView:
    """Show text array line by line"""
    def __init__(self, text, fax=None, pos='bottom left'):
        """
        text is an array of strings
        fax is either a figure or an axis handle
        """
        def rescale(xy, margin=0.03):
            return (1-2*margin)*xy + margin

        self.text = self.parse_text(text)
        self._text = None # matplotlib text object
        self._pos = _parse_pos(pos)
        self.figure, self._ax = _parse_fax(fax, ax_pos=(rescale(self._pos[0]), rescale(self._pos[1]), 0.02, 0.02))
        self.setup()
        self.update()
    
    def parse_text(self, text):
        if isinstance(text, dict):
            text = [f'{key} - {val}' for key, val in text.items()] 
        return text

    def setup(self):
        self._ax.axis('off')
        plt.show(block=False)

    def update(self, new_text=None):
        if new_text is not None:
            self.text = self.parse_text(new_text)
        if self._text is not None:
            self._text.remove()
        x, y, va, ha = self._pos
        self._text = self._ax.text(x, y, '\n'.join(self.text), va=va, ha=ha, family='monospace')
        plt.draw()

class SignalBrowserKeyPress(SignalBrowser):
    """Wrapper around plot_sync with key press features to make manual alignment process easier"""
    def __init__(self, plot_data, titlefunc=None, figure_handle=None, reset_on_change=False):
        super().__init__(plot_data, titlefunc, figure_handle, reset_on_change)
        self.event_keys = {'1': [], '2':[], '3':[], 't':[], 'd':[]}
    def __call__(self, event):
        from pprint import pprint
        super().__call__(event)
        if event.name == 'key_press_event':
            sr = self.data[self._current_idx].sr
            if event.key in self.event_keys:
                if event.key == '1':
                    self.first = int(float(event.xdata)*sr)
                    self.event_keys[event.key].append(self.first)
                    print(f'first: {self.first}')
                elif event.key == '2':
                    self.second = int(float(event.xdata)*sr)
                    self.event_keys[event.key].append(self.second)
                    print(f'second: {self.second}')
                elif event.key == '3':
                    self.third = int(float(event.xdata)*sr)
                    self.event_keys[event.key].append(self.third)
                    print(f'third: {self.third}')
                elif event.key == 't':
                    pprint(self.event_keys, width=1)
                    self.export = self.event_keys   
                elif event.key == 'd':
                    for key in self.event_keys:
                        self.event_keys[key].clear()
                    pprint(self.event_keys, width=1)

class ComponentBrowser(GenericBrowser):
    def __init__(self, data, data_transform, labels=None, figure_handle=None, class_names=None, desired_class_names=None, annotation_names=None):
        """
        data is a 2d numpy array with number of signals on dim1, and number of time points on dim2
        data_transform is the transformed data, still a 2d numpy array with number of signals x number of components
            For example, transformed using one of (sklearn.decomposition.PCA, umap.UMAP, sklearn.manifold.TSNE, sklearn.decomposition.FastICA)
        labels are n_signals x 1 array with each entry representing the class of each signal piece. MAKE SURE ALL CLASS LABELS ARE POSITIVE
        class_names is a dictionary, for example {0:'Unclassified', '1:Resonant', '2:NonResonant'}

        This GUI is meant to be used for 
          - 'corrections', where classes are modified / assigned
          - 'annotations', where labels or annotations (separate from a class assignment in the sense that each signal belongs exactly to one class, and a signal my have 0 or more annotations)

        example - 
            import projects.gaitmusic as gm
            mr = gm.MusicRunning01()
            lf = mr(10).ot
            sig_pieces = gm.gait_phase_analysis(lf, muscle_line_name='RSBL_Upper', target_samples=500)
            gui.ComponentBrowser(sig_pieces)

            Single-click on scatter plots to select a gait cycle.
            Press r to refresh 'recent history' plots.
            Double click on the time course plot to select a gait cycle from the time series plot.
        """
        super().__init__(figure_handle)
        self.alpha = {'manual':0.8, 'auto':0.3}
        self.data = data

        n_components = np.shape(data_transform)[1]
        
        if labels is None:
            self.labels = np.zeros(self.n_signals, dtype=int)
        else:
            assert len(labels) == self.n_signals
            self.labels = labels
        assert np.min(self.labels) >= 0 # make sure all class labels are zero or positive!

        class_labels = list(np.unique(self.labels))
        self.class_labels_str = [str(x) for x in class_labels] # for detecting keypresses
        self.n_classes = len(class_labels) 
        if class_names is None:
            self.class_names = {class_label: f'Class_{class_label}' for class_label in class_labels}
        else:
            assert set(class_names.keys()) == set(class_labels)
            self.class_names = class_names
        self.classes = [ClassLabel(label=label, name=self.class_names[label]) for label in self.labels]

        if desired_class_names is None:
            desired_class_names = self.class_names
        self.desired_class_names = desired_class_names

        if annotation_names is None:
            annotation_names = {1:'Representative', 2:'Best', 3:'Noisy', 4:'Worst'}
        self.annotation_names = annotation_names
        self.annotation_idx_str = [str(x) for x in self.annotation_names]

        self.cid.append(self.figure.canvas.mpl_connect('pick_event', self.onpick))
        self.cid.append(self.figure.canvas.mpl_connect('button_press_event', self.select_signal_piece_dblclick))

        n_scatter_plots = int(n_components*(n_components-1)/2)
        self.gs = GridSpec(3, max(n_scatter_plots, 4))

        self._data_index = 0
        self.plot_handles = {}
        self.plot_handles['ax_pca'] = {}
        plot_number = 0
        for xc in range(n_components-1):
            for yc in range(xc+1, n_components):
                this_ax = self.figure.add_subplot(self.gs[1, plot_number])
                this_ax.set_title(str((xc+1, yc+1)))
                self.plot_handles['ax_pca'][plot_number] = this_ax
                self.plot_handles[f'scatter_plot_{xc+1}_{yc+1}'] = this_ax.scatter(data_transform[:, xc], data_transform[:, yc], c=self.colors, alpha=self.alpha['auto'], picker=5)
                self.plot_handles[f'scatter_highlight_{xc+1}_{yc+1}'], = this_ax.plot([], [], 'o', color='darkorange')
                plot_number += 1

        self.plot_handles['signal_plots'] = []
        this_ax = self.figure.add_subplot(self.gs[2, 0])
        self.plot_handles['ax_signal_plots'] = this_ax
        for signal_count in range(self.n_signals):
            self.plot_handles['signal_plots'].append(this_ax.plot(self.data[signal_count, :])[0])

        self.plot_handles['ax_current_signal'] = self.figure.add_subplot(self.gs[2, 1])
        self.plot_handles['current_signal'], = self.plot_handles['ax_current_signal'].plot(list(range(self.n_timepts)), [np.nan]*self.n_timepts)
        self.plot_handles['ax_current_signal'].set_xlim(self.plot_handles['ax_signal_plots'].get_xlim())
        self.plot_handles['ax_current_signal'].set_ylim(self.plot_handles['ax_signal_plots'].get_ylim())

        self.plot_handles['ax_history_signal'] = self.figure.add_subplot(self.gs[2, 2])

        self.plot_handles['ax_signal_full'] = self.figure.add_subplot(self.gs[0, :])
        self.plot_handles['signal_full'] = []
        time_index = np.r_[:self.n_timepts]/self.n_timepts
        for idx, sig in enumerate(self.data):
            this_plot_handle, = self.plot_handles['ax_signal_full'].plot(
                idx + time_index, sig, color=self.colors[idx]
            ) # assumes that there is only one line drawn per signal
            self.plot_handles['signal_full'].append(this_plot_handle)
        self.plot_handles['signal_selected_piece'], = self.plot_handles['ax_signal_full'].plot([], [], color='gray', linewidth=2)
        
        this_ylim = self.plot_handles['ax_signal_full'].get_ylim()
        for x_pos in np.r_[:self.n_signals+1]: # separators between signals
            self.plot_handles['ax_signal_full'].plot([x_pos]*2, this_ylim, 'k', linewidth=0.2)
        self.memoryslots.disable()
        
        self._class_info_text = TextView([], self.figure, pos='bottom left')
        self.update_class_info_text()

        self._mode = 'correction' # ('correction' or 'annotation')
        self._mode_text = TextView([], self.figure, pos='center left')
        self.update_mode_text()
        self.add_key_binding('m', self.toggle_mode)

        self._annotation_text = TextView(['', 'Annotation list:']+[f'{k}:{v}' for k,v in self.annotation_names.items()], self.figure, pos='top left')

        self._message = TextView(['Last action : '], self.figure, pos='bottom right')

        self._desired_class_info_text = TextView([], self.figure, pos='bottom center')
        self.update_desired_class_info_text()

        self.add_key_binding('r', self.clear_axes)
        plt.show(block=False)

    @property
    def n_signals(self):
        return self.data.shape[0]
    
    @property
    def n_timepts(self):
        return self.data.shape[-1]

    @property
    def colors(self):
        return [cl.color for cl in self.classes]

    @property
    def signal(self) -> sampled.Data:
        """Return the 2d Numpy array as a signal."""
        return sampled.Data(self.data.flatten(), sr=self.n_timepts)
    
    def select_signal_piece_dblclick(self, event):
        """Double click a signal piece in the timecourse view to highlight that point."""
        if event.inaxes == self.plot_handles['ax_signal_full'] and event.dblclick: # If the click was inside the time course plot
            if 0 <= int(event.xdata) < self.data.shape[0]:
                self._data_index = int(event.xdata)
                self.update()
    
    def onpick(self, event):
        """Single click a projected point."""
        self.pick_event = event
        self._data_index = np.random.choice(event.ind)
        self.update()
        # this_data = event.artist._offsets[event.ind].data

    def update(self):
        super().update()
        for handle_name, handle in self.plot_handles.items():
            if 'scatter_plot_' in handle_name:
                this_data = np.squeeze(handle._offsets[self._data_index].data)
                self.plot_handles[handle_name.replace('_plot_', '_highlight_')].set_data(this_data[0], this_data[1])
        self.plot_handles['ax_history_signal'].plot(self.data[self._data_index, :])
        self.plot_handles['current_signal'].set_ydata(self.data[self._data_index, :])
        self.plot_handles['signal_selected_piece'].set_data(np.arange(self.n_timepts)/self.n_timepts+self._data_index, self.data[self._data_index, :])
        # self.plot_handles['signal_full'][self._data_index].linewidth = 3
        plt.draw()
    
    def update_class_info_text(self, draw=True):
        self._class_info_text.update(['Class list:'] + [f'{k}:{v}' for k,v in self.class_names.items()])
        if draw:
            plt.draw()
    
    def update_desired_class_info_text(self, draw=True):
        self._desired_class_info_text.update(['Desired class list:'] + [f'{k}:{v}' for k,v in self.desired_class_names.items()])
        if draw:
            plt.draw()
    
    def update_mode_text(self, draw=True):
        self._mode_text.update([f'mode: {self._mode}'])
        if draw:
            plt.draw()
    
    def update_message_text(self, text:str, draw=True):
        self._message.update([text])
        if draw:
            plt.draw()

    def toggle_mode(self, event=None): # add key binding to m for switching mode
        self._mode = {'correction':'annotation', 'annotation':'correction'}[self._mode]
        self.update_mode_text()
    
    def update_colors(self, data_idx=None, draw=True):
        if data_idx is None:
            data_idx = list(range(self.n_signals))
        assert isinstance(data_idx, (list, tuple))
        for this_data_idx in data_idx:
            this_color = self.classes[this_data_idx].color
            self.plot_handles['signal_full'][this_data_idx].set_color(this_color)
            for handle_name, handle in self.plot_handles.items():
                if 'scatter_plot_' in handle_name:
                    fc = handle.get_facecolors()
                    fc[this_data_idx, :3] = this_color
                    fc[this_data_idx, -1] = self.alpha['auto'] if self.classes[this_data_idx].is_auto() else self.alpha['manual']
                    handle.set_facecolors(fc)
        if draw:
            plt.draw()
    
    def update_all(self):
        self.update()
        self.update_class_info_text(draw=False)
        self.update_mode_text(draw=False)
        self.update_message_text('Default message', draw=False)
        self.update_colors(draw=False)
        plt.draw()

    def clear_axes(self, event=None):
        self.plot_handles['ax_history_signal'].clear()
        plt.draw()

    def __call__(self, event):
        super().__call__(event)
        if event.name == 'key_press_event' and event.inaxes == self.plot_handles['ax_signal_full'] and (0 <= int(event.xdata) < self.data.shape[0]):
            this_data_idx = int(event.xdata)
            if self._mode == 'correction':
                if (event.key in self.class_labels_str):
                    new_label = int(event.key)
                    original_label = self.classes[this_data_idx].original_label
                    if new_label == original_label:
                        self.classes[this_data_idx].set_auto()
                    else:
                        self.classes[this_data_idx].set_manual()
                    self.classes[this_data_idx].label = new_label
                    self.update_colors([this_data_idx])
            elif self._mode == 'annotation':
                if event.key in self.annotation_idx_str:
                    this_annotation = self.annotation_names[int(event.key)]
                    if this_annotation not in self.classes[this_data_idx].annotations:
                        self.classes[this_data_idx].annotations.append(this_annotation)
                        self.update_message_text(f'Adding annotation {this_annotation} to signal number {this_data_idx}')
    
    def classlabels_to_dict(self):
        fields_to_save = ('label', 'name', 'assignment_type', 'annotations', 'original_label')
        ret = {}
        for class_idx, class_label in enumerate(self.classes):
            ret[class_idx] = {fld:getattr(class_label, fld)for fld in fields_to_save}
        return ret
    
    def set_classlabels(self, classlabels_dict):
        assert set(classlabels_dict.keys()) == set(range(self.n_signals))
        self.classes = [ClassLabel(**this_label) for this_label in classlabels_dict.values()]


class ClassLabel:
    def __init__(self, 
            label:int,                      # class label (0 - unclassified, 1 - non-resonant, 2 - resonant, etc.)
            name:str = None,                # name of the class
            assignment_type:str = 'auto',   # class label was assigned automatically ('auto') or manually ('manual')
            annotations:list = None,        # for adding annotations to a given class instance
            original_label:int = None,
        ):
        assert label >= 0
        self._label = int(label)
        if original_label is None:
            self.original_label = label
        else:
            self.original_label = int(original_label)
        if name is None:
            name = f'Class_{label}'
        self.name = name
        assert assignment_type in ('auto', 'manual')
        self.assignment_type = assignment_type

        self.palette = plt.get_cmap('tab20')([np.r_[0:1.5:0.05]])[0][:, :3]
        # at the moment, colors are assigned automatically
        self._update_colors()

        if annotations is None:
            self.annotations = []
        else:
            assert isinstance(annotations, list)
            self.annotations = annotations

    @property
    def color(self):
        if self.is_auto():
            return self.color_auto
        return self.color_manual

    @property
    def label(self):
        return self._label
    
    @label.setter
    def label(self, val:int):
        self._label = int(val)
        self._update_colors()

    def _update_colors(self):
        if self._label == 0:
            self.color_auto = self.color_manual = (0.0, 0.0, 0.0) # black
        else:
            self.color_auto = self.palette[(self._label-1)*2+1] # lighter
            self.color_manual = self.palette[(self._label-1)*2]
        
    def is_auto(self):
        return (self.assignment_type == 'auto')
    
    def is_manual(self):
        return (self.assignment_type == 'manual')
    
    def set_auto(self):
        """Meant for undo-ing manual assignment"""
        self.assignment_type = 'auto'

    def set_manual(self):
        self.assignment_type = 'manual'
    
    def add_annotation(self, annot:str):
        self.annotations.append(annot)


### -------- Demonstration/example classes
class ButtonFigureDemo(plt.Figure):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.buttons = Buttons(parent=self)
        self.buttons.add(text='test', type_='Toggle')
        self.buttons.add(text='push button', type_='Push', action_func=self.test_callback)
        plt.show(block=False)
    
    def test_callback(self, event=None):
        print(event)


class SelectorFigureDemo:
    def __init__(self):
        f, ax = plt.subplots(1, 1)
        self.buttons = Buttons(parent=f)
        self.buttons.add(text='Start selection', type_='Push', action_func=self.start)
        self.buttons.add(text='Stop selection', type_='Push', action_func=self.stop)
        self.ax = ax
        self.x = np.random.rand(10)
        self.t = np.r_[:1:0.1]
        self.plot_handles = {}
        self.plot_handles['data'], = ax.plot(self.t, self.x)
        self.plot_handles['selected'], = ax.plot([], [], '.')
        plt.show(block=False)
        self.start()
        self.ind = set()
    
    def get_points(self):
        return np.vstack((self.t, self.x)).T

    def onselect(self, verts):
        """Select if not previously selected; Unselect if previously selected"""
        path = Path(verts)
        selected_ind = set(np.nonzero(path.contains_points(self.get_points()))[0])
        existing_ind = self.ind.intersection(selected_ind)
        new_ind = selected_ind - existing_ind
        self.ind = (self.ind - existing_ind).union(new_ind)
        idx = list(self.ind)
        if idx:
            self.plot_handles['selected'].set_data(self.t[idx], self.x[idx])
        else:
            self.plot_handles['selected'].set_data([], [])
        plt.draw()

    def start(self, event=None):
        self.lasso = LassoSelectorWidget(self.ax, onselect=self.onselect)

    def stop(self, event=None):
        self.lasso.disconnect_events()
