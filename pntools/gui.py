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
import io
import json
import os
from datetime import timedelta, datetime
from pathlib import Path

import ffmpeg
import numpy as np
from hdbscan import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import axes as maxes
from matplotlib import lines as mlines
from matplotlib.widgets import Button as ButtonWidget
from matplotlib.widgets import LassoSelector as LassoSelectorWidget
from matplotlib.path import Path as mPath
from matplotlib.gridspec import GridSpec

from pntools import sampled



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

        # tracking variable memory slots
        self._idx_memory_slots = self.initialize_memory_slots()
        self._memtext = None
        self._keybindingtext = None
        self.buttons = Buttons(parent=self)
        self.selectors = Selectors(parent=self)

        # for cleanup
        self.cid = []
        self.cid.append(self.figure.canvas.mpl_connect('key_press_event', self))
        self.cid.append(self.figure.canvas.mpl_connect('close_event', self))
    
    def update(self): # extended classes are expected to implement their update function!
        self.update_memory_slot_display()
    
    def update_without_clear(self):
        self.update_memory_slot_display()
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
                self._keypressdict[event.key][0]() # this may or may not redraw everything
            if event.key in self._idx_memory_slots:
                self.memory_slot_update(event.key)
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
    
    def memory_slot_update(self, key):
        """
        memory slot handling - Initiate when None, Go to the slot if it exists, frees slot if pressed when it exists
        key is the event.key triggered by a callback
        """
        if self._idx_memory_slots[key] is None:
            self._idx_memory_slots[key] = self._current_idx
            self.update_memory_slot_display()
        elif self._idx_memory_slots[key] == self._current_idx:
            self._idx_memory_slots[key] = None
            self.update_memory_slot_display()
        else:
            self._current_idx = self._idx_memory_slots[key]
            self.update()

    @staticmethod
    def initialize_memory_slots():
        return {str(k):None for k in range(1, 10)}

    def disable_memory_slots(self):
        self._idx_memory_slots = {}
    
    def enable_memory_slots(self):
        self._idx_memory_slots = self.initialize_memory_slots()

    def show_memory_slots(self, pos='bottom left'):
        self._memtext = TextView(self._idx_memory_slots, fax=self.figure, pos=pos)
    
    def update_memory_slot_display(self):
        """Refresh memory slot text if it is not hidden"""
        if self._memtext is not None:
            self._memtext.update(self._idx_memory_slots)

    def hide_memory_slots(self):
        """Hide the memory slot text"""
        if self._memtext is not None:
            self._memtext._text.remove()
        self._memtext = None

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
        self.update_without_clear() # panning is pointless if update clears the axis!!


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
        self.show_memory_slots()
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
            self.show_memory_slots()
            self.plot_func(self.get_current_data(), self.figure, **self.plot_kwargs)
        else:
            self.update_memory_slot_display()
            self.plot_func(self.get_current_data(), self.plot_handles, **self.plot_kwargs)
        if self.buttons['Auto limits'].state: # is True
            self.reset_axes()
        plt.draw()
    
    def udpate_without_clear(self):
        self.update_memory_slot_display()
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
            self._plot = self._ax.plot(this_data.t, this_data())[0]
        else:
            self._plot = self._ax.plot(this_data)[0]

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
            self._plot.set_data(this_data.t, this_data())
        else:
            self._plot.set_ydata()
        self._ax.set_title(self.titlefunc(self))
        if self.buttons['Auto limits'].state: # is True
            self.reset_axes()
        plt.draw()


class VideoBrowser(GenericBrowser):
    """Scroll through images of a video"""
    def __init__(self, vid_name, titlefunc=None, figure_handle=None):
        from decord import VideoReader
        super().__init__(figure_handle)

        if not os.path.exists(vid_name): # try looking in the CLIP FOLDER
            vid_name = os.path.join(CLIP_FOLDER, os.path.split(vid_name)[-1])        
        assert os.path.exists(vid_name)
        self.fname = vid_name
        self.name = os.path.splitext(os.path.split(vid_name)[1])[0]
        with open(vid_name, 'rb') as f:
            self.data = VideoReader(f)
        
        self._ax = self.figure.subplots(1, 1)
        this_data = self.data[0]
        self._im = self._ax.imshow(this_data.asnumpy())

        self.fps = self.data.get_avg_fps()
        plt.axis('off')
        if titlefunc is None:
            self.titlefunc = lambda s: f'Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}'
        
        self.set_default_keybindings()
        self.add_key_binding('e', self.extract_clip)
        self.show_memory_slots(pos='bottom left')

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
        if start_frame is None:
            start_frame = self._idx_memory_slots['1']
        if end_frame is None:
            end_frame = self._idx_memory_slots['2']
        assert end_frame > start_frame
        start_time = float(start_frame)/self.fps
        end_time = float(end_frame)/self.fps
        dur = end_time - start_time
        if out_rate is None:
            out_rate = self.fps
        if fname_out is None:
            fname_out = os.path.join(CLIP_FOLDER, os.path.splitext(self.name)[0] + '_s{:.3f}_e{:.3f}.mp4'.format(start_time, end_time))
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
        self.show_memory_slots(pos='bottom left')

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
        import subprocess
        import shutil
        if start_frame is None:
            start_frame = self._idx_memory_slots['1']
        if end_frame is None:
            end_frame = self._idx_memory_slots['2']
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
    """
    def __init__(self, vid_name, titlefunc=None, figure_handle=None):
        super().__init__(vid_name, titlefunc, figure_handle)
        self.hide_memory_slots()
        self.disable_memory_slots()

        self.fname_annotations = os.path.join(Path(self.fname).parent, Path(self.fname).stem + '_annotations.json')
        self.annotations = self.load_annotations()
        self.add_key_binding('s', on_press_function=self.save_annotations)

        self._current_label = '0'
        self._current_label_text = TextView([f'Current label: {self._current_label}'], self.figure, pos='bottom center')
        self.mpl_remove_bindings(['/', '.'])

        self.plot_handles = {}
        self.palette = sns.color_palette('Set2', max([int(x) for x in self.annotations.keys()])+1)
        for label, color in zip(self.annotations, self.palette):
            self.plot_handles[f'label_{label}'], = self._ax.plot([], [], 'o', color=color)

        plt.show(block=False)
        self.update()
    
    def __call__(self, event):
        super().__call__(event)
        if event.name == 'key_press_event':
            current_label = self._current_label
            frame_number = str(self._current_idx)
            if event.key == '/': # Add annotation at frame. If it exists, it'll get overwritten.
                self.annotations[current_label][frame_number] = float(event.xdata), float(event.ydata)
                self.update()
            elif event.key == '.': # remove annotation at the current frame if it exists
                self.annotations[current_label].pop(frame_number, None)
                self.update()
            elif event.key in self.annotations:
                self._current_label = event.key
                self.update_current_label_text(draw=True)
    
    def update(self):
        self.update_annotation_display(draw=False)
        self.update_current_label_text(draw=False)
        super().update()
        self.reset_axes()
    
    def update_annotation_display(self, draw=False):
        for label, annot_dict in self.annotations.items():
            this_data = ([], [])
            frame_number = str(self._current_idx)
            if frame_number in annot_dict: 
                this_data = annot_dict[frame_number]
            self.plot_handles[f'label_{label}'].set_data(*this_data)
        if draw:
            plt.draw()
    
    def update_current_label_text(self, draw=False):
        self._current_label_text.update([f'Current label: {self._current_label}'])
        if draw:
            plt.draw()

    def load_annotations(self, event=None):
        if os.path.exists(self.fname_annotations):
            with open(self.fname_annotations, 'r') as f:
                return json.load(f)
        return {str(label):{} for label in range(10)}
        
    def save_annotations(self, event=None):
        with open(self.fname_annotations, 'w') as f:
            json.dump(self.annotations, f, indent=4)

class TextView:
    """Show text array line by line"""
    def __init__(self, text, fax=None, pos='bottom left'):
        """
        text is an array of strings
        fax is either a figure or an axis handle
        """
        def rescale(xy, margin=0.01):
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
    def __init__(self, data, data_transform, labels, figure_handle=None):
        """
        data is a 2d numpy array with number of signals on dim1, and number of time points on dim2
        algorithm (class) - (sklearn.decomposition.PCA, umap.UMAP, sklearn.manifold.TSNE, sklearn.decomposition.FastICA)
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

        n_components = np.shape(data_transform)[1]

        palette = sns.color_palette('Set2', np.unique(labels).max() + 1)
        colors = [palette[x] if x >= 0 else (0.0, 0.0, 0.0) for x in labels]

        self.cid.append(self.figure.canvas.mpl_connect('pick_event', self.onpick))
        self.cid.append(self.figure.canvas.mpl_connect('button_press_event', self.select_gaitcycle_dblclick))

        self.data = data
        self.signal = sampled.Data(self.data.flatten(), sr=self.n_timepts)
        self.labels = labels

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
                self.plot_handles[f'scatter_plot_{xc+1}_{yc+1}'] = this_ax.scatter(data_transform[:, xc], data_transform[:, yc], c=colors, picker=5)
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
        self.plot_handles['signal_full'] = \
            [self.plot_handles['ax_signal_full'].plot(self.signal.t[i: i + self.n_timepts], self.signal()[i: i + self.n_timepts], color=colors[i // self.n_timepts]) for i in range(0, len(self.signal()) - self.n_timepts + 1, self.n_timepts)]
        self.plot_handles['signal_current_piece'], = self.plot_handles['ax_signal_full'].plot([], [], color='gray', linewidth=2)
        
        this_ylim = self.plot_handles['ax_signal_full'].get_ylim()
        for x_pos in np.r_[:self.n_signals+1]:
            self.plot_handles['ax_signal_full'].plot([x_pos]*2, this_ylim, 'k', linewidth=0.2)
        self.disable_memory_slots()
        self.add_key_binding('r', self.clear_axes)
        plt.show(block=False)

    @property
    def n_signals(self):
        return self.data.shape[0]
    
    @property
    def n_timepts(self):
        return self.data.shape[-1]
    
    def select_gaitcycle_dblclick(self, event):
        if event.inaxes == self.plot_handles['ax_signal_full'] and event.dblclick: # If the click was inside the time course plot
            if 0 <= int(event.xdata) < self.data.shape[0]:
                self._data_index = int(event.xdata)
                self.update()
    
    def onpick(self, event):
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
        self.plot_handles['signal_current_piece'].set_data(np.arange(self.n_timepts)/self.n_timepts+self._data_index, self.data[self._data_index, :])
        # self.plot_handles['signal_plots'][self._data_index].linewidth = 3
        plt.draw()
    
    def clear_axes(self, event=None):
        self.plot_handles['ax_history_signal'].clear()
        plt.draw()


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
