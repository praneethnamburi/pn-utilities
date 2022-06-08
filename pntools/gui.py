"""
Simple Graphical User Interface elements for browsing data

Classes:
    GenericBrowser - Generic class to browse data. Meant to be extended.
    SignalBrowser - Browse an array of pntools.Sampled.Data elements, or 2D arrays
    PlotBrowser - Scroll through an array of complex data where a plotting function is defined for each element
    VideoBrowser - Scroll through images of a video

    Future:
        Extend VideoBrowser to play, pause and extract clips using hotkeys. Show timeline in VideoBrowser.
        Add clickable navigation.
"""
import io
import os
from datetime import timedelta

import ffmpeg
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import axes as maxes
from matplotlib import lines as mlines
from matplotlib.widgets import Button as ButtonWidget
from matplotlib.widgets import LassoSelector as LassoSelectorWidget
from matplotlib.path import Path
from decord import VideoReader

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
        selected_ind = Path(verts).contains_points(self.get_data())
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
            figure_handle = list(self.plot_handles.values())[0].figure # figure_handle passed as input will be ignored
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
        s0 = self.selectors.add(list(self.plot_handles.values())[0])
        self.buttons.add(text='Selector 0', type_='Toggle', action_func=s0.toggle, start_state=s0.is_active)

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
