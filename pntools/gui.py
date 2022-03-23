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
import matplotlib as mpl
from PySide2.QtGui import QClipboard, QImage
from matplotlib import pyplot as plt
from matplotlib import axes as maxes
from decord import VideoReader

from pntools import sampled

CLIP_FOLDER = 'C:\\data\\_clipcollection'


class GenericBrowser:
    """
    Generic class to browse data. Meant to be extended before use.

    Default Navigation (arrow keys):
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
        self._fig = figure_handle
        self._keypressdict = {} # managed by add_key_binding
        self._bindings_removed = {}
        
        # tracking variable
        self._current_idx = 0

        # tracking variable memory slots
        self._idx_memory_slots = {str(k):None for k in range(1, 10)}
        self._memtext = None
        self._keybindingtext = None

        # for cleanup
        self.cid = []
        self.cid.append(self._fig.canvas.mpl_connect('key_press_event', self))
        self.cid.append(self._fig.canvas.mpl_connect('close_event', self))
    
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
            self._fig.canvas.mpl_disconnect(this_cid)
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
        buf = io.BytesIO()
        self._fig.savefig(buf)
        QClipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()
    
    def show_memory_slots(self, pos='bottom left'):
        self._memtext = TextView(self._idx_memory_slots, fax=self._fig, pos=pos)
    
    def update_memory_slot_display(self):
        """Refresh memory slot text if it is not hidden"""
        if self._memtext is not None:
            self._memtext.update(self._idx_memory_slots)

    def hide_memory_slots(self):
        """Hide the memory slot text"""
        if self._memtext is not None:
            self._memtext._text.remove()
        self._memtext = None

    def show_key_bindings(self, f=None, pos='bottom right'):
        f = {None: self._fig, 'new': plt.figure()}[f]
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
        ax = self._filter_sibling_axes(self._fig.axes, share=pan_ax, get_bool=False)
        if ax is None:
            return
        for this_ax in ax:
            lim1, lim2 = {'x': this_ax.get_xlim, 'y': this_ax.get_ylim}[pan_ax]()
            inc = (lim2-lim1)*frac
            if direction in ('up', 'right'):
                new_lim = (lim1+inc, lim2+inc)
            else:
                new_lim = (lim1-inc, lim2-inc)
            {'x': this_ax.set_xlim, 'y': this_ax.set_ylim}[pan_ax](new_lim)
        self.update_without_clear() # panning is pointless if update clears the axis!!


class PlotBrowser(GenericBrowser):
    """
    Takes a list of data, and a plotting function that parses each of the elements in the array.
    Assumes that the plotting function is going to make one figure.
    """
    def __init__(self, plot_data, plot_func, figure_handle=None, **plot_kwargs):
        """
            plot_data - list of data objects to browse
            plot_func - plotting function that accepts:
                an element in the plot_data list as its first input
                figure handle in which to plot as the second input
                keyword arguments to be passed to the plot_func
            plot_kwargs - these keyword arguments will be passed to plot_function after data and figure
        """
        # setup
        super().__init__(figure_handle)

        self.data = plot_data # list where each element serves as input to plot_func
        self.plot_func = plot_func
        self.plot_kwargs = plot_kwargs

        # initialize
        self.set_default_keybindings()
        plt.show(block=False)
        self.update()
        
    def get_current_data(self):
        return self.data[self._current_idx]

    def update(self):
        self._fig.clear()
        self.show_memory_slots()
        self.plot_func(self.get_current_data(), self._fig, **self.plot_kwargs)
        plt.draw()
    
    def udpate_without_clear(self):
        self.update_memory_slot_display()
        plt.draw()


class SignalBrowser(GenericBrowser):
    """
    Browse an array of pntools.Sampled.Data elements, or 2D arrays
    """
    def __init__(self, plot_data, titlefunc=None, figure_handle=None):
        super().__init__(figure_handle)

        self._ax = self._fig.subplots(1, 1)
        this_data = plot_data[0]
        if isinstance(this_data, sampled.Data):
            self._plot = self._ax.plot(this_data.t, this_data())[0]
        else:
            self._plot = self._ax.plot(this_data)[0]

        self.data = plot_data
        if titlefunc is None:
            if hasattr(self.data[0], 'name'):
                self.titlefunc=lambda s: f'{s.plot_data[s._current_idx].name}'
            else:
                self.titlefunc = lambda s: f'Plot number {s._current_idx}'
        else:
            self.titlefunc = titlefunc

        # initialize
        self.set_default_keybindings()
        plt.show(block=False)
        self.update()
    
    def update(self):
        this_data = self.data[self._current_idx]
        if isinstance(this_data, sampled.Data):
            self._plot.set_data(this_data.t, this_data())
        else:
            self._plot.set_ydata()
        self._ax.set_title(self.titlefunc(self))
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
        
        self._ax = self._fig.subplots(1, 1)
        this_data = self.data[0]
        self._im = self._ax.imshow(this_data.asnumpy())

        self.fps = self.data.get_avg_fps()
        plt.axis('off')
        if titlefunc is None:
            self.titlefunc = lambda s: f'Frame {s._current_idx}/{len(s)}, {s.fps} fps, {str(timedelta(seconds=s._current_idx/s.fps))}'
        
        self.set_default_keybindings()
        self.add_key_binding('e', self.extract_clip)
        self.show_memory_slots()
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
        self.text = self.parse_text(text)
        self._text = None # matplotlib text object
        self._fig, self._ax = self.parse_fax(fax)
        self._pos = self.parse_pos(pos)
        self.setup()
        self.update()
    
    def parse_text(self, text):
        if isinstance(text, dict):
            text = [f'{key} - {val}' for key, val in text.items()] 
        return text

    def parse_fax(self, fax):
        assert isinstance(fax, (type(None), plt.Figure, maxes.Axes))
        if fax is None:
            f = plt.figure()
            ax = f.add_axes((0.01, 0.01, 0.02, 0.98))
        elif isinstance(fax, plt.Figure):
            f = fax
            ax = f.add_axes((0.01, 0.01, 0.02, 0.98))
        else:
            f = fax.figure
            ax = fax
        return f, ax
    
    def parse_pos(self, pos):
        if isinstance(pos, str):
            updown, leftright = pos.replace('middle', 'center').split(' ')
            assert updown in ('top', 'center', 'bottom')
            assert leftright in ('left', 'center', 'right')
            y = {'top': 1, 'center': 0.5, 'bottom': 0}[updown]
            x = {'left': 0, 'center': 0.5, 'right': 1}[leftright]
            pos = (x, y, updown, leftright)
        assert len(pos) == 4
        return pos

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
