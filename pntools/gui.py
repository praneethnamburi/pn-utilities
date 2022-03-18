"""
Simple Graphical User Interface elements for browsing data

Classes:
    DataBrowser - Browse 2D arrays, or an array of sampled.Data elements
    PlotBrowser - Scroll through an array of complex data where a plotting function is defined for each element
"""
import io
import os
from datetime import timedelta

import ffmpeg
import matplotlib as mpl
from PySide2.QtGui import QClipboard, QImage
from matplotlib import pyplot as plt
from decord import VideoReader

from pntools import sampled


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
        self._keypressdict = {}
        self._bindings_removed = {}
        
        # tracking variable
        self._current_idx = 0

        # for cleanup
        self.cid = []
        self.cid.append(self._fig.canvas.mpl_connect('key_press_event', self))
        self.cid.append(self._fig.canvas.mpl_connect('close_event', self))
    
    def update(self):
        pass # inherited classes are expected to implement an update function!

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
        if event.name == 'key_press_event' and event.key in self._keypressdict:
            self._keypressdict[event.key]()

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

    # Event responses - useful to pair with add_key_binding
    # These capabilities can be assigned to different key bindings
    def add_key_binding(self, key_name, on_press_function):
        """
        This is useful to add key-bindings in classes that inherit from this one, or on the command line.
        See usage in the __init__ function
        """
        self.mpl_remove_bindings([key_name])
        self._keypressdict[key_name] = on_press_function

    def set_default_keybindings(self):
        self.add_key_binding('left', self.decrement)
        self.add_key_binding('right', self.increment)
        self.add_key_binding('up', (lambda s: s.increment(step=10)).__get__(self))
        self.add_key_binding('down', (lambda s: s.decrement(step=10)).__get__(self))
        self.add_key_binding('shift+left', self.increment_frac)
        self.add_key_binding('shift+right', self.decrement_frac)
        self.add_key_binding('shift+up', self.go_to_start)
        self.add_key_binding('shift+down', self.go_to_end)
        self.add_key_binding('ctrl+c', self.copy_to_clipboard)
    
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
        self.plot_func(self.get_current_data(), self._fig, **self.plot_kwargs)
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

CLIP_FOLDER = 'C:\\data\\_clipcollection'

class VideoBrowser(GenericBrowser):
    def __init__(self, vid_name, titlefunc=None, figure_handle=None):
        super().__init__(figure_handle)

        if not os.path.exists(vid_name): # try looking in the CLIP FOLDER
            vid_name = os.path.join(CLIP_FOLDER, os.path.split(vid_name)[-1])
        
        assert os.path.exists(vid_name)
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
        plt.show(block=False)
        self.update()

    def update(self):
        self._im.set_data(self.data[self._current_idx].asnumpy())
        self._ax.set_title(self.titlefunc(self))
        plt.draw()

    def extract_clip(self, start_frame, end_frame, fname_out=None, out_rate=None):
        start_time = float(start_frame)/self.fps
        end_time = float(end_frame)/self.fps
        dur = end_time - start_time
        if out_rate is None:
            out_rate = self.fps
        if fname_out is None:
            fname_out = os.path.join(CLIP_FOLDER, os.path.splitext(self.vid_name)[0] + '_s{:.3f}_e{:.3f}.mp4'.format(start_time, end_time))
        ffmpeg.input(self.vid_name, ss=start_time).output(fname_out, vcodec='h264_nvenc', t=dur, r=out_rate).run()
        return fname_out
