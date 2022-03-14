"""
Simple Graphical User Interface elements for browsing data

Classes:
    DataBrowser - Browse 2D arrays, or an array of sampled.Data elements
    PlotBrowser - Scroll through an array of complex data where a plotting function is defined for each element
"""
import io
from PySide2.QtGui import QClipboard, QImage
import matplotlib as mpl
from matplotlib import pyplot as plt

from pntools import sampled


class DataBrowser:
    """
    Browse 2D arrays, or an array of sampled data elements
    """
    def __init__(self, plot_data, titlefunc=None) -> None:
        """
        plot_data is a list of 1D arrays or sampled data
        """
        self._current_idx = 0
        self._keys_used = ['left', 'right']
        self._bindings_removed = {}
        for key in self._keys_used:
            this_param_name = [k for k, v in mpl.rcParams.items() if isinstance(v, list) and key in v]
            if this_param_name: # not an empty list
                assert len(this_param_name) == 1
                this_param_name = this_param_name[0]
                mpl.rcParams[this_param_name].remove(key)
                self._bindings_removed[this_param_name] = key
        
        self._fig, self._ax = plt.subplots()
        this_data = plot_data[0]
        if isinstance(this_data, sampled.Data):
            self._plot = self._ax.plot(this_data.t, this_data())[0]
        else:
            self._plot = self._ax.plot(this_data)[0]
        self.cid = self._fig.canvas.mpl_connect('key_press_event', self)
        self.closeid = self._fig.canvas.mpl_connect('close_event', self)
        self.n_plots = len(plot_data)
        self.data = plot_data
        if titlefunc is None:
            if hasattr(self.data[0], 'name'):
                self.titlefunc=lambda s: f'{s.data[s._current_idx].name}'
            else:
                self.titlefunc = lambda s: f'Plot number {s._current_idx}'
        else:
            self.titlefunc = titlefunc
        plt.show(block=False)
        self.update_plot()
    
    def __call__(self, event):
        # print(event.__dict__) # for debugging
        if event.name == 'key_press_event':
            update_flag = True
            if event.key == 'right':
                self._current_idx = min(self._current_idx+1, self.n_plots-1)
            elif event.key == 'left':
                self._current_idx = max(self._current_idx-1, 0)
            else:
                update_flag = False

            if update_flag:
                self.update_plot()

        elif event.name == 'close_event':
            self._fig.canvas.mpl_disconnect(self.cid)
            self._fig.canvas.mpl_disconnect(self.closeid)

            # restore default bindings
            for param_name, key in self._bindings_removed.items():
                if key not in mpl.rcParams[param_name]:
                    mpl.rcParams[param_name].append(key) # param names: keymap.back, keymap.forward)

    def update_plot(self):
        this_data = self.data[self._current_idx]
        if isinstance(this_data, sampled.Data):
            self._plot.set_data(this_data.t, this_data())
        else:
            self._plot.set_ydata()
        self._ax.set_title(self.titlefunc(self))
        plt.draw()


class PlotBrowser:
    """
    Takes a list of data, and a plotting function that parses each of the elements in the array.
    Assumes that the plotting function is going to make one figure.
    """
    def __init__(self, plot_data, plot_func, **plot_kwargs):
        """
            plot_data - list of data objects to browse
            plot_func - plotting function that accepts:
                an element in the plot_data list as its first input
                figure handle in which to plot as the second input
                keyword arguments to be passed to the plot_func
            plot_kwargs - these keyword arguments will be passed to plot_function after data and figure
        """
        self.plot_data = plot_data # list where each element serves as input to plot_func
        self.plot_func = plot_func
        self.plot_kwargs = plot_kwargs

        # tracking variable
        self._current_idx = 0

        # setup
        self._fig = plt.figure()

        # for cleanup
        self.cid = []
        self.cid.append(self._fig.canvas.mpl_connect('key_press_event', self))
        self.cid.append(self._fig.canvas.mpl_connect('close_event', self))

        # initialize
        self._keypressdict = {}
        self._bindings_removed = {} 
        self.add_key_binding('left', self.decrement)
        self.add_key_binding('right', self.increment)
        self.add_key_binding('ctrl+c', self.copy_to_clipboard)
        plt.show(block=False)
        self.update_figure()

    def mpl_remove_bindings(self, key_list):
        """If the existing key is bound to something in matplotlib, then remove it"""
        for key in key_list:
            this_param_name = [k for k, v in mpl.rcParams.items() if isinstance(v, list) and key in v]
            if this_param_name: # not an empty list
                assert len(this_param_name) == 1
                this_param_name = this_param_name[0]
                mpl.rcParams[this_param_name].remove(key)
                self._bindings_removed[this_param_name] = key
    
    def mpl_restore_bindings(self):
        """Restore any modified default keybindings in matplotlib"""
        for param_name, key in self._bindings_removed.items():
            if key not in mpl.rcParams[param_name]:
                mpl.rcParams[param_name].append(key) # param names: keymap.back, keymap.forward)
        self._bindings_removed[param_name] = {}

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
    
    def add_key_binding(self, key_name, on_press_function):
        """
        This is useful to add key-bindings in classes that inherit from this one, or on the command line.
        See usage in the __init__ function
        """
        self.mpl_remove_bindings([key_name])
        self._keypressdict[key_name] = on_press_function
        
    def get_current_data(self):
        return self.plot_data[self._current_idx]

    def increment(self):
        self._current_idx = min(self._current_idx+1, len(self.plot_data)-1)
        self.update_figure()

    def decrement(self):
        self._current_idx = max(self._current_idx-1, 0)
        self.update_figure()
    
    def copy_to_clipboard(self):
        buf = io.BytesIO()
        self._fig.savefig(buf)
        QClipboard().setImage(QImage.fromData(buf.getvalue()))
        buf.close()

    def save_figure(self):
        self._fig.savefig()

    def update_figure(self):
        self._fig.clear()
        self.plot_func(self.get_current_data(), self._fig, **self.plot_kwargs)
        plt.draw()
    