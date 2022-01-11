import matplotlib as mpl
from matplotlib import pyplot as plt

from pntools import sampled


class PlotBrowser:
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
            self.titlefunc = lambda s: f'Plot number {s._current_idx}'
        else:
            self.titlefunc = titlefunc
        plt.show(block=False)
    
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
