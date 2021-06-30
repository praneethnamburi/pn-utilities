"""
Tools for working with videos
"""
import os
import subprocess
import urllib
import time
from datetime import timedelta

import matplotlib as mpl
from matplotlib import pyplot as plt
from decord import VideoReader
import ffmpeg
from pytube import YouTube

CLIP_FOLDER = 'C:\\data\\_clipcollection'

class View(VideoReader):
    """
    View videos [in the future, annotate them and extract clips]
    
    Navigation (arrow keys):
        right       - forward one frame
        left        - back one frame
        up          - forward 10 frames
        down        - back 10 frames
        shift+left  - first frame
        shift+right - last frame
        shift+up    - forward nframes/20 frames
        shift+down  - back nframes/20 frames

    Player mode:
        Play and pause with audio
        
    Annotations:
        Text annotation at frames [event markings, observations]
        Drawing shapes
    """
    def __init__(self, vid_name):
        if not os.path.exists(vid_name):
            # try looking in the CLIP FOLDER
            vid_name = os.path.join(CLIP_FOLDER, os.path.split(vid_name)[-1])
        
        assert os.path.exists(vid_name)
        self.vid_name = vid_name
        with open(vid_name, 'rb') as f:
            super().__init__(f)
        self._current_frame = 0

        # remove default key bindings for the ones that we will use
        self._keys_used = ['left', 'right', 'up', 'down', 'shift+left', 'shift+right', 'shift+up', 'shift+down']
        self._bindings_removed = {}
        for key in self._keys_used:
            this_param_name = [k for k, v in mpl.rcParams.items() if isinstance(v, list) and key in v]
            if this_param_name: # not an empty list
                assert len(this_param_name) == 1
                this_param_name = this_param_name[0]
                mpl.rcParams[this_param_name].remove(key)
                self._bindings_removed[this_param_name] = key

        self._fig, self._ax = plt.subplots()
        self.im = self._ax.imshow(self[self._current_frame].asnumpy())
        self.cid = self._fig.canvas.mpl_connect('key_press_event', self)
        self.closeid = self._fig.canvas.mpl_connect('close_event', self)
        self.nframes = len(self)
        self.fps = self.get_avg_fps()
        plt.axis('off')
        self.update_image()
        # print(self.__dict__)
        plt.show(block=False)
    
    def __call__(self, event):
        # print(event.__dict__) # for debugging
        if event.name == 'key_press_event':
            update_flag = True
            if event.key == 'right':
                self._current_frame = min(self._current_frame+1, self.nframes-1)
            elif event.key == 'left':
                self._current_frame = max(self._current_frame-1, 0)
            elif event.key == 'up':
                self._current_frame = min(self._current_frame+10, self.nframes-1)
            elif event.key == 'down':
                self._current_frame = max(self._current_frame-10, 0)
            elif event.key == 'shift+right':
                self._current_frame = self.nframes-1
            elif event.key == 'shift+left':
                self._current_frame = 0
            elif event.key == 'shift+up':
                self._current_frame = min(self._current_frame+int(self.nframes/20), self.nframes-1)
            elif event.key == 'shift+down':
                self._current_frame = max(self._current_frame-int(self.nframes/20), 0)
            else:
                update_flag = False

            if update_flag:
                self.update_image()

        elif event.name == 'close_event':
            self._fig.canvas.mpl_disconnect(self.cid)
            self._fig.canvas.mpl_disconnect(self.closeid)

            # restore default bindings
            for param_name, key in self._bindings_removed.items():
                if key not in mpl.rcParams[param_name]:
                    mpl.rcParams[param_name].append(key) # param names: keymap.back, keymap.forward)

    def update_image(self):
        self.im.set_data(self[self._current_frame].asnumpy())
        
        self._ax.set_title('Frame {:d}/{:d}, {:f} fps, '.format(self._current_frame, self.nframes, self.fps) + str(timedelta(seconds=self._current_frame/self.fps)))
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
    

def download(url, start_time=None, end_time=None, dur=None, full_file=False):
    """
    Download a clip from a YouTube video.
    Downloads to the path specified in the module variable CLIP_FOLDER
    Returns the name of the downloaded video.
    """
    # url = 'https://www.youtube.com/watch?v=5umbf4ps0GQ'
    
    # download video
    for _ in range(3):
        try:
            yvid = YouTube(url)
            vid_found = True
            break
        except urllib.error.HTTPError:
            time.sleep(0.5)
            vid_found = False
    
    assert vid_found is True

    ys = yvid.streams.get_highest_resolution()
    fname_in = ys.download(CLIP_FOLDER, url.split('?v=')[-1])
    if full_file: # otherwise clip the video
        return fname_in

    # specify interval
    if start_time is None: # manually find the start and stop times
        this_vid = View(fname_in)
        start_time = float(input('Specify start frame: '))/this_vid.fps
        end_time = float(input('Specify end frame: ')+1)/this_vid.fps
        dur = end_time - start_time
    else:
        assert isinstance(start_time, (float, int))
        assert not (end_time is None and dur is None)
        if dur is None:
            end_time = start_time + dur
        else:
            dur = end_time - start_time

    # clip video
    fname_out = os.path.join(CLIP_FOLDER, os.path.splitext(fname_in)[0] + '_s{:.3f}_e{:.3f}.mp4'.format(start_time, end_time))
    ffmpeg.input(fname_in, ss=start_time).output(fname_out, vcodec='h264_nvenc', t=dur).run()
    
    return fname_out


def demo():
    vid_name = 'C:\\data\\_clipcollection\\59eH83HMI8E_s41.174_e42.642.mp4'
    return View(vid_name)
