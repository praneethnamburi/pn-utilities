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

class ParseAx:
    def __init__(self, ax=None) -> None:
        if ax is None:
            self._fig, self._ax = plt.subplots()
        else:
            import matplotlib.axes as maxes
            assert isinstance(ax, (maxes.Axes, plt.Figure))
            if isinstance(ax, plt.Figure):
                self._fig = ax
                if not self._fig.axes:
                    self._ax = self._fig.subplots(1, 1)
                else:
                    # find an empty axis
                    for this_ax in self._fig.axes:
                        # if axis does not have images, lines, or scatter plots
                        if not any((bool(this_ax.get_images()), bool(this_ax.lines), bool(this_ax.collections))):
                            break
                    self._ax = this_ax
    

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
