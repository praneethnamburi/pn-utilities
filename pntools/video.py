"""
Tools for working with videos
"""
import os
import re
import subprocess
import urllib
import time
from pathlib import Path

import ffmpeg
from pytube import YouTube

CLIP_FOLDER = 'C:\\data\\_clipcollection'


def detect_black_frames(vid_file):
    """Detect black frames in a video file using ffmpeg.

    Returns a dictionary with lists of start times, end times, duration, and the file name that was given as input
    """
    assert os.path.exists(vid_file)
    def find_float(str, pre):
        return float(re.search(f'(?<={pre})[.\\d]+', str).group(0))
    bd = subprocess.getoutput(f'ffmpeg -i "{vid_file}" -vf blackdetect=d=0:pix_th=.01 -f rawvideo -y /NUL')
    blackframe_segments = [b for b in bd.splitlines() if 'blackdetect' in b]
    start = [find_float(seg, 'black_start:') for seg in blackframe_segments]
    end = [find_float(seg, 'black_end:') for seg in blackframe_segments]
    duration = [find_float(seg, 'black_duration:') for seg in blackframe_segments]
    return {'start': start, 'end': end, 'duration': duration, 'fname': vid_file}

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
