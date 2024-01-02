"""
Tools for working with videos. It contains some wrappers that call ffmpeg for processing videos.

Use the gui module for browsing videos.
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


def probe(vid_file):
    """return the output of ffprobe"""
    assert os.path.exists(vid_file)
    return subprocess.getoutput(f'ffprobe -hide_banner -show_entries stream=duration "{vid_file}"')

def get_sr(vid_file):
    """Parse ffprobe output to get sampling rate of a video"""
    x = probe(vid_file)
    try:
        sr = round(float(re.search('fps, (.+?) tbr,', x).group(1)))
    except AttributeError:
        sr = None
    return sr

def get_dur(vid_file):
    """Parse ffprobe output to get the duration of a video file"""
    return float(probe(vid_file).split('[STREAM]')[-1].split('[/STREAM]')[0].split('duration=')[-1].rstrip('\n'))

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

def interp_black_frames(vid_file, vid_output=None, overwrite=False):
    """Interpolate black frames in a video"""
    if vid_output is None:
        vid_output = os.path.join(Path(vid_file).parent, f'{Path(vid_file).stem} bfinterp{Path(vid_file).suffix}')
    if (not os.path.exists(vid_output)) or overwrite:
        vid_sr = get_sr(vid_file)
        this_cmd = f'ffmpeg -y -i "{vid_file}" -vf blackframe=0,metadata=select:key=lavfi.blackframe.pblack:value=50:function=less,framerate=fps={vid_sr} -c:v h264_nvenc "{vid_output}"'
        return subprocess.getoutput(this_cmd)
    return "Did not interpolate."

def make_montage2x2(vid_files, vid_output=None, aud_file=None, overwrite=False):
    """
    Create a 2x2 montage with 4 video files using ffmpeg xstack
    
    Inputs:
        vid_files - list/tuple of 4 video file names
        vid_output (optional) - name of the output file
        aud_file (optional) - full path to the audio file

    Returns:
        Output from ffmpeg
    """
    assert len(vid_files) == 4
    vid_inputs = '" -i "'.join([''] + vid_files).removeprefix('" ') + '"'
    if vid_output is None:
        v0 = Path(vid_files[0])
        pre = str(v0.stem)
        vid_output = f'{os.path.join(v0.parent, pre)}-montage.mp4'
    aud_input_str = '' 
    aud_input = '0' # get audio from the first file if there is no audio
    aud_codec_str = ''
    if aud_file is not None:
        assert os.path.exists(aud_file)
        aud_input_str = f' -i "{aud_file}"'
        aud_input = '4'
        aud_codec_str = ' -c:a aac'
        aud_map = f"-map {aud_input}:a "
    else: # no audio file, check if the first video has audio in it
        if not has_audio(vid_files[0]):
            aud_map = "" # don't add audio if the first video doesn't have an audio stream in it
        else:
            aud_map = f"-map {aud_input}:a " # grab the audio from the first video file

    ret = ''
    if (not os.path.exists(vid_output)) or overwrite:
        this_cmd = f'ffmpeg -y {vid_inputs}{aud_input_str} -filter_complex "[0:v][1:v][2:v][3:v]xstack=inputs=4:layout=0_0|w0_0|0_h0|w0_h0[v]" -map "[v]" {aud_map}-c:v h264_nvenc{aud_codec_str} "{vid_output}"'
        ret = subprocess.getoutput(this_cmd)
    return ret

def has_audio(vid_file:str) -> bool:
    """Does the video file have audio?"""
    ret = subprocess.getoutput(f'ffprobe -i "{vid_file}" -show_streams -select_streams a -loglevel error')
    return bool(ret)

def separate_audio(vid_file:str):
    aud_file = os.path.join(Path(vid_file).parent, Path(vid_file).stem + '.aac')
    ret = subprocess.getoutput(f'ffmpeg -i "{vid_file}" -vn -acodec copy "{aud_file}"')
    return ret

def reencode(vid_files, out_files=None, preset='plain', overwrite=False):
    """
    Often used to re-encode videos using ffmpeg to save disk space
    preset - 
        'plain' - (default) simply re-encodes with h264_nvenc code with default ffmpeg presets
        'color' - settings that were being used for re-encoding optitrack prime color cameras
        'reference' - settings that were being used for re-encoding optitrack reference videos (black and white)
        'ffpreset' - preset is set to this value when using one of ffmpeg's default presets 
            - 'default', 'slow', 'medium', 'fast', 'hp', 'hq', 'bd', 'll', 'llhq', 'llhp', 'lossless', 'losslesshp'
    """
    if preset in ('default', 'slow', 'medium', 'fast', 'hp', 'hq', 'bd', 'll', 'llhq', 'llhp', 'lossless', 'losslesshp'):
        ffpreset = preset
        preset = 'ffpreset'
    known_presets = ('plain', 'color', 'reference', 'quality', 'ffpreset')
    assert preset in known_presets # reference -> b&w motion capture reference
    if isinstance(vid_files, str):
        vid_files = [vid_files]
    
    if out_files is None:
        out_files = []
        for vid_file in vid_files:
            vf = Path(vid_file)
            out_files.append(f'{vf.parent / vf.stem}{"_reencode" if vf.suffix.lower()==".mp4" else ""}.mp4')

    if isinstance(out_files, str):
        out_files = [out_files]
    
    ret = []
    for vid_file, out_file in zip(vid_files, out_files):
        cmd_pre = f'ffmpeg -{"y" if overwrite else "n"} -i "{vid_file}"'
        cmd_post = f'"{out_file}"'
        if preset == 'plain':
            cmd = f'{cmd_pre} -c:v h264_nvenc {cmd_post}'
        elif preset == 'color':
            cmd = f'{cmd_pre} -c:v h264_nvenc -vsync vfr -b:v 12M -an {cmd_post}'
        elif preset == 'reference':
            cmd = f'{cmd_pre} -c:v h264_nvenc -vf transpose=1 {cmd_post}'
        elif preset == 'ffpreset':
            cmd = f'{cmd_pre} -c:v h264_nvenc -preset {ffpreset} {cmd_post}'
        else:
            raise ValueError(f'Unknown preset {preset}')
        ret.append(subprocess.getoutput(cmd))
    return ret

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
