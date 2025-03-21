"""
Tools for working with videos. It contains some wrappers that call ffmpeg for processing videos.

Use the gui module for browsing videos.
"""
import os
import re
import subprocess
import time
import urllib
from pathlib import Path

import cv2 as cv
import ffmpeg
from decord import VideoReader, cpu, gpu

CLIP_FOLDER = 'C:\\data\\_clipcollection'

AUDIO_FILE_EXTENSIONS = ('.3gp', '.aa', '.aac', '.aax', '.act', '.aiff', '.alac', '.amr', '.ape', '.au', '.awb', '.dss', '.dvf', '.flac', '.gsm', '.iklax', '.ivs', '.m4a', '.m4b', '.mmf', '.movpkg', '.mp3', '.mpc', '.msv', '.nmf', '.oga', '.mogg', '.opus', '.ra', '.rf64', '.sln', '.tta', '.voc', '.vox', '.wav', '.wma', '.wv', '.8svx', '.cda')
VIDEO_FILE_EXTENSIONS = ('.mkv', '.flv', '.vob', '.ogv', '.drc', '.avi', '.mov', '.wmv', '.yuv', '.mts', '.m2ts', '.ts', '.qt', '.rmvb', '.viv', '.asf', '.amv', '.mp4', '.m4p', '.m4v', 'mpg', '.mpe', '.mpv', '.mpeg', '.mp2', '.m2v', '.m4v', '.svi', '.3gp', '.3g2', '.mxf', '.roq', '.nsv', '.flv')
AUDIO_OR_VIDEO_FILE_EXTENSIONS = ('.raw', '.webm', '.ogg', '.rm')

def ffmpeg_is_found():
    ret = subprocess.getoutput('ffmpeg -version')
    if ret.startswith('ffmpeg version'):
        return True
    return False

if not ffmpeg_is_found():
    print('WARNING: ffmpeg not found. This video module will not work.')

def _get_codec_types(vid_file:str):
    return subprocess.getoutput(f'ffprobe -loglevel error -show_entries stream=codec_type -of default=nw=1 "{vid_file}"')

def is_video(vid_file:str, verbose=False) -> bool: # refactored into datanavigator.utils
    codec_types = _get_codec_types(vid_file)
    if verbose:
        my_print = print
    else:
        my_print = lambda x:None

    ret = False
    if 'Invalid data' in codec_types:
        my_print('Not an audio or video file')
    if 'codec_type=video' in codec_types:
        ret = True
        my_print('Video stream found')
    if 'codec_type=audio' in codec_types:
        my_print('Audio stream found')
    return ret

def has_audio(vid_file:str) -> bool:
    codec_types = _get_codec_types(vid_file)
    return 'codec_type=audio' in codec_types

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

def separate_audio(vid_file:str):
    aud_file = os.path.join(Path(vid_file).parent, Path(vid_file).stem + '.aac')
    if not os.path.exists(aud_file):
        print(f"Separating audio from {aud_file}")
        ret = subprocess.getoutput(f'ffmpeg -i "{vid_file}" -vn -acodec copy "{aud_file}"')
        return ret
    print(f"{aud_file} already exists. Skipping!")
        

def process_slomo(vid_file:str, factor=8.0):
      vid_file_out = os.path.join(Path(vid_file).parent, Path(vid_file).stem + '_slomo.mp4')
      if not os.path.exists(vid_file_out):
        ret = subprocess.getoutput(f'ffmpeg -i "{vid_file}" -c:v h264_nvenc -filter:v "setpts={factor:.1f}*PTS" -an "{vid_file_out}"')
        return ret
      print(f"{vid_file_out} already exists. Skipping!")

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
        elif preset == 'quality':
            cmd = f'{cmd_pre} -c:v h264_nvenc -q:v 0 {cmd_post}'
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

try:
    def download(url, start_time=None, end_time=None, dur=None, full_file=False):
        """
        Download a clip from a YouTube video.
        Downloads to the path specified in the module variable CLIP_FOLDER
        Returns the name of the downloaded video.
        """
        from pytube import YouTube

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
except ModuleNotFoundError:
    print('Install pytube to attemp video downloading from youtube.')