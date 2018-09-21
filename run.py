#!/usr/bin/python3
import cv2
import time
import argparse
import pickle
import os
import numpy as np
from copy import deepcopy
from datetime import datetime
from tqdm import tqdm

from utils.misc import process_emoji_dir
from utils.videoStream import VideoStream
from utils.mastermind import Mastermind
from utils.drawer import Drawer

from faceDet.mobnet_dlib import Mobnet_FD as FaceDet
from faceReg.FR_embedding import FR_openface as FaceReg
from tracker.deepsort_tracker_FR import DeepSort as Tracker

parser = argparse.ArgumentParser()
parser.add_argument('-v','--vid_paths', nargs='+', help='Video filepaths/streams for \
                    all cameras, e.g.: 0',required=True)
parser.add_argument('--rtsp',help='Additional flag for when RTSP stream is the input',action='store_true')
parser.add_argument('--time',help='Verbose toggle to show timings',action='store_true')
# parser.add_argument('--display',help='Verbose toggle to display intermediate live video',action='store_true')
parser.add_argument('-e','--emo_dir',help='Emoji directory', type=str,required=True)
parser.add_argument('--out',help='Dir for storing processed frames', type=str)

args = parser.parse_args()
video_paths = args.vid_paths
rtsp_mode = args.rtsp
# show_live = args.display
show_live = True
out_dir = args.out
show_time = args.time
emo_dir = os.path.normpath(args.emo_dir)
assert os.path.isdir(emo_dir),'emo_dir is not a directory!'
emo_list = process_emoji_dir(emo_dir)

if out_dir is not None:
    capture = True
    try:
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        assert os.path.isdir(out_dir),'out_dir not a dir!'
    except Exception as e:
        print('Out dir exception:{}'.format(e))
        # capture = False
    # print('Will attempt to write out to {}, is okay if not present now.'.format(out_dir))
else:
    capture = False

cam_names = []
if rtsp_mode:
    cam_names = ['rtsp_stream0']
    video_mode = False
else:
    temp = []
    for vid in video_paths:
        if vid.isdigit():
            int_vid = int(vid)
            temp.append(int_vid)
            cam_names.append('Webcam{}'.format(int_vid))
            video_mode = False
        else:
            video_mode = True
            temp.append(vid)
            cam_names.append(''.join(os.path.basename(vid).split('.')[:-1]))
    video_paths = temp

num_vid_streams = len(video_paths)
## Currently only supports one vid stream
video_path = video_paths[0]
cam_name = cam_names[0]
print('Video name: {}'.format(cam_name))
print('Video path: {}'.format(video_path))
if video_mode:
    # then video stream does not drop frames to maintain real-time-liness, therefore, queue maxlen is None
    stream = VideoStream(cam_name, video_path, queueSize = None)
else:
    stream = VideoStream(cam_name, video_path)  
if video_mode:
    video_info = stream.getInfo()

if capture:
    out_frames = []
    # vm_writeDir = os.path.join(out_dir, 'outFrames')
# else:
    # vm_writeDir = None

faceReg = FaceReg(gpu_usage=0.5)
#faceDet network is loaded after faceReg as gpu usage % cannot be specified for faceDet
faceDet = FaceDet(threshold=0.1)
mastermind = Mastermind(cam_name, emo_list, Tracker, max_age=2, nn_budget=10)
drawer = Drawer(emo_dir)
frame_count = 0
started = False
stream.start()
if not video_mode:
    time.sleep(1)
start_whole = time.time()
try:
    while True:
        if stream.more():
            # print('Reading for new frame')
            if not started:
                started = True
            frame = stream.read()

            bbs, aligned_faces = faceDet.detect_align_faces(frame)
            embeddings = faceReg.get_embeds(aligned_faces)
            emoji_bbs = mastermind.update(frame, frame_count, bbs, embeddings)
            show_frame = drawer.draw_emoji(frame, emoji_bbs)
            # show_frame = deepcopy(frame)
            if show_live:
                cv2.imshow('',show_frame)
            if capture:
                out_frames.append(show_frame)
                # stream.capture(frame_count)            
            frame_count += 1

            # cv2.waitKey(0)
            if cv2.waitKey(1) & 0xFF == ord('q'): # FOR CNN
                print('Q pressed! Terminating..')
                break
        elif video_mode and started:
            print('Video Completed!')
            break

except KeyboardInterrupt:
    print('KeyboardInterrupt! Terminating..')
    cv2.destroyAllWindows()

processing_fps = int(frame_count/(time.time()-start_whole))
print('Avg FPS:', processing_fps)
stream.stop()
cv2.destroyAllWindows()

print('Writing out video..')
if video_mode:
    write_fps = video_info['fps']
    out_name = cam_name
else:
    write_fps = processing_fps
    out_name = 'out' 

out_path = os.path.join(out_dir, '{}.avi'.format(out_name))
i = 1
while os.path.exists(out_path):
    out_path = os.path.join(out_dir, '{}_{}.avi'.format(out_name, i))
    i += 1
h, w = out_frames[0].shape[:2]
fourcc = cv2.VideoWriter_fourcc(*'h264') 
out = cv2.VideoWriter(out_path, fourcc, write_fps, (w,h))
for frame in tqdm(out_frames):
    out.write(frame)
out.release()
print('Written to {} @ {:0.2f}FPS'.format(out_path, write_fps))
