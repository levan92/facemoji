#!/usr/bin/python3
import cv2
import time
import argparse
import pickle
import os
import numpy as np
from copy import deepcopy
from datetime import datetime

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
parser.add_argument('--time',help='Verbose toggle to show timings',action='store_true')
# parser.add_argument('--display',help='Verbose toggle to display intermediate live video',action='store_true')
parser.add_argument('-e','--emo_dir',help='Emoji directory', type=str,required=True)
parser.add_argument('--capture',help='Dir for storing frames', type=str)

args = parser.parse_args()
video_paths = args.vid_paths
# show_live = args.display
show_live = True
out_dir = args.capture
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
    print('Will attempt to write out to {}, is okay if not present now.'.format(out_dir))
else:
    capture = False

cam_names = []
if video_paths is not None:
    temp = []
    for vid in video_paths:
        if vid.isdigit():
            int_vid = int(vid)
            temp.append(int_vid)
            cam_names.append('Webcam{}'.format(int_vid))
        else:
            temp.append(vid)
            cam_names.append(os.path.basename(vid))
    video_paths = temp

num_vid_streams = len(video_paths)
## Currently only supports one vid stream
video_path = video_paths[0]
cam_name = cam_names[0]
print('Video name: {}'.format(cam_name))
print('Video path: {}'.format(video_path))

if capture:
    vm_writeDir = os.path.join(out_dir, 'outFrames')
else:
    vm_writeDir = None

faceReg = FaceReg(gpu_usage=0.5)
#faceDet network is loaded after faceReg as gpu usage % cannot be specified for faceDet
faceDet = FaceDet(threshold=0.3)
mastermind = Mastermind(cam_name, emo_list, Tracker, nn_budget=10)
drawer = Drawer(emo_dir)
stream = VideoStream(cam_name, video_path, writeDir=vm_writeDir) 
frame_count = 0
stream.start()
start_whole = time.time()
try:
    while True:
        if stream.more():
            # print('Reading for new frame')
            frame = stream.read()

            bbs, aligned_faces = faceDet.detect_align_faces(frame)
            embeddings = faceReg.get_embeds(aligned_faces)
            emoji_bbs = mastermind.update(frame, frame_count, bbs, embeddings)
            show_frame = drawer.draw_emoji(frame, emoji_bbs)
            # show_frame = deepcopy(frame)
            if show_live:
                cv2.imshow('',show_frame)
            if capture:
                stream.capture(frame_count)            
            # mastermind.reset()
            frame_count += 1

            # cv2.waitKey(0)
        if cv2.waitKey(1) & 0xFF == ord('q'): # FOR CNN
            print('Avg FPS:', frame_count/(time.time()-start_whole))
            break

except KeyboardInterrupt:
    print('Avg FPS:', frame_count/(time.time()-start_whole))
    cv2.destroyAllWindows()
    print('Killing facemoji..')
    os._exit(0)

avg_fps = frame_count/(time.time()-start_whole)
print('Avg FPS:', avg_fps)

# video.release()
# vid.stop()
cv2.destroyAllWindows()
print('Killing facemoji..')
os._exit(0)
