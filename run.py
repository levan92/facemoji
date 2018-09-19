#!/usr/bin/python3
import cv2
import time
import argparse
import pickle
import os
import numpy as np
from copy import deepcopy
from datetime import datetime

from utils.videoStream import VideoStream

from faceDet.mobnet_dlib import Mobnet_FD as FaceDet
from faceReg.FR_embedding import FR_openface as FaceReg
from tracker.deepsort_tracker_FR import DeepSort as Tracker

parser = argparse.ArgumentParser()
parser.add_argument('-v','--vid_paths', nargs='+', help='Video filepaths/streams for \
                    all cameras, e.g.: 0')
parser.add_argument('--time',help='Verbose toggle to show timings',action='store_true')
# parser.add_argument('--display',help='Verbose toggle to display intermediate live video',action='store_true')
parser.add_argument('--capture',help='Dir for storing frames', type=str)

args = parser.parse_args()
video_paths = args.vid_paths
# show_live = args.display
show_live = True
out_dir = args.capture
show_time = args.time

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

faceReg = FaceReg(gpu_usage=0.5)
#faceDet network is loaded after faceReg as gpu usage % cannot be specified for faceDet
faceDet = FaceDet()
tracker = Tracker()
# # a list of Masterminds
# masterminds = []
# if capture:
#     mm_writeDir = os.path.join(out_dir, 'outTracks')
# else:
#     mm_writeDir = None
# for cam in cam_names:
#     masterminds.append(Mastermind(cam, classifier, btManager, Tracker, 
#                                     max_age=25, 
#                                     pred_sampling=1, 
#                                     # pred_sampling=3, 
#                                     conf_threshold=0.2, # DOESNT APPLY if clf_type is LinearSvm, please look into mastermind.py for the hardcoded bucket values  
#                                     diff_threshold=0.1,
#                                     det_conf_threshold=0.9,
#                                     vote_mode=False, # Voting mode is a thresholded voting system
#                                     # floating mode is a voting system weighted to confidence of prediction
#                                     push_top_k=None, # num of top suspects to push to UI given above conf threshold per evaluation, None means all above threshold are sent.
#                                     avrg_alpha=0.15, # (0,1.0], closer to 1.0 means we put full weight on the latest,
#                                     writeDir=mm_writeDir
#                                     )
#                       )
if capture:
    vm_writeDir = os.path.join(out_dir, 'outFrames')
else:
    vm_writeDir = None

# vidManager = VideoManager(cam_names, video_paths, ipFinder, 
#                 queueSize=5,  # queueSize: number of frames stored in queue, smaller this number, the more 'real-time' it gets, but 'throws away' more frames. 
#                 reconnectThreshold=2, # seconds before reconnecting whenever stream is stuck (no new frames in)
#                 writeDir=vm_writeDir, #If none, default is 'outFrames' at wd
#                 )
# drawer = Drawer(num_vid_streams, vidInfos=vidManager.getAllInfo())

## Currently only supports one vid stream
video_path = video_paths[0]
cam_name = cam_names[0]
print('Video name: {}'.format(cam_name))
print('Video path: {}'.format(video_path))

stream = VideoStream(cam_name, video_path, writeDir=vm_writeDir) 
frame_count = 0
stream.start()
start_whole = time.time()
try:
    while True:
        if stream.more():
            frame = stream.read()

            bbs, aligned_faces = faceDet.detect_align_faces(frame)
            embeddings = faceReg.get_embeds(aligned_faces)

            # emojis = mastermind.update(frame, frame_count, bbs, embeddings))
            # show_frame = drawer.draw_emoji(frame, emojis)
            show_frame = deepcopy(frame)
            if show_live:
                cv2.imshow('',show_frame)
            if capture:
                stream.capture(frame_count)            
            # mastermind.reset()
            frame_count += 1
        
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
