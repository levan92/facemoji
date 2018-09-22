#!/usr/bin/python3
from threading import Thread
import sys
import cv2
import os
from datetime import datetime
from queue import Queue
from collections import deque
import time
import re

class FramesStream:
    def __init__(self, camName, frames_dir):
        assert os.path.isdir(frames_dir),'frames dir not a dir'
        self.camName = camName
        self.frames_dir = frames_dir
        find_nums = re.compile(r"[+-]?\d+(?:\.\d+)?")
        frames = [(f, int(find_nums.search(f).group(0))) for f in os.listdir(frames_dir) if f.endswith(('jpg','png'))]
        self.frames = sorted(frames, key=lambda x: x[1])
        self.total_frames = len(self.frames)
        print('FramesStream for {} initialised!'.format(self.camName))

    def getInfo(self):
        video_info = {}
        h, w = cv2.imread(os.path.join(self.frames_dir, self.frames[0][0])).shape[:2]
        video_info['width'] = int(w)
        video_info['height'] = int(h)
        video_info['total_frames'] = self.total_frames
        return video_info

    def start(self):
        self.needle = 0
        print('FileStream started')

    def read(self):
        self.currentFrame = cv2.imread(os.path.join(self.frames_dir, self.frames[self.needle][0]))
        self.needle += 1
        return self.currentFrame

    def more(self):
        return self.needle < self.total_frames

    def stop(self):
        self.stopped = True
        print('FileStream stopped!')
        # self.stream.release()


if __name__ == '__main__':
    # stream = VideoStream('abc', 0) 
    stream = FramesStream('abc', './framesDir')
    info = stream.getInfo()
    print(info)
    stream.start()
    while True:
        if stream.more():
            frame = stream.read()

            cv2.imshow('',frame)
            cv2.waitKey(1)
        else:
            break
    # string = '123.png'
    # # match = re.search(r'\d+.?\d*', string)
    # nums = re.compile(r"[+-]?\d+(?:\.\d+)?")
    # match = nums.search(string).group(0)
    # print(match)
 
