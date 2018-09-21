from dlib import mmod_rectangle
import cv2
import numpy as np
import copy
import dlib
import os

class Drawer(object):
    def __init__(self, emo_dir, scale=1.6):
        # self.color = color
        # self.font = font
        # self.fontScale = 0.7
        # self.fontThickness = 2
        # self.frameHeight = None
        self.scale = scale
        self.emo_dir = emo_dir

    # def _resize(self, frame):
    #     height, width = frame.shape[:2]
    #     if height != self.frameHeight:
    #         scale = float(height) / self.frameHeight
    #         frame = cv2.resize(frame, (int(width / scale), int(self.frameHeight) ) )
    #     return frame

    # def _BGR2BGRA(self, frame):
    #     b_channel, g_channel, r_channel = cv2.split(frame)
    #     alpha_channel = np.ones(b_channel.shape, dtype=b_channel.dtype) * 50 #creating a dummy alpha channel image.
    #     img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))

    def draw_emoji(self, frame, emoji_bbs):
        '''
        Draw emoji is non-trivial omg.
        
        Params
        -------
        emoji_bbs : list of tuples
            Each tuple consist of (emoji_name, bb) pairs
        
        Returns
        -------
        Drawn frame ndarray

        '''
        if emoji_bbs is None or len(emoji_bbs) == 0:
            return frame
        frameDC = copy.deepcopy(frame)
        frame_h, frame_w = frameDC.shape[:2]
        for emoji_name, bb in emoji_bbs:
            l,t,r,b = bb
            w, h = (r-l, b-t)
            if w == 0 or h==0 or h/w > 10:
                continue 

            # If you want to visualise the face bb, uncomment this
            # cv2.rectangle(frameDC, (l,t), (r,b), (0,0,255), 3)

            # Add additional scaling to emoji to cover more generous amount of the face
            add_w = int((w * (self.scale-1))/2)
            add_h = int((h * (self.scale-1))/2)
            l = l-add_w if (l-add_w)>0 else 0
            t = t-add_h if (t-add_h)>0 else 0
            r = r+add_w if (r+add_w)<frame_w-1 else frame_w-1
            b = b+add_h if (b+add_h)<frame_h-1 else frame_h-1
            w, h = (r-l, b-t)

            # Gimme that emoji
            emoji = cv2.imread(os.path.join(self.emo_dir, emoji_name), cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji,(w, h))
            emoji, emoji_alpha = emoji[:,:,:3], emoji[:,:,3]

            # Take the alpha channel of the emoji and create a binary mask
            _, mask = cv2.threshold(emoji_alpha,10,1,cv2.THRESH_BINARY)
            mask = np.repeat(mask[:,:,np.newaxis],3, axis=2).astype(int)
            # its inverse as well
            inv_mask = np.logical_not(mask).astype(int)
            
            # Isolate emoji
            emoji_only = np.multiply(emoji, mask)
            roi = frameDC[t:b,l:r]
            # Remove face
            frame_unmasked = np.multiply(roi, inv_mask)
            
            # Add emoji face to removed face
            roi = frame_unmasked + emoji_only
            # Replace that region of interest
            frameDC[t:b,l:r] = roi


        return frameDC

