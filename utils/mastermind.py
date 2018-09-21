import numpy as np
from collections import defaultdict
from operator import itemgetter
import json
import pickle
import os
import time
import random
# from .writer import Writer

class Mastermind():
    '''
    max_age : int
        Maximum number of missed misses before a track is deleted.
    nn_budget: int 
        Maximum size of the appearance descriptors, if None, no budget is enforced
    '''
    def __init__(self, cam, emoji_list, Tracker_class, max_age=2,
                nn_budget=10):
        self.cam = cam
        self.tracker = Tracker_class(max_age = max_age, 
                                    nn_budget = nn_budget) 
        # self.emoji2track = {}
        self.track2emoji = {}
        self.unassigned_emojis = emoji_list
        self.original_emojis = emoji_list
        # self.writer = Writer(writeDir, context, cam, max_store=None)

    def _unassign(self, track_id):
        emoji = self.track2emoji.pop(track_id)
        self.unassigned_emojis.append(emoji)

    def _assign(self, track_id):
        if len(self.unassigned_emojis) > 0:
            chosen_emoji = random.choice(self.unassigned_emojis)
            self.unassigned_emojis.remove(chosen_emoji)
            # assumption that no repeat in emoji names is held
        else:
            assert False,'Find evan to code this part'
            #TODO cannot just randomly steal from emoji2track, because some of these track might still be in view
        # self.emoji2track[chosen_emoji] = track_id
        self.track2emoji[track_id] = chosen_emoji
        return chosen_emoji

    def update(self, frame, frame_count, bbs, embeddings):
        tracks = None
        tracks, del_tracks = self.tracker.update_tracks(frame, bbs, embeddings)

        for track in del_tracks:
            self._unassign(track.track_id)

        emoji_bbs = []
        for track in tracks:
            if track.is_tentative():
                # new comer
                track_id = track.track_id
                bb = [int(x) for x in track.to_tlbr()]
                chosen_emoji = self._assign(track_id)
                emoji_bbs.append((chosen_emoji, bb)) 
            elif track.is_confirmed():
                track_id = track.track_id
                bb = [int(x) for x in track.to_tlbr()]
                emoji_bbs.append((self.track2emoji[track_id], bb))

        return emoji_bbs
