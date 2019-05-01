import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
# from keras.models import load_model, model_from_json
from keras.utils import CustomObjectScope
import time
import cv2
import numpy as np
import os
if __name__ == '__main__':
    from openface_nn4_small2 import openface_nn4_small2
else:
    from .openface_nn4_small2 import openface_nn4_small2

CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def norm_img(img):
    img = np.around(img/255.0, decimals=12)
    return img

def unnorm_img(img):
    img = np.around(img*255.0)
    return img

def flip_lr(faces):
    assert isinstance(faces[0],np.ndarray),'face not a np.array!'
    flipped = []
    for face in faces:
        flipped.append(np.fliplr(face))
    return np.array(flipped)

class FR_openface:
    def __init__(self, model_h5=None, gpu_usage=0.5, do_flip=False):
        config = tf.ConfigProto()
        # config.gpu_options.allow_growth = True
        config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
        set_session(tf.Session(config=config))
        # if model_h5 is None:
            # model_h5 = 'nn4.small2.py3.h5'
            # model_h5 = 'nn4.small2.cas-peal.304.unfrozen.h5'
        # model_h5 = os.path.join(CURR_DIR, model_h5)
        # assert os.path.exists(model_h5),'{} does not exists'.format(model_h5)

        if model_h5 is None:
            model_name = 'nn4.small2'
        weights_h5 = os.path.join(CURR_DIR,'{}.weights.h5'.format(model_name))        
        assert os.path.exists(weights_h5),'{} does not exists'.format(weights_h5)

        self.model = openface_nn4_small2(weights = weights_h5)
        # with CustomObjectScope({'tf':tf}):
        #     self.model = load_model(model_h5)     
        self.do_flip = do_flip
        #warm up
        self.model.predict_on_batch(np.zeros((1,96,96,3)))
        print("FACE RECOGNITION: Openface network using Keras initialised")
        return   

    def get_embeds(self, faces_bgr):
        '''
        input:
            faces_bgr: list of faces
        '''
        if faces_bgr is None or len(faces_bgr) == 0:
            return []
        assert faces_bgr is not None,'no face given'
        normed_faces = norm_img(np.array(faces_bgr))

        try:
            embs = self.model.predict_on_batch(normed_faces)
            if self.do_flip:
                flipped_faces = flip_lr(normed_faces)
                flipped_embs = self.model.predict_on_batch(flipped_faces)
                embs = np.concatenate((embs, flipped_embs), axis=1)
        except Exception as e:
            print("WARNING from FR inference: {}".format(e))
            return []
        return embs
    
    def get_embed(self, face_bgr):
        '''
        input:
            face_bgr: only one face
        '''
        return self.get_embeds([face_bgr])

    # def get_embeds(self, faces_bgr):
    #     if faces_bgr is None or len(faces_bgr) == 0:
    #         return []
    #     faces_bgr = np.array(faces_bgr)
    #     normed_faces = norm_img(faces_bgr)
    #     embeddings = self.model.predict_on_batch(normed_faces)
    #     return embeddings

    def get_embeds_batch(self, faces_bgr_batch):
        # if faces_bgr is None or len(faces_bgr) == 0:
            # return []
        # sizes = []
        sizes = [len(cam_faces) for cam_faces in faces_bgr_batch]
        # for cam_faces in faces_bgr_batch:
        #     sizes.append(len(cam_faces))
        #     cam_faces = np.array(cam_faces)
        
        flat_faces = [face for cam_faces in faces_bgr_batch for face in cam_faces]
        flat_embeddings = []
        if len(flat_faces) > 0:
            flat_embeddings = self.get_embeds(flat_faces)
            if len(flat_embeddings) == 0:
                return []
        # flat_faces = np.array(flat_faces)
        # flat_faces = norm_img(flat_faces)
        # # print(flat_faces)
        # # start = time.time()
        # flat_embeddings = []

        # if len(flat_faces) > 0:
        #     try:
        #         flat_embeddings = self.model.predict_on_batch(flat_faces)
        #     except Exception as e:
        #         print("WARNING from FR_batch: {}".format(e))
        #         return []

        # print('pure embed takes: {}'.format(time.time()-start))
        all_embeddings = []
        for size in sizes:
            all_embeddings.append(flat_embeddings[:size])
            flat_embeddings = flat_embeddings[size:]

        return all_embeddings



if __name__ == "__main__":
    # import time
    faceReg = FR_openface(gpu_usage=0.8)
    faceReg.model.summary()
    # image = cv2.imread('/home/dh/Workspace/FR/master_fr/dlib_FD/test_frame1.png')
    # image = cv2.resize(image, (96,96))
    # print(image.shape)
    # num_face = 10
    # num_cam = 10
    # faces = [image] * num_face
    # cams_faces = [faces] * num_cam
    # cams_faces = np.array(cams_faces)
    # print(cams_faces.shape)
    # n = 20
    # start = time.time()
    # for _ in range(n):
    #     embeddings = faceReg.get_embeds_batch(cams_faces)
   
    
    # print('time for face reg:{}'.format((time.time() - start)/n))
