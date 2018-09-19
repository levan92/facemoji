#!/usr/bin/python3
import dlib
import os 
import numpy as np
import cv2
import time
import tensorflow as tf 
CURR_DIR = os.path.dirname(os.path.abspath(__file__))

def read_label_map(csv_path):
    label_map = {}
    with open(csv_path,'r') as f:
        for line in f.readlines():
            if line.startswith('#'):
                continue
            idx, cls_label = line.split(',')
            label_map[cls_label.strip()] = int(idx)
    assert 'face' in label_map,'face not in label_map_csv!'
    return label_map

class Mobnet_TF(object):
    def __init__(self, fd_pb, label_csv, gpu_usage=None, threshold=0.5):
        """Tensorflow detector
        """
        self.frozen_graph = fd_pb
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            self.graphDef = tf.GraphDef()
            with tf.gfile.GFile(fd_pb, 'rb') as fid:
                serialized_graph = fid.read()
                self.graphDef.ParseFromString(serialized_graph)
                tf.import_graph_def(self.graphDef, name='')

            config = tf.ConfigProto()
            if gpu_usage is None:
                config.gpu_options.allow_growth = True
                print('Initalising Mobilenet SSD FD at unlimited gpu usage (allow_growth)..')
            else:
                config.gpu_options.per_process_gpu_memory_fraction = gpu_usage
                print('Initalising Mobilenet SSD FD at {} gpu usage..'.format(gpu_usage))
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            # self.windowNotSet = True
        
        self.label_map = read_label_map(label_csv)
        self.threshold = threshold

    def _post_process(self, boxes, scores, classes, im_size):
        bbs = []
        im_height, im_width = im_size
        for i, score in enumerate(scores):
            if score > self.threshold and int(classes[i]) == self.label_map['face']:
                box = boxes[i]
                t = box[0] * im_height
                l = box[1] * im_width
                b = box[2] * im_height
                r = box[3] * im_width
                w = r - l
                h = b - t
                bb = {'rect':{'t': t, 
                              'l': l,
                              'r': r,
                              'b': b, 
                              'w': w, 
                              'h': h },
                      'confidence': score}
                bbs.append(bb)
        return bbs

    def __call__(self, image):
        """
        image: bgr image
        returns: bbs, list of {'rect':{'t': boxes[0], 
                                       'l': boxes[1],
                                       'r': boxes[3],
                                       'b': boxes[2], 
                                       'w': boxes[3] - boxes[1], 
                                       'h': boxes[2] - boxes[0] },
                                'confidence': score}

        """
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        im_size = image_np.shape[:2]
        # image_np = image
        # the array based representation of the image will be used later in order to prepare the
        # result image with boxes and labels on it.
        # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # Each box represents a part of the image where a particular object was detected.
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        # Each score represent how level of confidence for each of the objects.
        # Score is shown on the result image, together with the class label.
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # Actual detection.
        # start_time = time.time()
        (boxes, scores, classes, _) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        # elapsed_time = time.time() - start_time
        # print('inference time cost: {}'.format(elapsed_time))

        return self._post_process(np.squeeze(boxes), np.squeeze(scores), np.squeeze(classes), im_size)

TEMPLATE = np.float32([
        (0.0792396913815, 0.339223741112), (0.0829219487236, 0.456955367943),
        (0.0967927109165, 0.575648016728), (0.122141515615, 0.691921601066),
        (0.168687863544, 0.800341263616), (0.239789390707, 0.895732504778),
        (0.325662452515, 0.977068762493), (0.422318282013, 1.04329000149),
        (0.531777802068, 1.06080371126), (0.641296298053, 1.03981924107),
        (0.738105872266, 0.972268833998), (0.824444363295, 0.889624082279),
        (0.894792677532, 0.792494155836), (0.939395486253, 0.681546643421),
        (0.96111933829, 0.562238253072), (0.970579841181, 0.441758925744),
        (0.971193274221, 0.322118743967), (0.163846223133, 0.249151738053),
        (0.21780354657, 0.204255863861), (0.291299351124, 0.192367318323),
        (0.367460241458, 0.203582210627), (0.4392945113, 0.233135599851),
        (0.586445962425, 0.228141644834), (0.660152671635, 0.195923841854),
        (0.737466449096, 0.182360984545), (0.813236546239, 0.192828009114),
        (0.8707571886, 0.235293377042), (0.51534533827, 0.31863546193),
        (0.516221448289, 0.396200446263), (0.517118861835, 0.473797687758),
        (0.51816430343, 0.553157797772), (0.433701156035, 0.604054457668),
        (0.475501237769, 0.62076344024), (0.520712933176, 0.634268222208),
        (0.565874114041, 0.618796581487), (0.607054002672, 0.60157671656),
        (0.252418718401, 0.331052263829), (0.298663015648, 0.302646354002),
        (0.355749724218, 0.303020650651), (0.403718978315, 0.33867711083),
        (0.352507175597, 0.349987615384), (0.296791759886, 0.350478978225),
        (0.631326076346, 0.334136672344), (0.679073381078, 0.29645404267),
        (0.73597236153, 0.294721285802), (0.782865376271, 0.321305281656),
        (0.740312274764, 0.341849376713), (0.68499850091, 0.343734332172),
        (0.353167761422, 0.746189164237), (0.414587777921, 0.719053835073),
        (0.477677654595, 0.706835892494), (0.522732900812, 0.717092275768),
        (0.569832064287, 0.705414478982), (0.635195811927, 0.71565572516),
        (0.69951672331, 0.739419187253), (0.639447159575, 0.805236879972),
        (0.576410514055, 0.835436670169), (0.525398405766, 0.841706377792),
        (0.47641545769, 0.837505914975), (0.41379548902, 0.810045601727),
        (0.380084785646, 0.749979603086), (0.477955996282, 0.74513234612),
        (0.523389793327, 0.748924302636), (0.571057789237, 0.74332894691),
        (0.672409137852, 0.744177032192), (0.572539621444, 0.776609286626),
        (0.5240106503, 0.783370783245), (0.477561227414, 0.778476346951)])

TPL_MIN, TPL_MAX = np.min(TEMPLATE, axis=0), np.max(TEMPLATE, axis=0)
MINMAX_TEMPLATE = (TEMPLATE - TPL_MIN) / (TPL_MAX - TPL_MIN)

#: Landmark indices.
INNER_EYES_AND_BOTTOM_LIP = [39, 42, 57]
OUTER_EYES_AND_NOSE = [36, 45, 33]

def bb2dlibrect(bb):
    return dlib.rectangle(left=int(bb['rect']['l']), 
                          top=int(bb['rect']['t']), 
                          right=int(bb['rect']['r']), 
                          bottom=int(bb['rect']['b']))

class Mobnet_FD:
    def __init__(self, fd_pb=None, label_csv=None, landmarks_dat=None, gpu_usage=None, max_n =None, **kwargs):
        if fd_pb is None:
            fd_pb = os.path.join(CURR_DIR, "mobnet_frozen_graph.pb")
            assert os.path.exists(fd_pb),'{} does not exist'.format(fd_pb)

        if label_csv is None:
            label_csv = os.path.join(CURR_DIR, "mobnet_label_map.csv")
            assert os.path.exists(label_csv),'{} does not exists'.format(label_csv)

        if landmarks_dat is None:
            landmarks_dat = os.path.join(CURR_DIR, 'shape_predictor_68_face_landmarks.dat')
            # landmarks_dat = os.path.join(CURR_DIR, 'shape_predictor_5_face_landmarks.dat')
            assert os.path.exists(landmarks_dat),'{} does not exists'.format(landmarks_dat)

        self.detector = Mobnet_TF(fd_pb, label_csv, gpu_usage=gpu_usage)
        self.predictor = dlib.shape_predictor(landmarks_dat)
        self.max_n = max_n
        # warm up
        ret = self.detector(np.zeros((10,10,3), dtype=np.uint8))
        self.i = 0
        print("FACE DETECTION: Mobilenet SSD FD object initalised")

    def detect(self, img3chnl):
        '''
        returns: bbs, list of {'rect':{'t': boxes[0], 
                                       'l': boxes[1],
                                       'r': boxes[3],
                                       'b': boxes[2], 
                                       'w': boxes[3] - boxes[1], 
                                       'h': boxes[2] - boxes[0] },
                                'confidence': score}
        '''
        assert img3chnl is not None,'FD didnt rcv img'
        
        try:
            return self.detector(img3chnl)
        except Exception as e:
            print("WARNING from FD detect: {}".format(e))
            return []

    def detect_bb(self, img3chnl):
        '''
        pass through fn for enrol2phone.py
        '''
        return self.detect(img3chnl)        
    
    def _detect_batch(self, img3chnls):
        '''
        :return: array of bbs.
        '''
        assert img3chnls is not None,'FD didnt rcv img'
        all_bbs = []
        for img3chnl in img3chnls:
            try:
                if img3chnl is None or img3chnl.dtype != np.uint8:
                    all_bbs.append([])
                else:
                    all_bbs.append(self.detector(img3chnl))
            except Exception as e:
                print("WARNING from FD detect_batch: {}".format(e))
                all_bbs.append([])
        return all_bbs

    def align(self, img3chnl, bb_dlib_rect, imgDim):
        # start = time.time()
        points = self.predictor(img3chnl, bb_dlib_rect)
        # mid = time.time()
        landmarks = list(map(lambda p:(p.x, p.y), points.parts()))
        npLandmarks = np.float32(landmarks)
        npLandmarksIndices = np.array(INNER_EYES_AND_BOTTOM_LIP)
        H = cv2.getAffineTransform(npLandmarks[npLandmarksIndices],
                                   imgDim * MINMAX_TEMPLATE[npLandmarksIndices])
        aligned_face = cv2.warpAffine(img3chnl, H, (imgDim, imgDim))
        # print('Time taken for pred:{}, for affine:{}'.format(mid - start, time.time()-mid))
        return aligned_face

    def _align_batch(self, img3chnl, bbs, imgDim):
        '''
        For batch faces
        bbs: list of bb, not dlib rect yet
        '''
        assert img3chnl is not None, 'Landmark predictor didnt rcv img'
        assert bbs is not None, 'Landmark predictor didnt rcv bb'
        aligned_faces = []
        for bb in bbs:
            aligned_face = self.align(img3chnl, bb2dlibrect(bb), imgDim)
            # cv2.imwrite('1/aligned_{}.jpg'.format(self.i), aligned_face)
            # cv2.imwrite('img.jpg', img3chnl)
            aligned_faces.append(aligned_face)
            self.i+=1
        return aligned_faces

    def detect_align_faces_batch(self, img3chnls, imgDim=96, num_face=None):
        all_bbs = self._detect_batch(img3chnls)
        all_aligned_faces = []
        all_bbs_less = []
        for i, bbs in enumerate(all_bbs):
            if len(bbs)==0:
                aligned_faces = []                
            else:
                if self.max_n is not None:
                    # print(bbs)
                    bbs = sorted(bbs, key=lambda bb: bb['rect']['w'] * bb['rect']['h'], reverse=True)[:self.max_n]
                aligned_faces = self._align_batch(img3chnls[i], bbs, imgDim)
            
            all_aligned_faces.append(aligned_faces)
            all_bbs_less.append(bbs)
        return all_bbs_less, all_aligned_faces

    def detect_align_faces(self, img3chnl, imgDim=96, num_face=None):
        all_bbs, all_aligned_faces = self.detect_align_faces_batch([img3chnl])
        return all_bbs[0], all_aligned_faces[0]


if __name__ == '__main__':
    fd = Mobnet_FD()
    img = cv2.imread('/home/dh/Workspace/FR/Data/pics/IMG_4670.jpeg')
    fd.detect(img)
