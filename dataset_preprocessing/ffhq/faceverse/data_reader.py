import cv2
from PIL import Image
import numpy as np
import threading
import copy
import time
import os
from third_libs.OpenSeeFace.tracker import Tracker
from util_functions import get_length
import mediapipe as mp
import glob


class OnlineReader(threading.Thread):
    def __init__(self, camera_id, width, height, tar_size):
        super(OnlineReader, self).__init__()
        self.camera_id = camera_id
        self.height, self.width = height, width#480, 640# 1080, 1920 480,640 600,800 720,1280 
        self.tar_size = tar_size
        self.frame = np.zeros((height, width, 3), dtype=np.uint8)
        self.frame_num = 0
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(3, width)
        self.cap.set(4, height)
        fourcc= cv2.VideoWriter_fourcc('M','J','P','G')
        self.cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        self.thread_exit = False
        self.thread_lock = threading.Lock()
        self.length_scale = 1.0
        self.tracker = Tracker(width, height, threshold=None, max_threads=1,
                              max_faces=1, discard_after=100, scan_every=300, 
                              silent=True, model_type=3, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
                              use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)

    def get_data(self):
        return copy.deepcopy(self.frame), copy.deepcopy(self.frame_num)

    def run(self):
        while not self.thread_exit:
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                if self.frame_num == 0:
                    preds = self.tracker.predict(frame)
                    if len(preds) == 0 or preds[0].lms is None:
                        print('No face detected in online reader!')
                        time.sleep(0.015)
                        continue
                    # try more times in the fisrt frame for better landmarks
                    for _ in range(3):
                        preds = self.tracker.predict(frame)
                    if len(preds) == 0 or preds[0].lms is None:
                        time.sleep(0.015)
                        print('No face detected in offline reader!')
                        continue
                    lms = (preds[0].lms[:66, :2].copy() + 0.5).astype(np.int64)
                    lms = lms[:, [1, 0]]
                    self.border = 500
                    self.half_length = int(get_length(lms) * self.length_scale)
                    self.crop_center = lms[29].copy() + self.border
                    print('First frame:', self.half_length, self.crop_center)
        
                frame_b = cv2.copyMakeBorder(frame, self.border, self.border, self.border, self.border, cv2.BORDER_CONSTANT, value=0)
                frame_b = Image.fromarray(frame_b[self.crop_center[1] - self.half_length:self.crop_center[1] + self.half_length, 
                                            self.crop_center[0] - self.half_length:self.crop_center[0] + self.half_length])
                align = np.asarray(frame_b.resize((self.tar_size, self.tar_size), Image.ANTIALIAS))
                self.thread_lock.acquire()
                self.frame_num += 1
                self.frame = align
                self.thread_lock.release()
            else:
                self.thread_exit = True
        self.cap.release()


class OnlineDetecder(threading.Thread):
    def __init__(self, camera_id, width, height, tar_size, batch_size):
        super(OnlineDetecder, self).__init__()
        self.onreader = OnlineReader(camera_id, width, height, tar_size)
        self.tracker_tar = Tracker(tar_size, tar_size, threshold=None, max_threads=4,
                                max_faces=1, discard_after=100, scan_every=100, 
                                silent=True, model_type=4, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
                                use_retinaface=1, max_feature_updates=900, static_model=True, try_hard=0)
        self.detected = False
        self.thread_exit = False
        self.tar_size = tar_size
        self.thread_lock = threading.Lock()
        self.batch_size = batch_size
        self.frame = np.zeros((batch_size, tar_size, tar_size, 3), dtype=np.uint8)
        #self.lms = np.zeros((batch_size, 66, 2), dtype=np.int64)
        self.lms = np.zeros((batch_size, 478, 2), dtype=np.int64)
        self.frame_num = 0
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_data(self):
        return copy.deepcopy(self.frame), copy.deepcopy(self.lms), copy.deepcopy(self.frame_num)

    def run(self):
        self.onreader.start()
        while not self.thread_exit:
            self.onreader.thread_lock.acquire()
            align, frame_num = self.onreader.get_data()
            self.onreader.thread_lock.release()
            if frame_num <= self.frame_num:
                #print('wait frame')
                time.sleep(0.015)
                continue
            '''
            preds = self.tracker_tar.predict(align)
            if len(preds) == 0 or preds[0].lms is None:
                print('No face detected in online reader!')
                self.detected = False
                time.sleep(0.015)
                continue
            # try more times in the fisrt frame for better landmarks
            if not self.detected:
                for _ in range(3):
                    preds = self.tracker_tar.predict(align)
                if len(preds) == 0 or preds[0].lms is None:
                    print('No face detected in offline reader!')
                    self.detected = False
                    time.sleep(0.015)
                    continue
            lms = (preds[0].lms[:66, :2].copy() + 0.5).astype(np.int64)
            lms = lms[:, [1, 0]]
            '''
            results = self.face_tracker.process(align)
            if len(results.multi_face_landmarks) == 0:
                print('No face detected in online reader!')
                self.detected = False
                time.sleep(0.015)
                continue
            lms = np.zeros((478, 2), dtype=np.int64)
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                #print(idx, landmark.x)
                #if idx < 468:
                lms[idx, 0] = int(landmark.x * self.tar_size)
                lms[idx, 1] = int(landmark.y * self.tar_size)
            
            self.detected = True
            
            self.thread_lock.acquire()
            self.frame[self.frame_num % self.batch_size] = align
            self.lms[self.frame_num % self.batch_size] = lms
            self.frame_num += 1#frame_num
            self.thread_lock.release()
            #print(self.frame_num, frame_num)
        self.onreader.thread_exit = True


class OfflineReader:
    def __init__(self, path, tar_size, image_size, skip_frames=0):
        self.skip_frames = skip_frames
        self.tar_size = tar_size
        self.image_size = image_size
        self.cap = cv2.VideoCapture(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.num_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_num = 0
        self.length_scale = 1.0
        self.detected = False
        self.height, self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.tracker = Tracker(self.width, self.height, threshold=None, max_threads=1,
                              max_faces=1, discard_after=100, scan_every=300, 
                              silent=True, model_type=4, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
                              use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def get_data(self):
        ret, frame = self.cap.read()
        if ret:
            while self.frame_num < self.skip_frames:
                _, frame = self.cap.read()
                self.frame_num += 1
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if not self.detected:
                preds = self.tracker.predict(frame)
                if len(preds) == 0 or preds[0].lms is None:
                    print('No face detected in offline reader!')
                    self.frame_num += 1
                    self.detected = False
                    return False, False, [], [], []
                # try more times in the fisrt frame for better landmarks
                for _ in range(3):
                    preds = self.tracker.predict(frame)
                    if len(preds) == 0 or preds[0].lms is None:
                        print('No face detected in offline reader!')
                        self.frame_num += 1
                        self.detected = False
                        return False, False, [], [], []
                lms = (preds[0].lms[:66, :2].copy() + 0.5).astype(np.int64)
                lms = lms[:, [1, 0]]
                self.border = 500
                self.half_length = int(get_length(lms) * self.length_scale)
                self.crop_center = lms[29].copy() + self.border
                print('First frame:', self.half_length, self.crop_center)
    
            frame_b = cv2.copyMakeBorder(frame, self.border, self.border, self.border, self.border, cv2.BORDER_CONSTANT, value=0)
            frame_b = Image.fromarray(frame_b[self.crop_center[1] - self.half_length:self.crop_center[1] + self.half_length, 
                                        self.crop_center[0] - self.half_length:self.crop_center[0] + self.half_length])
            align = np.asarray(frame_b.resize((self.tar_size, self.tar_size), Image.ANTIALIAS))
            outimg = np.asarray(frame_b.resize((self.image_size, self.image_size), Image.ANTIALIAS))
            
            results = self.face_tracker.process(align)
            if len(results.multi_face_landmarks) == 0:
                print('No face detected in offline reader!')
                self.frame_num += 1
                return False, False, [], [], []
            
            lms = np.zeros((478, 2), dtype=np.int64)
            for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
                #print(idx, landmark.x)
                #if idx < 468:
                lms[idx, 0] = int(landmark.x * self.tar_size)
                lms[idx, 1] = int(landmark.y * self.tar_size)
            self.frame_num += 1
            self.detected = True
            return True, align, lms, outimg, self.frame_num
        else:
            self.cap.release()
            print('Reach the end of the video')
            return False, True, [], [], []


class ImageReader_FFHQ:
    def __init__(self, path, image_size):
        self.path = path
        self.imagelist = os.listdir(path)
        self.imagelist = sorted([path for path in self.imagelist if path.endswith('png')])
        self.num_frames = len(self.imagelist)
        self.frame_num = 0
        #self.height = height
        #self.width = width
        self.image_size = image_size
        
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #self.tracker = Tracker(width, height, threshold=None, max_threads=1,
        #                max_faces=1, discard_after=100, scan_every=300, 
        #                silent=True, model_type=3, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
        #                use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)

    def get_data(self):
        if self.frame_num == self.num_frames:
            print('Reach the end of the folder')
            return False, True, [], [], []

        frame = cv2.imread(os.path.join(self.path, self.imagelist[self.frame_num]), -1)[:, :, :3]
        frame = frame[:, :, ::-1]
        height, width = frame.shape[:2]
        if height != self.image_size or width != self.image_size:
            frame = cv2.resize(frame, (self.image_size, self.image_size))
        #frame = np.concatenate([frame, frame[-176:]], axis=0)
        
        results = self.face_tracker.process(frame)
        if len(results.multi_face_landmarks) == 0:
            print('No face detected in ' + self.imagelist[self.frame_num])
            self.frame_num += 1
            return False, False, [], [], []
        lms = np.zeros((478, 2), dtype=np.int64)
        for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            #print(idx, landmark.x)
            #if idx < 468:
            lms[idx, 0] = int(landmark.x * self.image_size)
            lms[idx, 1] = int(landmark.y * self.image_size)
        self.frame_num += 1
        return True, frame, lms, self.frame_num, self.imagelist[self.frame_num - 1]


class ImageReader:
    def __init__(self, path, image_size):
        self.path = path
        self.subfile = os.listdir(path)
        self.imagelist = []
        # for sub in self.subfile:
        #     for f in os.listdir(os.path.join(path, sub)):
        #         self.imagelist.append(os.path.join(sub, f))

        self.imagelist = os.listdir(path)
        self.num_frames = len(self.imagelist)
        print(self.num_frames)
        self.frame_num = 0
        #self.height = height
        #self.width = width
        self.image_size = image_size
        
        self.face_tracker = mp.solutions.face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5)
        #self.tracker = Tracker(width, height, threshold=None, max_threads=1,
        #                max_faces=1, discard_after=100, scan_every=300, 
        #                silent=True, model_type=3, model_dir='third_libs/OpenSeeFace/models', no_gaze=True, detection_threshold=0.6, 
        #                use_retinaface=1, max_feature_updates=900, static_model=False, try_hard=0)

    def get_data(self):
        if self.frame_num == self.num_frames:
            print('Reach the end of the folder')
            return False, True, [], [], []

        frame = cv2.imread(os.path.join(self.path, self.imagelist[self.frame_num]), -1)[:, :, :3]
        frame = frame[:, :, ::-1]
        height, width = frame.shape[:2]
        if height != self.image_size or width != self.image_size:
            frame = cv2.resize(frame, (self.image_size, self.image_size))
        #frame = np.concatenate([frame, frame[-176:]], axis=0)
        try:
            results = self.face_tracker.process(frame)
            if len(results.multi_face_landmarks) == 0:
                print('No face detected in ' + self.imagelist[self.frame_num])
                self.frame_num += 1
                return False, False, [], [], []
        except:
            print('No face detected in ' + self.imagelist[self.frame_num])
            self.frame_num += 1
            with open("error.txt", 'a') as f:
                f.write(self.imagelist[self.frame_num - 1] + '\n')
            return False, False, [], [], []
        lms = np.zeros((478, 2), dtype=np.int64)
        for idx, landmark in enumerate(results.multi_face_landmarks[0].landmark):
            #print(idx, landmark.x)
            #if idx < 468:
            lms[idx, 0] = int(landmark.x * self.image_size)
            lms[idx, 1] = int(landmark.y * self.image_size)
        self.frame_num += 1
        return True, frame, lms, self.frame_num, self.imagelist[self.frame_num - 1].split(os.sep)[0] + '_' + self.imagelist[self.frame_num - 1].split(os.sep)[1]

