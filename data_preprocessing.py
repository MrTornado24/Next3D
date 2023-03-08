from email.mime import base
from logging import raiseExceptions
from urllib.request import parse_http_list
import trimesh
import numpy as np
import glob
import os
import json
from mtcnn import MTCNN

# def load_obj(path):
#     """Load wavefront OBJ from file."""
#     v = []
#     vindices = []

#     with open(path, "r") as f:
#         while True:
#             line = f.readline()

#             if line == "":
#                 break

#             if line[:2] == "v ":
#                 v.append([float(x) for x in line.split()[1:]])
#             elif line[:2] == "f ":
#                 vindices.append([int(entry.split('/')[0]) - 1 for entry in line.split()[1:]])

#     return np.array(v), np.array(vindices)


# deca_mesh = 'data/ffhq/meshes512x512/00000/img00000031.obj'
# v, f = load_obj(deca_mesh)
# mesh = trimesh.Trimesh(v * 2.6, f)
# shift = [-0.0013, -0.1344, -0.0390]
# trimesh.exchange.export.export_mesh(mesh, 'debug.obj')

# eg3d_mesh = 'out/eg3d_mesh.obj'
# v, f = load_obj(eg3d_mesh)
# mesh = trimesh.Trimesh(v / 512 - 0.5, f)
# trimesh.exchange.export.export_mesh(mesh, "eg3d_debug.obj")

# save_root = 'D:/datasets/VFHQ_processed/meshes512x512'
# mesh_paths = glob.glob('D:/projects/DECA/debug_2/*')
# for path in mesh_paths:
#     basedir = os.path.basename(path)
#     obj_files = glob.glob(path + '/*.obj')
#     obj_files = [obj for obj in obj_files if not obj.endswith('detail.obj')]
#     for obj in obj_files:
#         new_path = os.path.join(save_root, basedir)
#         new_path = new_path.replace('/', '\\')
#         os.makedirs(new_path, exist_ok=True)
#         obj = obj.replace('/', '\\')
#         cmd = f'move {obj} {new_path}'
#         os.system(cmd)

# from tqdm import tqdm
# dataset = {'labels':[]}
# root = glob.glob('D:/projects/rt_gene/gazes512x512/*')
# for path in tqdm(root):
#     files = glob.glob(path + '/*.txt')
#     for file in files:
#         filename = file[33:].replace('txt', 'png')
#         with open(file,'r') as f:
#             gaze = f.readline().strip('\n')
#             gaze = np.array(np.mat(gaze))[0].tolist()
#         dataset["labels"].append([filename, gaze])
        
# with open('dataset_gaze.json', "w") as f:
#     json.dump(dataset, f, indent=4)

# root = glob.glob('data/vfhq/images512x512/*')
# fname = 'data/vfhq/images512x512/dataset.json'
# with open(fname, 'rb') as f:
#     labels = json.load(f)['labels']

# labels = dict(labels)
# f = open('invalid.txt', 'w')
# for path in root:
#     sub_paths = glob.glob(path + '/*.png')
#     for sub in sub_paths:
#         sub = os.path.join(sub.split('\\')[-2], sub.split('\\')[-1])
#         try:
#             _ = labels[sub]
#         except:
#             f.write(sub+'\n')
        
# f.close()

# # 总共5981个视频序列
# # 最短的100帧
# root_path = glob.glob('F:/VFHQ-512-new/*')
# new_root = 'D:/datasets/VFHQ_new/images512x512'
# os.makedirs(new_root, exist_ok=True)
# for path in root_path:
#     sub_paths = sorted(glob.glob(path + '/*.png'))
#     sub_paths = [path for i, path in enumerate(sub_paths) if i % 5 == 0][:20]
#     id = os.path.basename(path)
#     new_path = os.path.join(new_root, id).replace('/', '\\')
#     os.makedirs(new_path, exist_ok=True)
#     for sub_path in sub_paths:
#         sub_path = sub_path.replace('/', '\\')
#         cmd = f'xcopy {sub_path} {new_path}'
#         os.system(cmd)


import os
import sys
import requests
import html
import hashlib
import PIL.Image
import PIL.ImageFile
import numpy as np
import scipy.ndimage
import threading
import queue
import time
import json
import uuid
import glob
import argparse
import itertools
import shutil
from collections import OrderedDict, defaultdict
import cv2
import scipy.io as sio
from imutils import face_utils
from scipy.io import savemat
import dlib
from tqdm import tqdm
from skimage import io
from PIL import Image, ImageDraw

def recreate_aligned_images_fast(mat_data, src_file='demo.png', dst_dir='aligned512x512', output_size=512, transform_size=4096, enable_padding=True, save_path=None):
    print('Recreating aligned images...')
    os.makedirs(dst_dir, exist_ok=True)
    if save_path is None:
        save_path = src_file.replace('images512x512', dst_dir)
    
    mat = sio.loadmat(mat_data)
    # Parse landmarks.
    # pylint: disable=unused-variable
    lm = np.array(mat['face_keypoints_2d'])[:, :2]
    lm_chin          = lm[0  : 17]  # left-right
    lm_eyebrow_left  = lm[17 : 22]  # left-right
    lm_eyebrow_right = lm[22 : 27]  # left-right
    lm_nose          = lm[27 : 31]  # top-down
    lm_nostrils      = lm[31 : 36]  # top-down
    lm_eye_left      = lm[36 : 42]  # left-clockwise
    lm_eye_right     = lm[42 : 48]  # left-clockwise
    lm_mouth_outer   = lm[48 : 60]  # left-clockwise
    lm_mouth_inner   = lm[60 : 68]  # left-clockwise

    # Calculate auxiliary vectors.
    eye_left     = np.mean(lm_eye_left, axis=0)
    eye_right    = np.mean(lm_eye_right, axis=0)
    eye_avg      = (eye_left + eye_right) * 0.5
    eye_to_eye   = eye_right - eye_left
    mouth_left   = lm_mouth_outer[0]
    mouth_right  = lm_mouth_outer[6]
    mouth_avg    = (mouth_left + mouth_right) * 0.5
    eye_to_mouth = mouth_avg - eye_avg

    # Choose oriented crop rectangle.
    x = eye_to_eye - np.flipud(eye_to_mouth) * [-1, 1]
    x /= np.hypot(*x)
    x *= max(np.hypot(*eye_to_eye) * 2.0, np.hypot(*eye_to_mouth) * 1.8)
    y = np.flipud(x) * [-1, 1]
    # q_scale = 1.8
    # x = q_scale * x
    # y = q_scale * y
    c = eye_avg + eye_to_mouth * 0.1
    quad = np.stack([c - x - y, c - x + y, c + x + y, c + x - y])
    qsize = np.hypot(*x) * 2

    # Load in-the-wild image.
    if not os.path.isfile(src_file):
        print('\nCannot find source image. Please run "--wilds" before "--align".')
        return
    img = PIL.Image.open(src_file)
    
    import time

    # Shrink.
    start_time = time.time()
    shrink = int(np.floor(qsize / output_size * 0.5))
    if shrink > 1:
        rsize = (int(np.rint(float(img.size[0]) / shrink)), int(np.rint(float(img.size[1]) / shrink)))
        img = img.resize(rsize, PIL.Image.ANTIALIAS)
        quad /= shrink
        qsize /= shrink
    print("shrink--- %s seconds ---" % (time.time() - start_time))

    # Crop.
    start_time = time.time()
    border = max(int(np.rint(qsize * 0.1)), 3)
    crop = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    crop = (max(crop[0] - border, 0), max(crop[1] - border, 0), min(crop[2] + border, img.size[0]), min(crop[3] + border, img.size[1]))
    if crop[2] - crop[0] < img.size[0] or crop[3] - crop[1] < img.size[1]:
        img = img.crop(crop)
        quad -= crop[0:2]
    print("crop--- %s seconds ---" % (time.time() - start_time))

    # Pad.
    start_time = time.time()
    pad = (int(np.floor(min(quad[:,0]))), int(np.floor(min(quad[:,1]))), int(np.ceil(max(quad[:,0]))), int(np.ceil(max(quad[:,1]))))
    pad = (max(-pad[0] + border, 0), max(-pad[1] + border, 0), max(pad[2] - img.size[0] + border, 0), max(pad[3] - img.size[1] + border, 0))
    if enable_padding and max(pad) > border - 4:
        pad = np.maximum(pad, int(np.rint(qsize * 0.3)))
        img = np.pad(np.float32(img), ((pad[1], pad[3]), (pad[0], pad[2]), (0, 0)), 'reflect')
        h, w, _ = img.shape
        y, x, _ = np.ogrid[:h, :w, :1]
        mask = np.maximum(1.0 - np.minimum(np.float32(x) / pad[0], np.float32(w-1-x) / pad[2]), 1.0 - np.minimum(np.float32(y) / pad[1], np.float32(h-1-y) / pad[3]))
        blur = qsize * 0.02
        img += (scipy.ndimage.gaussian_filter(img, [blur, blur, 0]) - img) * np.clip(mask * 3.0 + 1.0, 0.0, 1.0)
        img += (np.median(img, axis=(0,1)) - img) * np.clip(mask, 0.0, 1.0)
        img = PIL.Image.fromarray(np.uint8(np.clip(np.rint(img), 0, 255)), 'RGB')
        quad += pad[:2]
    print("pad--- %s seconds ---" % (time.time() - start_time))

    # Transform.
    start_time = time.time()
    img = img.transform((transform_size, transform_size), PIL.Image.QUAD, (quad + 0.5).flatten(), PIL.Image.BILINEAR)
    if output_size < transform_size:
        img = img.resize((output_size, output_size), PIL.Image.ANTIALIAS)
    print("transform--- %s seconds ---" % (time.time() - start_time))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    img.save(save_path)


def detect_landmarks(src_file, detector, predictor):
    # img = PIL.Image.open(src_file).convert('RGB')
    img = io.imread(src_file)
    dets = detector(img, 1)
    shape = predictor(img, dets[0])
    shape = face_utils.shape_to_np(shape)
    res = {'lm68': None}
    res['lm68'] = shape
    return res

def detect_landmarks_mtcnn(src_file, detector, predictor):
    img = io.imread(src_file)
    result = detector.detect_faces(img)
    if len(result)>0:
            index = 0
            if len(result)>1: # if multiple faces, take the biggest face
                size = -100000
                for r in range(len(result)):
                    size_ = result[r]["box"][2] + result[r]["box"][3]
                    if size < size_:
                        size = size_
                        index = r

    det = result[index]['box']
    left = det[0]
    top = det[1]
    right = det[2]
    bottom = det[3] 
    det = dlib.rectangle(left, top, right, bottom) 
    shape = predictor(img, det)
    shape = face_utils.shape_to_np(shape)
    res = {'lm68': None}
    res['lm68'] = shape
    return res


if __name__ == '__main__':
    # mat_data = 'coeff_debug.mat'
    # image_path = 'D:/datasets/VFHQ_new/images512x512/Clip+322wQTwVYwM+P0+C0+F263-397/00000025.png'
    # recreate_aligned_images_fast(mat_data, image_path)
    

    # img = PIL.Image.open(image_path).convert('RGB')
    # img = io.imread(image_path)
    # img = np.array(img)
    detector = dlib.get_frontal_face_detector()
    # detector_mtcnn = MTCNN()
    # p = "D:/projects/eg3d_official/dataset_preprocessing/ffhq/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat"
    # try:
    #     predictor = dlib.shape_predictor(p)
    #     det = detector(img, 1)[0]
    # except:
    #     ## replace dlib by mtcnn
    #     result = detector_mtcnn.detect_faces(img)
    #     if len(result)>0:
    #         index = 0
    #         if len(result)>1: # if multiple faces, take the biggest face
    #             size = -100000
    #             for r in range(len(result)):
    #                 size_ = result[r]["box"][2] + result[r]["box"][3]
    #                 if size < size_:
    #                     size = size_
    #                     index = r

    #     det = result[index]['box']
    #     left = det[0]
    #     top = det[1]
    #     right = det[2]
    #     bottom = det[3] 
    #     det = dlib.rectangle(left, top, right, bottom) 
    # shape = predictor(img, det)
    # shape = face_utils.shape_to_np(shape)
    # res = {'lm68': None}
    # res['lm68'] = shape
    # savemat('coeff_debug.mat', res)

    # p = "D:/projects/eg3d_official/dataset_preprocessing/ffhq/facial-landmarks-recognition/shape_predictor_68_face_landmarks.dat"
    # predictor = dlib.shape_predictor(p)
    # detector_mtcnn = MTCNN()
    # path_root = glob.glob('D:/datasets/VFHQ_n/images512x512/*')
    # with open('lm_invalid_mtcnn.txt', 'w') as f:
    #     for path in tqdm(path_root):
    #         if not os.path.isdir(path):
    #             continue
    #         image_paths = glob.glob(path + '/*.png')
    #         for image_path in image_paths:
    #             save_path = image_path.replace('images', 'landmarks').replace('png', 'mat')
    #             if os.path.isfile(save_path):
    #                     continue
    #             lms = detect_landmarks(image_path, detector, predictor)
    #             os.makedirs(os.path.dirname(save_path), exist_ok=True)
    #             savemat(save_path, lms)
    #                 # # f.write(image_path+'\n')
    #                 # raiseExceptions
    # f.close()


    # path_root = glob.glob('D:/datasets/VFHQ_new/images512x512/*')
    # with open('align_invalid_mtcnn.txt', 'w') as f:
    #     for path in path_root:
    #         if not os.path.isdir(path):
    #                 continue
    #         image_paths = glob.glob(path + '/*.png')
    #         for image_path in image_paths:
    #             mat_data = image_path.replace('images', 'landmarks').replace('png', 'mat')
    #             try:
    #                 recreate_aligned_images_fast(mat_data, image_path)
    #             except:
    #                 f.write(image_path+'\n')

    # f.close()


def visualize_landmark(image_array, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    origin_img = Image.fromarray(image_array[:, :, ::-1])
    draw = ImageDraw.Draw(origin_img)
    for i in landmarks:
        draw.point(i)
    # imshow(origin_img)
    origin_img.save("debug.png")


if __name__ == '__main__':
    # img_name = '00000000.png'
    # image_array = cv2.imread(img_name)
    # mat_data = 'keypoints_static_0000.mat'
    # # mat_data = 'coeff_debug.mat'
    # landmarks = sio.loadmat(mat_data)
    # # Parse landmarks.
    # # pylint: disable=unused-variable
    # lms = np.array(landmarks['lm68'])
    # # visualize_landmark(image_array, lms)
    # recreate_aligned_images_fast(mat_data, src_file=img_name)


    path_root = glob.glob('D:/datasets/MEAD_processed/landmarks512x512/*')[7:]
    with open('align_invalid_mead.txt', 'w') as f:
        for path in path_root:
            landmarks = glob.glob(path + '/*/*.mat' )
            for landmark in landmarks:
                image_path = landmark.replace('keypoints_static_0000.mat', 'image_0000.png')
                save_path = os.path.dirname(landmark) + '.png'
                try:
                    recreate_aligned_images_fast(landmark, src_file=image_path, save_path=save_path)
                except:
                    f.write(image_path+'\n')