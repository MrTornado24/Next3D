# SPDX-FileCopyrightText: Copyright (c) 2021-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.

"""Streaming images and labels from datasets created with dataset_tool.py."""

from imp import load_module
import os
from unittest import skip
import numpy as np
import zipfile
import PIL.Image
import json
import torch
import dnnlib
from plyfile import PlyData, PlyElement
import pandas as pd
import math
import random
from typing import List, Dict, Tuple

from dataset_tool import sample_frames

NUMPY_INTEGER_TYPES = [np.int8, np.int16, np.int32, np.int64, np.uint8, np.uint16, np.uint32, np.uint64]
NUMPY_FLOAT_TYPES = [np.float16, np.float32, np.float64, np.single, np.double]


try:
    import pyspng
except ImportError:
    pyspng = None

#----------------------------------------------------------------------------

class Dataset(torch.utils.data.Dataset):
    def __init__(self,
        name,                   # Name of the dataset.
        raw_shape,              # Shape of the raw image data (NCHW).
        max_size    = None,     # Artificially limit the size of the dataset. None = no limit. Applied before xflip.
        use_labels  = False,    # Enable conditioning labels? False = label dimension is zero.
        xflip       = False,    # Artificially double the size of the dataset via x-flips. Applied after max_size.
        load_obj    = True,
        random_seed = 0,        # Random seed to use when applying max_size.
    ):
        self._name = name
        self._raw_shape = list(raw_shape)
        self._use_labels = use_labels
        self._raw_labels = None
        self._label_shape = None
        self.load_obj = load_obj
        
        # Apply max_size.
        self._raw_idx = np.arange(self._raw_shape[0], dtype=np.int64)
        if (max_size is not None) and (self._raw_idx.size > max_size):
            np.random.RandomState(random_seed).shuffle(self._raw_idx)
            self._raw_idx = np.sort(self._raw_idx[:max_size])

        # Apply xflip.
        self._xflip = np.zeros(self._raw_idx.size, dtype=np.uint8)
        if xflip:
            self._raw_idx = np.tile(self._raw_idx, 2)
            self._xflip = np.concatenate([self._xflip, np.ones_like(self._xflip)])

    def _get_raw_labels(self):
        if self._raw_labels is None:
            self._raw_labels = self._load_raw_labels() if self._use_labels else None
            if self._raw_labels is None:
                self._raw_labels = np.zeros([self._raw_shape[0], 0], dtype=np.float32)
            assert isinstance(self._raw_labels, np.ndarray)
            assert self._raw_labels.shape[0] == self._raw_shape[0]
            assert self._raw_labels.dtype in [np.float32, np.int64]
            if self._raw_labels.dtype == np.int64:
                assert self._raw_labels.ndim == 1
                assert np.all(self._raw_labels >= 0)
            self._raw_labels_std = self._raw_labels.std(0)
        return self._raw_labels

    def close(self): # to be overridden by subclass
        pass

    def _load_raw_image(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_verts_ply(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_verts_obj(self, raw_idx): # to be overridden by subclass
        raise NotImplementedError

    def _load_raw_labels(self): # to be overridden by subclass
        raise NotImplementedError

    def __getstate__(self):
        return dict(self.__dict__, _raw_labels=None)

    def __del__(self):
        try:
            self.close()
        except:
            pass

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def __len__(self):
        return self._raw_idx.size

    def __getitem__(self, idx):
        image = self._load_raw_image(self._raw_idx[idx], resolution=self.resolution)
        if self.load_obj:
            verts = self._load_raw_verts_obj(self._raw_idx[idx])
        if self.load_lms:
            lms = self._load_raw_lms(self._raw_idx[idx])
            verts = np.concatenate((verts, lms), 0)
        assert isinstance(image, np.ndarray)
        assert list(image.shape) == self.image_shape
        assert image.dtype == np.uint8
        if self._xflip[idx]:
            assert image.ndim == 3 # CHW
            image = image[:, :, ::-1]
        if self.load_obj:
            return image.copy(), self.get_label(idx), verts.copy()
        else:
            return image.copy(), self.get_label(idx)

    def get_label(self, idx):
        label = self._get_raw_labels()[self._raw_idx[idx]]
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_vert(self, idx):
        if self.mesh_type == '.obj':
            verts = self._load_raw_verts_obj(self._raw_idx[idx])
        elif self.mesh_type == '.ply':
            verts = self._load_raw_verts_ply(self._raw_idx[idx])
        else:
            raise NotImplementedError
        if self.load_lms:
            lms = self._load_raw_lms(self._raw_idx[idx])
            verts = np.concatenate((verts, lms), 0)
        return verts.copy()


    def get_details(self, idx):
        d = dnnlib.EasyDict()
        d.raw_idx = int(self._raw_idx[idx])
        d.xflip = (int(self._xflip[idx]) != 0)
        d.raw_label = self._get_raw_labels()[d.raw_idx].copy()
        return d

    def get_label_std(self):
        return self._raw_labels_std

    @property
    def name(self):
        return self._name

    @property
    def image_shape(self):
        return list(self._raw_shape[1:])

    @property
    def num_channels(self):
        assert len(self.image_shape) == 3 # CHW
        return self.image_shape[0]

    @property
    def resolution(self):
        assert len(self.image_shape) == 3 # CHW
        assert self.image_shape[1] == self.image_shape[2]
        return self.image_shape[1]

    @property
    def label_shape(self):
        if self._label_shape is None:
            raw_labels = self._get_raw_labels()
            if raw_labels.dtype == np.int64:
                self._label_shape = [int(np.max(raw_labels)) + 1]
            else:
                self._label_shape = raw_labels.shape[1:]
        return list(self._label_shape)

    @property
    def label_dim(self):
        return self.label_shape[-1]

    @property
    def gen_label_dim(self):
            return 25 # 25 for camera params only

    @property
    def has_labels(self):
        return any(x != 0 for x in self.label_shape)

    @property
    def has_onehot_labels(self):
        return self._get_raw_labels().dtype == np.int64

#----------------------------------------------------------------------------

class ImageFolderDataset(Dataset):
    def __init__(self,
        path,                   # Path to directory or zip.
        mesh_path       = None, # Path to mesh.
        mesh_type       = '.obj',
        resolution      = None, # Ensure specific resolution, None = highest available.
        load_exp        = False,
        load_lms        = False,
        **super_kwargs,         # Additional arguments for the Dataset base class.
    ):
        self._path = path
        self._mesh_path = mesh_path
        self.mesh_type = mesh_type
        self._zipfile = None
        self.load_exp = load_exp
        self.load_lms = load_lms

        if os.path.isdir(self._path):
            self._type = 'dir'
            self._all_fnames = {os.path.relpath(os.path.join(root, fname), start=self._path) for root, _dirs, files in os.walk(self._path) for fname in files}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        if os.path.isdir(self._mesh_path):
            self._type = 'dir'
            self._all_mesh_fnames = {os.path.relpath(os.path.join(root, fname), start=self._mesh_path) for root, _dirs, files in os.walk(self._mesh_path) for fname in files}
        elif self._file_ext(self._mesh_path) == '.zip':
            self._type = 'zip'
            self._all_mesh_fnames = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must point to a directory or zip')

        PIL.Image.init()
        self._image_fnames = sorted(fname for fname in self._all_fnames if self._file_ext(fname) in PIL.Image.EXTENSION)[:139900]
        self._mesh_fnames = [fname.replace('png', 'obj') for fname in self._image_fnames]
        if len(self._image_fnames) == 0:
            raise IOError('No image files found in the specified path')
        
        self.vertmean = np.zeros((5023,3)) # 5023
        self.vertstd = 1.0

        name = os.path.splitext(os.path.basename(self._path))[0]
        raw_shape = [len(self._image_fnames)] + list(self._load_raw_image(0, resolution).shape)
        if resolution is not None and (raw_shape[2] != resolution or raw_shape[3] != resolution):
            raise IOError('Image files do not match the specified resolution')
        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    @staticmethod
    def _file_ext(fname):
        return os.path.splitext(fname)[1].lower()

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname, path=None):
        if not path:
            path = self._path
        if self._type == 'dir':
            return open(os.path.join(path, fname), 'rb')
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_image(self, raw_idx, resolution=None):
        fname = self._image_fnames[raw_idx]
        with self._open_file(fname) as f:
            image = PIL.Image.open(f)
            if resolution:
                image = image.resize((resolution, resolution))
            image = np.array(image)
        if image.ndim == 2:
            image = image[:, :, np.newaxis] # HW => HWC
        image = image.transpose(2, 0, 1) # HWC => CHW
        return image

    def _load_raw_labels(self):
        ### debug: load exp labels and camera labels
        fname = 'dataset_mead.json'
        fname_exp = 'dataset_exp.json'
        if fname not in self._all_fnames:
            return None
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if self.load_exp and fname_exp in self._all_fnames:
            with self._open_file(fname_exp) as f:
                exps = json.load(f)['labels']
        else:
            exps = None
        if labels is None:
            return None
        labels = dict(labels)
        labels = [labels[fname.replace('\\', '/')] for fname in self._image_fnames]
        labels = np.array(labels)
        labels = labels.astype({1: np.int64, 2: np.float32}[labels.ndim])

        if exps is not None:
            exps = dict(exps)
            exps = [exps[fname] for fname in self._image_fnames]
            exps = np.array(exps)
            exps = exps.astype({1: np.int64, 2: np.float32}[exps.ndim])
            labels = np.concatenate((labels, exps), 1)

        return labels

    def _load_raw_verts_ply(self, raw_idx):
        fname = self._mesh_fnames[raw_idx]
        plydata = PlyData.read(os.path.join(self._mesh_path, fname))
        data = plydata.elements[0].data  # 读取数据
        data_pd = pd.DataFrame(data)  # 转换成DataFrame, 因为DataFrame可以解析结构化的数据
        data_np = np.zeros(data_pd.shape, dtype=np.float)  # 初始化储存数据的array
        property_names = data[0].dtype.names  # 读取property的名字
        for i, name in enumerate(property_names):  # 按property读取数据，这样可以保证读出的数据是同样的数据类型。
            data_np[:, i] = data_pd[name]
        v = data_np[:,:3].astype(np.float32)
        # v = v *1000. + np.array([[0., 0., 1000.]], dtype=np.float32)
        verts = verts.reshape((-1, 3))
        # rotate and scale
        scale, rot = 2.6, 0.
        rotate_matrix = np.array([[ math.cos(rot),  0,  math.sin(rot)],
                             [ 0,              1,              0],
                             [-math.sin(rot),  0,  math.cos(rot)]])
        
        rotate_matrix = np.array([[ 1,  0,  0],
                             [ 0,  math.cos(rot), -math.sin(rot)],
                             [ 0,  math.sin(rot),  math.cos(rot)]])

        v = np.matmul(v*scale, rotate_matrix.T)
        return v

    def _load_raw_verts_obj(self, raw_idx):
        fname = self._mesh_fnames[raw_idx]
        v = []

        with open(os.path.join(self._mesh_path, fname), "r") as f:
            while True:
                line = f.readline()

                if line == "":
                    break

                if line[:2] == "v ":
                    v.append([float(x) for x in line.split()[1:]])
        
        v = np.array(v).reshape((-1, 3))
        return v.astype(np.float32)

    def _load_raw_lms(self, raw_idx):
        fname = self._mesh_fnames[raw_idx]
        lms = np.loadtxt(os.path.join(self._mesh_path, fname).replace('meshes', 'lms').replace('obj', 'txt'))
        return lms.astype(np.float32)    

class VideoFramesFolderDataset(Dataset):
    def __init__(self,
        path,                                           # Path to directory or zip.
        mesh_path,              
        mesh_type= '.obj',
        resolution=None,                                # Unused arg for backward compatibility
        load_exp=False,         
        load_n_consecutive: int=None,                   # Should we load first N frames for each video?
        load_n_consecutive_random_offset: bool=True,    # Should we use a random offset when loading consecutive frames?
        max_num_frames: int=1024,                       
        subsample_factor: int=1,                        # Sampling factor, i.e. decreasing the temporal resolution
        discard_short_videos: bool=False,               # Should we discard videos that are shorter than `load_n_consecutive`?
        sampling_dict = {},                             # Video sampling configs.
        **super_kwargs,                                 # Additional arguments for the Dataset base class.
    ):  
        self._mesh_path = mesh_path
        self.mesh_type = mesh_type
        self.load_exp = load_exp
        self.sampling_dict = sampling_dict
        self.max_num_frames = max_num_frames
        self._path = path
        self._zipfile = None
        self.load_n_consecutive = load_n_consecutive
        self.load_n_consecutive_random_offset = load_n_consecutive_random_offset
        self.subsample_factor = subsample_factor
        self.discard_short_videos = discard_short_videos

        if self.subsample_factor > 1 and self.load_n_consecutive is None:
            raise NotImplementedError("Can do subsampling only when loading consecutive frames.")

        listdir_full_paths = lambda d: sorted([os.path.join(d, x) for x in os.listdir(d)])
        name = os.path.splitext(os.path.basename(self._path))[0]

        if os.path.isdir(self._path):
            self._type = 'dir'
            # We assume that the depth is 2
            self._all_objects = {o for d in listdir_full_paths(self._path) for o in (([d] + listdir_full_paths(d)) if os.path.isdir(d) else [d])}
            self._all_objects = {os.path.relpath(o, start=self._path) for o in {self._path}.union(self._all_objects)}
        elif self._file_ext(self._path) == '.zip':
            self._type = 'zip'
            self._all_objects = set(self._get_zipfile().namelist())
        else:
            raise IOError('Path must be either a directory or point to a zip archive')

        PIL.Image.init()
        self._video_dir2frames = {}
        objects = sorted([d for d in self._all_objects])
        root_path_depth = len(os.path.normpath(objects[0]).split(os.path.sep))
        curr_d = objects[0] # Root path is the first element
        curr_frames = 0 # limit frames number for each video clip
        for o in objects:
            curr_obj_depth = len(os.path.normpath(o).split(os.path.sep))

            if self._file_ext(o) in PIL.Image.EXTENSION:
                if curr_frames >= 100:
                    continue
                assert o.startswith(curr_d), f"Object {o} is out of sync. It should lie inside {curr_d}"
                assert curr_obj_depth == root_path_depth + 1, "Frame images should be inside directories"
                if not curr_d in self._video_dir2frames:
                    self._video_dir2frames[curr_d] = []
                self._video_dir2frames[curr_d].append(o)
                curr_frames += 1
            elif self._file_ext(o) == 'json':
                assert curr_obj_depth == root_path_depth + 1, "Classes info file should be inside the root dir"
                pass
            else:
                # We encountered a new directory
                assert curr_obj_depth == root_path_depth, f"Video directories should be inside the root dir. {o} is not."
                if curr_d in self._video_dir2frames:
                    sorted_files = sorted(self._video_dir2frames[curr_d])
                    self._video_dir2frames[curr_d] = sorted_files
                curr_d = o
                curr_frames = 0
        for video_name, video in  self._video_dir2frames.items():
            assert len(video) == 100
        if self.discard_short_videos:
            self._video_dir2frames = {d: fs for d, fs in self._video_dir2frames.items() if len(fs) >= self.load_n_consecutive * self.subsample_factor} # {'video_0': ['video_0/img_0.png', 'video_0/img_1.png']}

        self._video_idx2frames = [frames for frames in self._video_dir2frames.values()]

        if len(self._video_idx2frames) == 0:
            raise IOError('No videos found in the specified archive')

        raw_shape = [len(self._video_idx2frames)] + list(self._load_raw_frames(0, [0])[0][0].shape)

        super().__init__(name=name, raw_shape=raw_shape, **super_kwargs)

    def _get_zipfile(self):
        assert self._type == 'zip'
        if self._zipfile is None:
            self._zipfile = zipfile.ZipFile(self._path)
        return self._zipfile

    def _open_file(self, fname, type='rb'):
        if self._type == 'dir':
            return open(os.path.join(self._path, fname), type)
        if self._type == 'zip':
            return self._get_zipfile().open(fname, 'r')
        return None

    def close(self):
        try:
            if self._zipfile is not None:
                self._zipfile.close()
        finally:
            self._zipfile = None

    def __getstate__(self):
        return dict(super().__getstate__(), _zipfile=None)

    def _load_raw_labels(self):
        """
        We leave the `dataset.json` file in the same format as in the original SG2-ADA repo:
        it's `labels` field is a hashmap of filename-label pairs.
        """
        ### debug: load exp labels and camera labels
        fname = 'dataset.json'
        with self._open_file(fname) as f:
            labels = json.load(f)['labels']
        if labels is None:
            return None
        labels = dict(labels)
        _labels = []
        for video_name, video in self._video_dir2frames.items():
            _label = [labels[fname] for fname in video]
            _labels.append(_label)
        labels = np.array(_labels)
        labels = labels.astype({1: np.int64, 2: np.int64, 3: np.float32}[labels.ndim])

        return labels

    def __getitem__(self, idx: int) -> Dict:
        if self.load_n_consecutive:
            num_frames_available = len(self._video_idx2frames[self._raw_idx[idx]])
            assert num_frames_available - self.load_n_consecutive * self.subsample_factor >= 0, f"We have only {num_frames_available} frames available, cannot load {self.load_n_consecutive} frames."

            if self.load_n_consecutive_random_offset:
                random_offset = random.randint(0, num_frames_available - self.load_n_consecutive * self.subsample_factor + self.subsample_factor - 1)
            else:
                random_offset = 0
            frames_idx = np.arange(0, self.load_n_consecutive * self.subsample_factor, self.subsample_factor) + random_offset
        else:
            frames_idx = None

        frames, verts, frames_idx = self._load_raw_frames(self._raw_idx[idx], frames_idx=frames_idx)

        assert isinstance(frames, np.ndarray)
        assert list(frames[0].shape) == self.image_shape
        assert frames.dtype == np.uint8

        if self._xflip[idx]:
            assert frames.ndim == 4 # TCHW
            frames = frames[:, :, :, ::-1]

        return frames.copy(), np.array(self.get_label(idx, frames_idx)), verts.copy() # self.get_video_len(idx),

    def get_video_len(self, idx: int) -> int:
        return min(self.max_num_frames, len(self._video_idx2frames[self._raw_idx[idx]]))

    def get_label(self, idx, frames_idx): # rewrite this method for video dataset
        labels = self._get_raw_labels()[self._raw_idx[idx]]
        label = []
        for frame_idx in frames_idx:
            label.append(labels[frame_idx])
        label = np.array(label)
        if label.dtype == np.int64:
            onehot = np.zeros(self.label_shape, dtype=np.float32)
            onehot[label] = 1
            label = onehot
        return label.copy()

    def get_vert(self, idx, frames_idx): # rewrite this method for video dataset
        frame_paths = self._video_idx2frames[idx]
        verts = []
        for frame_idx in frames_idx:
            vert_path = frame_paths[frame_idx].replace('videos', 'meshes').replace('png', 'obj')
            vert_path = os.path.join(self._mesh_path, vert_path)
            verts.append(load_vert_from_buffer(vert_path))
        return np.array(verts).copy()
            

    def _load_raw_frames(self, raw_idx: int, frames_idx: List[int]=None):
        frame_paths = self._video_idx2frames[raw_idx]
        total_len = len(frame_paths)
        offset = 0
        images = []
        verts = []

        if frames_idx is None:
            assert not self.sampling_dict is None, f"The dataset was created without `cfg.sampling` config and cannot sample frames on its own."
            if total_len > self.max_num_frames:
                offset = random.randint(0, total_len - self.max_num_frames)
            frames_idx = sample_frames(self.sampling_dict, total_video_len=min(total_len, self.max_num_frames)) + offset
        else:
            frames_idx = np.array(frames_idx)

        for frame_idx in frames_idx:
            with self._open_file(frame_paths[frame_idx]) as f:
                images.append(load_image_from_buffer(f))
            vert_path = frame_paths[frame_idx].replace('png', 'obj')
            vert_path = os.path.join(self._mesh_path, vert_path)
            verts.append(load_vert_from_buffer(vert_path))

        return np.array(images), np.array(verts), frames_idx

    def compute_max_num_frames(self) -> int:
        return max(len(frames) for frames in self._video_idx2frames)

#----------------------------------------------------------------------------

def load_image_from_buffer(f, use_pyspng: bool=False) -> np.ndarray:
    if use_pyspng:
        image = pyspng.load(f.read())
    else:
        image = np.array(PIL.Image.open(f))
    if image.ndim == 2:
        image = image[:, :, np.newaxis] # HW => HWC
    image = image.transpose(2, 0, 1) # HWC => CHW

    return image

#----------------------------------------------------------------------------

def load_vert_from_buffer(fname):
        v = []
        with open(fname, "r") as f:
            while True:
                line = f.readline()

                if line == "":
                    break

                if line[:2] == "v ":
                    v.append([float(x) for x in line.split()[1:]])
    
        v = np.array(v).reshape((-1, 3))
        return v.astype(np.float32)

#----------------------------------------------------------------------------

def video_to_image_dataset_kwargs(video_dataset_kwargs: dnnlib.EasyDict) -> dnnlib.EasyDict:
    """Converts video dataset kwargs to image dataset kwargs"""
    return dnnlib.EasyDict(
        class_name='training.dataset.ImageFolderDataset',
        path=video_dataset_kwargs.path,
        use_labels=video_dataset_kwargs.use_labels,
        xflip=video_dataset_kwargs.xflip,
        resolution=video_dataset_kwargs.resolution,
        random_seed=video_dataset_kwargs.get('random_seed'),
        # Explicitly ignoring the max size, since we are now interested
        # in the number of images instead of the number of videos
        # max_size=video_dataset_kwargs.max_size,
    )

#----------------------------------------------------------------------------

def remove_root(fname: os.PathLike, root_name: os.PathLike):
    """`root_name` should NOT start with '/'"""
    if fname == root_name or fname == ('/' + root_name):
        return ''
    elif fname.startswith(root_name + '/'):
        return fname[len(root_name) + 1:]
    elif fname.startswith('/' + root_name + '/'):
        return fname[len(root_name) + 2:]
    else:
        return fname

#----------------------------------------------------------------------------
