from __future__ import annotations
from collections import deque
from statistics import mean
from enum import Enum
import struct
from typing import Tuple
import datetime
import uuid
import numpy as np
from timecode import Timecode


class PyFaceVerse:
    """PyLiveLinkFace class

    Can be used to receive PyLiveLinkFace from the PyLiveLinkFace IPhone app or
    other PyLiveLinkFace compatible programs like this library.
    """

    def __init__(self, name: str = "Python_LiveLinkFace", 
                        uuid: str = str(uuid.uuid1()), fps=60, 
                        filter_size: int = 3) -> None:

        # properties
        self.uuid = uuid
        self.name = name
        self.fps = fps
        self._filter_size = filter_size

        self._version = 6
        now = datetime.datetime.now()
        timcode = Timecode(self._fps, f'{now.hour}:{now.minute}:{now.second}:{now.microsecond * 0.001}')
        self._frames = timcode.frames
        self._sub_frame = 1056060032                # I don't know how to calculate this
        self._denominator = int(self._fps / 60)     # 1 most of the time
        self._blend_shapes = [0.000] * 174
        self._old_blend_shapes = []                 # used for filtering
        for i in range(174):
            self._old_blend_shapes.append(deque([0.0], maxlen = self._filter_size))

    @property
    def uuid(self) -> str:
        return self._uuid

    @uuid.setter
    def uuid(self, value: str) -> None:
        # uuid needs to start with a $, if it doesn't add it
        if not value.startswith("$"):
            self._uuid = '$' + value
        else:
            self._uuid = value

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def fps(self) -> int:
        return self._fps

    @fps.setter
    def fps(self, value: int) -> None:
        if value < 1:
            raise ValueError("Only fps values greater than 1 are allowed.")
        self._fps = value

    def encode(self) -> bytes:
        """ Encodes the PyLiveLinkFace object into a bytes object so it can be 
        send over a network. """              
        
        version_packed = struct.pack('<I', self._version)
        uuiid_packed = bytes(self._uuid, 'utf-8')
        name_lenght_packed = struct.pack('!i', len(self._name))
        name_packed = bytes(self._name, 'utf-8')

        now = datetime.datetime.now()
        timcode = Timecode(
            self._fps, f'{now.hour}:{now.minute}:{now.second}:{now.microsecond * 0.001}')
        frames_packed = struct.pack("!II", timcode.frames, self._sub_frame)  
        frame_rate_packed = struct.pack("!II", self._fps, self._denominator)
        data_packed = struct.pack('!B174f', 174, *self._blend_shapes)
        
        return version_packed + uuiid_packed + name_lenght_packed + name_packed + \
            frames_packed + frame_rate_packed + data_packed

    def set_blendshape(self, index: int, value: float, 
                        no_filter: bool = False) -> None:
        """ Sets the value of the blendshape. 
        
        The function will use mean to filter between the old and the new 
        values, unless `no_filter` is set to True.

        Parameters
        ----------
        index : int
            Index of the BlendShape to get the value from.
        value: float
            Value to set the BlendShape to, should be in the range of 0 - 1 for 
            the blendshapes and between -1 and 1 for the head rotation 
            (yaw, pitch, roll).
        no_filter: bool
            If set to True, the blendshape will be set to the value without 
            filtering.
        
        Returns
        ----------
        None
        """

        if no_filter:
            self._blend_shapes[index] = float(value)
        else:
            self._old_blend_shapes[index].append(float(value))
            filterd_value = mean(self._old_blend_shapes[index])
            self._blend_shapes[index] = filterd_value

