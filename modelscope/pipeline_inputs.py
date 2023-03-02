# Copyright (c) Alibaba, Inc. and its affiliates.

import numpy as np
from PIL import Image

from modelscope.utils.constant import Tasks


class InputKeys(object):
    IMAGE = 'image'
    TEXT = 'text'
    VIDEO = 'video'


class InputType(object):
    IMAGE = 'image'
    TEXT = 'text'
    AUDIO = 'audio'
    VIDEO = 'video'
    BOX = 'box'
    DICT = 'dict'
    LIST = 'list'
    INT = 'int'


INPUT_TYPE = {
    InputType.IMAGE: (str, np.ndarray, Image.Image),
    InputType.TEXT: str,
    InputType.AUDIO: (str, bytes, np.ndarray),
    InputType.VIDEO: (str, np.ndarray, 'cv2.VideoCapture'),
    InputType.BOX: (list, np.ndarray),
    InputType.DICT: (dict, type(None)),
    InputType.LIST: (list, type(None)),
    InputType.INT: int,
}


def check_input_type(input_type, input):
    expected_type = INPUT_TYPE[input_type]
    if input_type == InputType.VIDEO:
        # special type checking using class name, to avoid introduction of opencv dependency into fundamental framework.
        assert type(input).__name__ == 'VideoCapture' or isinstance(input, expected_type),\
            f'invalid input type for {input_type}, expected {expected_type} but got {type(input)}\n {input}'
    else:
        assert isinstance(input, expected_type), \
            f'invalid input type for {input_type}, expected {expected_type} but got {type(input)}\n {input}'


TASK_INPUTS = {
    # if task input is single var, value is  InputType
    # if task input is a tuple,  value is tuple of InputType
    # if task input is a dict, value is a dict of InputType, where key
    # equals the one needed in pipeline input dict
    # if task input is a list, value is a set of input format, in which
    # each element corresponds to one input format as described above.
    # ============ face tasks ===================
    Tasks.face_2d_keypoints:
    InputType.IMAGE,
    Tasks.face_detection:
    InputType.IMAGE,
    Tasks.facial_expression_recognition:
    InputType.IMAGE,
    Tasks.face_attribute_recognition:
    InputType.IMAGE,
    Tasks.face_recognition:
    InputType.IMAGE
}
