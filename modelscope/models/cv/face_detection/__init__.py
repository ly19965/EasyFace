# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .mogface import MogFaceDetector
    from .mtcnn import MtcnnFaceDetector
    from .retinaface import RetinaFaceDetection
    from .ulfd_slim import UlfdFaceDetector
    from .scrfd import ScrfdDetect
    from .scrfd import TinyMogDetect
    from .scrfd import SCRFDPreprocessor
    from .scrfd import DamoFdDetect
else:
    _import_structure = {
        'ulfd_slim': ['UlfdFaceDetector'],
        'retinaface': ['RetinaFaceDetection'],
        'mtcnn': ['MtcnnFaceDetector'],
        'mogface': ['MogFaceDetector'],
        'scrfd': ['TinyMogDetect', 'ScrfdDetect', 'SCRFDPreprocessor', 'DamoFdDetect'],
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
