# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .base import Preprocessor
    from .builder import PREPROCESSORS, build_preprocessor
    from .common import Compose, ToTensor, Filter
    from .image import (LoadImage, load_image,
                        ImageColorEnhanceFinetunePreprocessor,
                        ImageInstanceSegmentationPreprocessor,
                        ImageDenoisePreprocessor, ImageDeblurPreprocessor)
else:
    _import_structure = {
        'base': ['Preprocessor'],
        'builder': ['PREPROCESSORS', 'build_preprocessor'],
        'common': ['Compose', 'ToTensor', 'Filter'],
        'image': [
            'LoadImage', 'load_image', 'ImageColorEnhanceFinetunePreprocessor',
            'ImageInstanceSegmentationPreprocessor',
            'ImageDenoisePreprocessor', 'ImageDeblurPreprocessor'
        ]
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
