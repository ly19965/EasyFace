# Copyright (c) Alibaba, Inc. and its affiliates.
from typing import TYPE_CHECKING

from modelscope.utils.import_utils import LazyImportModule

if TYPE_CHECKING:
    from .builder import build_trainer
    from .trainer import EpochBasedTrainer

else:
    _import_structure = {
        'base': ['DummyTrainer'],
        'builder': ['build_trainer'],
        'trainer': ['EpochBasedTrainer']
    }

    import sys

    sys.modules[__name__] = LazyImportModule(
        __name__,
        globals()['__file__'],
        _import_structure,
        module_spec=__spec__,
        extra_objects={},
    )
