# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.utils.constant import Fields, Tasks


class Models(object):
    """ Names for different models.

        Holds the standard model name to use for identifying different model.
    This should be used to register models.

        Model name should only contain model information but not task information.
    """
    tinynas_damoyolo = 'tinynas-damoyolo'
    # face models
    scrfd = 'scrfd'
    face_2d_keypoints = 'face-2d-keypoints'
    fer = 'fer'
    fairface = 'fairface'
    retinaface = 'retinaface'
    mogface = 'mogface'
    mtcnn = 'mtcnn'
    ulfd = 'ulfd'
    rts = 'rts'
    flir = 'flir'
    arcface = 'arcface'
    facemask = 'facemask'
    flc = 'flc'
    tinymog = 'tinymog'
    damofd = 'damofd'


class TaskModels(object):
    pass
class Heads(object):
    pass

class Pipelines(object):
    """ Names for different pipelines.

        Holds the standard pipline name to use for identifying different pipeline.
    This should be used to register pipelines.

        For pipeline which support different models and implements the common function, we
    should use task name for this pipeline.
        For pipeline which suuport only one model, we should use ${Model}-${Task} as its name.
    """
    # vision tasks
    face_2d_keypoints = 'mobilenet_face-2d-keypoints_alignment'
    salient_detection = 'u2net-salient-detection'
    salient_boudary_detection = 'res2net-salient-detection'
    camouflaged_detection = 'res2net-camouflaged-detection'
    image_demoire = 'uhdm-image-demoireing'
    image_classification = 'image-classification'
    face_detection = 'resnet-face-detection-scrfd10gkps'
    face_liveness_ir = 'manual-face-liveness-flir'
    face_liveness_rgb = 'manual-face-liveness-flir'
    face_liveness_xc = 'manual-face-liveness-flxc'
    card_detection = 'resnet-card-detection-scrfd34gkps'
    ulfd_face_detection = 'manual-face-detection-ulfd'
    tinymog_face_detection = 'manual-face-detection-tinymog'
    facial_expression_recognition = 'vgg19-facial-expression-recognition-fer'
    facial_landmark_confidence = 'manual-facial-landmark-confidence-flcm'
    face_attribute_recognition = 'resnet34-face-attribute-recognition-fairface'
    retina_face_detection = 'resnet50-face-detection-retinaface'
    mog_face_detection = 'resnet101-face-detection-cvpr22papermogface'
    mtcnn_face_detection = 'manual-face-detection-mtcnn'
    face_recognition = 'ir101-face-recognition-cfglint'
    face_recognition_ood = 'ir-face-recognition-ood-rts'
    face_quality_assessment = 'manual-face-quality-assessment-fqa'
    face_recognition_ood = 'ir-face-recognition-rts'
    face_recognition_onnx_ir = 'manual-face-recognition-frir'
    face_recognition_onnx_fm = 'manual-face-recognition-frfm'
    arc_face_recognition = 'ir50-face-recognition-arcface'
    mask_face_recognition = 'resnet-face-recognition-facemask'


DEFAULT_MODEL_FOR_PIPELINE = {
    # TaskName: (pipeline_module_name, model_repo)
    Tasks.face_detection:
    (Pipelines.mog_face_detection,
     'damo/cv_resnet101_face-detection_cvpr22papermogface'),
    Tasks.face_liveness: (Pipelines.face_liveness_ir,
                          'damo/cv_manual_face-liveness_flir'),
    Tasks.face_recognition: (Pipelines.face_recognition,
                             'damo/cv_ir101_facerecognition_cfglint'),
    Tasks.facial_expression_recognition:
    (Pipelines.facial_expression_recognition,
     'damo/cv_vgg19_facial-expression-recognition_fer'),
    Tasks.face_attribute_recognition:
    (Pipelines.face_attribute_recognition,
     'damo/cv_resnet34_face-attribute-recognition_fairface'),
    Tasks.face_2d_keypoints: (Pipelines.face_2d_keypoints,
                              'damo/cv_mobilenet_face-2d-keypoints_alignment'),
    Tasks.face_quality_assessment:
    (Pipelines.face_quality_assessment,
     'damo/cv_manual_face-quality-assessment_fqa'),
}
class CVTrainers(object):
    face_detection_scrfd = 'face-detection-scrfd'


class Trainers(CVTrainers):
    """ Names for different trainer.

        Holds the standard trainer name to use for identifying different trainer.
    This should be used to register trainers.

        For a general Trainer, you can use EpochBasedTrainer.
        For a model specific Trainer, you can use ${ModelName}-${Task}-trainer.
    """

    default = 'trainer'
    easycv = 'easycv'
    tinynas_damoyolo = 'tinynas-damoyolo'

    @staticmethod
    def get_trainer_domain(attribute_or_value):
        if attribute_or_value in vars(
                CVTrainers) or attribute_or_value in vars(CVTrainers).values():
            return Fields.cv
        elif attribute_or_value in vars(
                NLPTrainers) or attribute_or_value in vars(
                    NLPTrainers).values():
            return Fields.nlp
        elif attribute_or_value in vars(
                AudioTrainers) or attribute_or_value in vars(
                    AudioTrainers).values():
            return Fields.audio
        elif attribute_or_value in vars(
                MultiModalTrainers) or attribute_or_value in vars(
                    MultiModalTrainers).values():
            return Fields.multi_modal
        elif attribute_or_value == Trainers.default:
            return Trainers.default
        elif attribute_or_value == Trainers.easycv:
            return Trainers.easycv
        else:
            return 'unknown'


class Preprocessors(object):
    """ Names for different preprocessor.

        Holds the standard preprocessor name to use for identifying different preprocessor.
    This should be used to register preprocessors.

        For a general preprocessor, just use the function name as preprocessor name such as
    resize-image, random-crop
        For a model-specific preprocessor, use ${modelname}-${fuction}
    """

    # cv preprocessor
    load_image = 'load-image'
    image_denoise_preprocessor = 'image-denoise-preprocessor'
    image_deblur_preprocessor = 'image-deblur-preprocessor'
    object_detection_tinynas_preprocessor = 'object-detection-tinynas-preprocessor'
    image_classification_mmcv_preprocessor = 'image-classification-mmcv-preprocessor'
    image_color_enhance_preprocessor = 'image-color-enhance-preprocessor'
    image_instance_segmentation_preprocessor = 'image-instance-segmentation-preprocessor'
    image_driving_perception_preprocessor = 'image-driving-perception-preprocessor'
    image_portrait_enhancement_preprocessor = 'image-portrait-enhancement-preprocessor'
    image_quality_assessment_mos_preprocessor = 'image-quality_assessment-mos-preprocessor'
    video_summarization_preprocessor = 'video-summarization-preprocessor'
    movie_scene_segmentation_preprocessor = 'movie-scene-segmentation-preprocessor'
    image_classification_bypass_preprocessor = 'image-classification-bypass-preprocessor'
    object_detection_scrfd = 'object-detection-scrfd'



class Metrics(object):
    """ Names for different metrics.
    """

    # accuracy
    accuracy = 'accuracy'

    multi_average_precision = 'mAP'
    audio_noise_metric = 'audio-noise-metric'
    PPL = 'ppl'

    # text gen
    BLEU = 'bleu'

    # metrics for image denoise task
    image_denoise_metric = 'image-denoise-metric'
    # metrics for video frame-interpolation task
    video_frame_interpolation_metric = 'video-frame-interpolation-metric'
    # metrics for real-world video super-resolution task
    video_super_resolution_metric = 'video-super-resolution-metric'

    # metric for image instance segmentation task
    image_ins_seg_coco_metric = 'image-ins-seg-coco-metric'
    # metrics for sequence classification task
    seq_cls_metric = 'seq-cls-metric'
    # loss metric
    loss_metric = 'loss-metric'
    # metrics for token-classification task
    token_cls_metric = 'token-cls-metric'
    # metrics for text-generation task
    text_gen_metric = 'text-gen-metric'
    # file saving wrapper
    prediction_saving_wrapper = 'prediction-saving-wrapper'
    # metrics for image-color-enhance task
    image_color_enhance_metric = 'image-color-enhance-metric'
    # metrics for image-portrait-enhancement task
    image_portrait_enhancement_metric = 'image-portrait-enhancement-metric'
    video_summarization_metric = 'video-summarization-metric'
    # metric for movie-scene-segmentation task
    movie_scene_segmentation_metric = 'movie-scene-segmentation-metric'
    # metric for inpainting task
    image_inpainting_metric = 'image-inpainting-metric'
    # metric for ocr
    NED = 'ned'
    # metric for cross-modal retrieval
    inbatch_recall = 'inbatch_recall'
    # metric for referring-video-object-segmentation task
    referring_video_object_segmentation_metric = 'referring-video-object-segmentation-metric'
    # metric for video stabilization task
    video_stabilization_metric = 'video-stabilization-metric'
    # metirc for image-quality-assessment-mos task
    image_quality_assessment_mos_metric = 'image-quality-assessment-mos-metric'
    # metirc for image-quality-assessment-degradation task
    image_quality_assessment_degradation_metric = 'image-quality-assessment-degradation-metric'
    # metric for text-ranking task
    text_ranking_metric = 'text-ranking-metric'


class Optimizers(object):
    """ Names for different OPTIMIZER.

        Holds the standard optimizer name to use for identifying different optimizer.
        This should be used to register optimizer.
    """

    default = 'optimizer'

    SGD = 'SGD'


class Hooks(object):
    """ Names for different hooks.

        All kinds of hooks are defined here
    """
    # lr
    LrSchedulerHook = 'LrSchedulerHook'
    PlateauLrSchedulerHook = 'PlateauLrSchedulerHook'
    NoneLrSchedulerHook = 'NoneLrSchedulerHook'

    # optimizer
    OptimizerHook = 'OptimizerHook'
    TorchAMPOptimizerHook = 'TorchAMPOptimizerHook'
    ApexAMPOptimizerHook = 'ApexAMPOptimizerHook'
    NoneOptimizerHook = 'NoneOptimizerHook'

    # checkpoint
    CheckpointHook = 'CheckpointHook'
    BestCkptSaverHook = 'BestCkptSaverHook'
    LoadCheckpointHook = 'LoadCheckpointHook'

    # logger
    TextLoggerHook = 'TextLoggerHook'
    TensorboardHook = 'TensorboardHook'

    IterTimerHook = 'IterTimerHook'
    EvaluationHook = 'EvaluationHook'

    # Compression
    SparsityHook = 'SparsityHook'

    # CLIP logit_scale clamp
    ClipClampLogitScaleHook = 'ClipClampLogitScaleHook'

    # train
    EarlyStopHook = 'EarlyStopHook'
    DeepspeedHook = 'DeepspeedHook'


class LR_Schedulers(object):
    """learning rate scheduler is defined here

    """
    LinearWarmup = 'LinearWarmup'
    ConstantWarmup = 'ConstantWarmup'
    ExponentialWarmup = 'ExponentialWarmup'


class Datasets(object):
    """ Names for different datasets.
    """
    ClsDataset = 'ClsDataset'
    Face2dKeypointsDataset = 'FaceKeypointDataset'
    HandCocoWholeBodyDataset = 'HandCocoWholeBodyDataset'
    HumanWholeBodyKeypointDataset = 'WholeBodyCocoTopDownDataset'
    SegDataset = 'SegDataset'
    DetDataset = 'DetDataset'
    DetImagesMixDataset = 'DetImagesMixDataset'
    PanopticDataset = 'PanopticDataset'
    PairedDataset = 'PairedDataset'
