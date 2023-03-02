<div align="center">
  <img src="demo/modelscope.gif" width="40%" height="40%" />
</div>

<div align="center">

<!-- [![Documentation Status](https://readthedocs.org/projects/easy-cv/badge/?version=latest)](https://easy-cv.readthedocs.io/en/latest/) -->
[![license](https://img.shields.io/github/license/modelscope/modelscope.svg)](https://github.com/modelscope/modelscope/blob/master/LICENSE)
</div>


<h4 align="center">
    <a href=#EasyFace> 特性 </a> |
    <a href=#安装> 安装 </a> |
    <a href=#单模型推理> 单模型推理</a> | 
    <a href=#单模型训练和微调> 单模型训练/微调</a> |
    <a href=#单模型选型和对比> 单模型选型/对比</a>  
    <!--- <a href=#人脸识别系统多模块一键选型/对比> 人脸识别系统多模块一键选型/对比</a> -->
</h4>

## EasyFace

**EasyFace**旨在快速选型/了解/对比/体验人脸相关sota模型，依托于[**Modelscope**](https://modelscope.cn/home)开发库和[**Pytorch**](https://pytorch.org)框架，EasyFace具有以下特性:
- 快速体验/对比/选型Sota的人脸相关模型, 涉及人脸检测，人脸识别，人脸关键点，人脸表情识别，人脸活体检测等领域，目前支持人脸检测相关sota模型。
- 5行代码即可进行模型推理，10行代码进行模型训练/Finetune, 20行代码对比不同模型在自建/公开数据集上的精度以及可视化结果。
- 基于现有模型快速搭建[**创空间**](https://modelscope.cn/studios/damo/face_album/summary)应用。

## News 📢

<!--- 🔥 **`2023-03-20`**：新增DamoFR人脸识别模型，基于Vit Backbone 围绕data-centric以及patch-level hard example  mining策略重新设计了Transformer-based Small/Medium/Large 人脸识别backbone，效果sota，已release不同算力下的sota人脸识别，口罩人脸识别DamoFR模型，[**paper**]() and [**project**]()；-->

🔥 **`2023-03-10`**：新增DamoFD（ICLR23）人脸检测关键点模型，基于SCRFD框架进一步搜索了FD-friendly backbone结构。 在0.5/2.5/10/34 GFlops VGA分辨率的算力约束条件下性能均超过SCRFD。其中提出的轻量级的检测器DDSAR-0.5G在VGA分辨率0.5GFlops条件下WiderFace上hard集精度为71.03(超过SCRFD 2.5个点)，欢迎大家一键使用(支持训练和推理)，[**paper**](https://openreview.net/forum?id=NkJOhtNKX91)。

🔥  **`2023-03-10`**：新增4个人脸检测模型，包括DamoFD，MogFace，RetinaFace，Mtcnn。

## 支持模型列表
**`对应模型的推理和训练单元测试放在face_project目录下`**

### 推理

🔥 **`人脸检测`**
- DamoFD，MogFace，RetinaFace，Mtcnn。
- ['damo/cv_ddsar_face-detection_iclr23-damofd', 'damo/cv_resnet101_face-detection_cvpr22papermogface',  'damo/cv_resnet50_face-detection_retinaface', 'damo/cv_manual_face-detection_mtcnn']

### 训练
🔥 **`人脸检测`**
- DamoFD
- ['damo/cv_ddsar_face-detection_iclr23-damofd']

## 安装
```
conda create --offline -n  EasyFace python=3.8
conda activate EasyFace
# pytorch >= 1.3.0
pip install torch==1.8.1+cu102  torchvision==0.9.1+cu102  --extra-index-url https://download.pytorch.org/whl/cu102
git clone https://github.com/ly19965/FaceMaas
cd FaceMaas
pip install -r requirements.txt
mim install mmcv-full
```

## 单模型推理
从支持推理的模型列表里选择想体验的模型, e.g.人脸检测模型DamoFD_0.5g

### 单张图片推理
```python
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import  Tasks

face_detection = pipeline(task=Tasks.face_detection, model='damo/cv_ddsar_face-detection_iclr23-damofd')
# 支持 url image and abs dir image path
img_path = 'https://modelscope.oss-cn-beijing.aliyuncs.com/test/images/face_detection2.jpeg' 
result = face_detection(img_path)

# 提供可视化结果
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.preprocessors.image import LoadImage
img = LoadImage.convert_to_ndarray(img_path)
cv2.imwrite('srcImg.jpg', img)
img_draw = draw_face_detection_result('srcImg.jpg', result)
import matplotlib.pyplot as plt
plt.imshow(img_draw)
```

### Mini公开数据集推理
```python
import os.path as osp
import cv2
import os
import numpy as np
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import voc_ap, image_eval,img_pr_info, gen_gt_info, dataset_pr_info, bbox_overlap

model_id = 'damo/cv_ddsar_face-detection_iclr23-damofd'
val_set = MsDataset.load('widerface_mini_train_val', namespace='ly261666', split='validation')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)
img_base_path = next(iter(val_set))[1]
img_dir = osp.join(img_base_path, 'val_data')
img_gt = osp.join(img_base_path, 'val_label.txt')
gt_info = gen_gt_info(img_gt)
pred_info = {}
iou_th = 0.5
thresh_num = 1000
face_detection_func = pipeline(Tasks.face_detection, model=model_id)
count_face = 0
pr_curve = np.zeros((thresh_num, 2)).astype('float')
for img_name in os.listdir(img_dir):
    abs_img_name = osp.join(img_dir, img_name)
    result = face_detection_func(abs_img_name)
    pred_info = np.concatenate([result['boxes'], np.array(result['scores'])[:,np.newaxis]], axis=1)
    gt_box = np.array(gt_info[img_name])
    pred_recall, proposal_list = image_eval(pred_info, gt_box, iou_th)
    _img_pr_info, fp = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
    pr_curve += _img_pr_info
    count_face += gt_box.shape[0]
    
pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
propose = pr_curve[:, 0]
recall = pr_curve[:, 1]
for srecall in np.arange(0.1, 1.0001, 0.1):
    rindex = len(np.where(recall<=srecall)[0])-1
    rthresh = 1.0 - float(rindex)/thresh_num
    print('Recall-Precision-Thresh:', recall[rindex], propose[rindex], rthresh)
ap = voc_ap(recall, propose)
print('ap: %.5f, iou_th: %.2f'%(ap, iou_th))
```

## 单模型训练和微调
从支持训练的模型列表里选择想体验的模型, e.g.人脸检测模型DamoFD_0.5g

### 训练

```python
import os
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.hub.snapshot_download import snapshot_download

model_id = 'damo/cv_ddsar_face-detection_iclr23-damofd'
ms_ds_widerface = MsDataset.load('WIDER_FACE_mini', namespace='shaoxuan')  # remove '_mini' for full dataset

data_path = ms_ds_widerface.config_kwargs['split_config']
train_dir = data_path['train']
val_dir = data_path['validation']

def get_name(dir_name):
    names = [i for i in os.listdir(dir_name) if not i.startswith('_')]
    return names[0]

train_root = train_dir + '/' + get_name(train_dir) + '/'
val_root = val_dir + '/' + get_name(val_dir) + '/'
cache_path = snapshot_download(model_id)
tmp_dir = tempfile.TemporaryDirectory().name
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

def _cfg_modify_fn(cfg):
    cfg.checkpoint_config.interval = 1
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 1
    cfg.data.workers_per_gpu = 1
    cfg.data.samples_per_gpu = 4
    return cfg

kwargs = dict(
    cfg_file=os.path.join(cache_path, 'DamoFD_lms.py'),
    work_dir=tmp_dir,
    train_root=train_root,
    val_root=val_root,
    total_epochs=1,  # run #epochs
    cfg_modify_fn=_cfg_modify_fn)

trainer = build_trainer(name=Trainers.face_detection_scrfd, default_args=kwargs)
trainer.train()
```

### 模型微调

```python
import os
import tempfile
from modelscope.msdatasets import MsDataset
from modelscope.metainfo import Trainers
from modelscope.trainers import build_trainer
from modelscope.hub.snapshot_download import snapshot_download
from modelscope.utils.constant import ModelFile

model_id = 'damo/cv_ddsar_face-detection_iclr23-damofd'
ms_ds_widerface = MsDataset.load('WIDER_FACE_mini', namespace='shaoxuan')  # remove '_mini' for full dataset

data_path = ms_ds_widerface.config_kwargs['split_config']
train_dir = data_path['train']
val_dir = data_path['validation']

def get_name(dir_name):
    names = [i for i in os.listdir(dir_name) if not i.startswith('_')]
    return names[0]

train_root = train_dir + '/' + get_name(train_dir) + '/'
val_root = val_dir + '/' + get_name(val_dir) + '/'
cache_path = snapshot_download(model_id)
tmp_dir = tempfile.TemporaryDirectory().name
pretrain_epochs = 640
ft_epochs = 1
total_epochs = pretrain_epochs + ft_epochs
if not os.path.exists(tmp_dir):
    os.makedirs(tmp_dir)

def _cfg_modify_fn(cfg):
    cfg.checkpoint_config.interval = 1
    cfg.log_config.interval = 10
    cfg.evaluation.interval = 1
    cfg.data.workers_per_gpu = 1
    cfg.data.samples_per_gpu = 4
    return cfg

kwargs = dict(
    cfg_file=os.path.join(cache_path, 'DamoFD_lms.py'),
    work_dir=tmp_dir,
    train_root=train_root,
    val_root=val_root,
    resume_from=os.path.join(cache_path, ModelFile.TORCH_MODEL_FILE),
    total_epochs=total_epochs,  # run #epochs
    cfg_modify_fn=_cfg_modify_fn)

trainer = build_trainer(name=Trainers.face_detection_scrfd, default_args=kwargs)
trainer.train()
```

## 单模型选型和对比
```python
import os.path as osp
import cv2
import os
import numpy as np
from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import voc_ap, image_eval,img_pr_info, gen_gt_info, dataset_pr_info, bbox_overlap

model_id_list = ['damo/cv_ddsar_face-detection_iclr23-damofd', 'damo/cv_resnet101_face-detection_cvpr22papermogface',  'damo/cv_resnet50_face-detection_retinaface', 'damo/cv_manual_face-detection_mtcnn'] 
val_set = MsDataset.load('widerface_mini_train_val', namespace='ly261666', split='validation')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)
img_base_path = next(iter(val_set))[1]
img_dir = osp.join(img_base_path, 'val_data')
img_gt = osp.join(img_base_path, 'val_label.txt')
gt_info = gen_gt_info(img_gt)
pred_info = {}
iou_th = 0.5
thresh_num = 1000
count_face = 0
conf_th = 0.01
final_info = ""
pr_curve = np.zeros((thresh_num, 2)).astype('float')
for model_id in model_id_list:
    pr_curve = np.zeros((thresh_num, 2)).astype('float')
    count_face = 0
    if 'mtcnn' in model_id:
        face_detection_func = pipeline(Tasks.face_detection, model=model_id, conf_th=0.7) # Mtcnn only support high conf threshold
    elif 'damofd' in model_id:
        face_detection_func = pipeline(Tasks.face_detection, model=model_id) # Revise conf_th in DamoFD_lms.py
    else:
        face_detection_func = pipeline(Tasks.face_detection, model=model_id, conf_th=0.01)
    for idx, img_name in enumerate(os.listdir(img_dir)):
        print ('model_id: {}, inference img: {} {}/{}'.format(model_id, img_name, idx+1, len(os.listdir(img_dir))))
        abs_img_name = osp.join(img_dir, img_name)
        result = face_detection_func(abs_img_name)
        pred_info = np.concatenate([result['boxes'], np.array(result['scores'])[:,np.newaxis]], axis=1)
        gt_box = np.array(gt_info[img_name])
        pred_recall, proposal_list = image_eval(pred_info, gt_box, iou_th)
        _img_pr_info, fp = img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)
        pr_curve += _img_pr_info
        count_face += gt_box.shape[0]
    
    pr_curve = dataset_pr_info(thresh_num, pr_curve, count_face)
    propose = pr_curve[:, 0]
    recall = pr_curve[:, 1]
    for srecall in np.arange(0.1, 1.0001, 0.1):
        rindex = len(np.where(recall<=srecall)[0])-1
        rthresh = 1.0 - float(rindex)/thresh_num
        print('Recall-Precision-Thresh:', recall[rindex], propose[rindex], rthresh)
    ap = voc_ap(recall, propose)
    result_info = 'model_id: {}, ap: {:.5f}, iou_th: {:.2f}'.format(model_id, ap, iou_th)
    print(result_info)
    final_info += result_info + '\n'
print("Overall Result:")
print(final_info)
```


<!--- ## 人脸识别系统多模块一键选型/对比 -->






