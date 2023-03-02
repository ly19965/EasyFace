# Copyright (c) Alibaba, Inc. and its affiliates.
import os.path as osp
import unittest

import cv2
import os
import numpy as np

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_no_lm_result
from modelscope.utils.test_utils import test_level

from modelscope.msdatasets import MsDataset
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level
from modelscope.utils.constant import DownloadMode
from modelscope.utils.cv.image_utils import voc_ap, image_eval,img_pr_info, gen_gt_info, dataset_pr_info, bbox_overlap

class RetinaFaceDetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.model_id = 'damo/cv_resnet50_face-detection_retinaface'
        self.img_path = 'data/test/images/mog_face_detection.jpg'

    def show_result(self, img_path, detection_result):
        img = draw_face_detection_no_lm_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_run_modelhub(self):
        face_detection = pipeline(Tasks.face_detection, model=self.model_id)

        result = face_detection(self.img_path)
        self.show_result(self.img_path, result)

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        val_set = MsDataset.load('widerface_mini_train_val', namespace='ly261666', split='validation')#, download_mode=DownloadMode.FORCE_REDOWNLOAD)
        img_base_path = next(iter(val_set))[1]
        img_dir = osp.join(img_base_path, 'val_data')
        img_gt = osp.join(img_base_path, 'val_label.txt')
        gt_info = gen_gt_info(img_gt)
        pred_info = {}
        iou_th = 0.5
        thresh_num = 1000
        face_detection_func = pipeline(Tasks.face_detection, model=self.model_id, conf_th=0.7)
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        for idx, img_name in enumerate(os.listdir(img_dir)):
            print ('inference img: {} {}/{}'.format(img_name, idx+1, len(os.listdir(img_dir))))
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
        self.show_result(abs_img_name, result)

if __name__ == '__main__':
    unittest.main()
