# Copyright (c) Alibaba, Inc. and its affiliates.
import os
import os.path as osp
import unittest

import cv2
import numpy as np

from modelscope.msdatasets import MsDataset
from modelscope.pipelines import pipeline
from modelscope.utils.constant import DownloadMode, Tasks
from modelscope.utils.cv.image_utils import draw_face_detection_result
from modelscope.utils.demo_utils import DemoCompatibilityCheck
from modelscope.utils.test_utils import test_level


class TinyMogFaceDetectionTest(unittest.TestCase, DemoCompatibilityCheck):
    def setUp(self) -> None:
        self.task = Tasks.face_detection
        self.model_id = 'damo/cv_manual_face-detection_tinymog'
        self.img_path = 'data/test/images/mog_face_detection.jpg'

    def show_result(self, img_path, detection_result):
        img = draw_face_detection_result(img_path, detection_result)
        cv2.imwrite('result.png', img)
        print(f'output written to {osp.abspath("result.png")}')

    def voc_ap(self, rec, prec):

        # correct AP calculation
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
        return ap

    def image_eval(self, pred, gt, iou_thresh):
        """ single image evaluation
        pred: Nx5
        gt: Nx4
        ignore:
        """
        _pred = pred.copy()
        _gt = gt.copy()
        pred_recall = np.zeros(_pred.shape[0])
        recall_list = np.zeros(_gt.shape[0])
        proposal_list = np.ones(_pred.shape[0])

        #_pred[:, 2] = _pred[:, 2] + _pred[:, 0]
        #_pred[:, 3] = _pred[:, 3] + _pred[:, 1]
        _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
        _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

        for h in range(_pred.shape[0]):
            gt_overlap = self.bbox_overlap(_gt, _pred[h])
            #gt_overlap = gt_overlap_list[h]
            max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()

            if max_overlap >= iou_thresh:
                if recall_list[max_idx] == 0:
                    recall_list[max_idx] = 1

            r_keep_index = np.where(recall_list == 1)[0]
            pred_recall[h] = len(r_keep_index)

        return pred_recall, proposal_list

    def img_pr_info(self, thresh_num, pred_info, proposal_list, pred_recall):
        pr_info = np.zeros((thresh_num, 2)).astype('float')
        fp = np.zeros((pred_info.shape[0], ), dtype=np.int32)
        last_info = [-1, -1]
        for t in range(thresh_num):

            thresh = 1 - (t + 1) / thresh_num
            r_index = np.where(pred_info[:, 4] >= thresh)[0]
            if len(r_index) == 0:
                pr_info[t, 0] = 0
                pr_info[t, 1] = 0
            else:
                r_index = r_index[-1]
                p_index = np.where(proposal_list[:r_index + 1] == 1)[0]
                pr_info[t, 0] = len(p_index)  #valid pred number
                pr_info[t, 1] = pred_recall[r_index]  # valid gt number

                if t > 0 and pr_info[t, 0] > pr_info[t - 1, 0] and pr_info[
                        t, 1] == pr_info[t - 1, 1]:
                    fp[r_index] = 1
        return pr_info, fp

    def gen_gt_info(self, img_gt):
        gt_info = {}
        fo = open(img_gt)
        for line in fo:
            if 'jpg' in line:
                img_name = line.strip()
                gt_info[img_name] = []
                continue
            gt_info[img_name].append(
                [float(item) for item in line.strip().split(' ')[:4]])
        return gt_info

    def dataset_pr_info(self, thresh_num, pr_curve, count_face):
        _pr_curve = np.zeros((thresh_num, 2))
        for i in range(thresh_num):
            _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
            _pr_curve[i, 1] = pr_curve[i, 1] / count_face
        return _pr_curve

    def bbox_overlap(self, a, b):
        x1 = np.maximum(a[:, 0], b[0])
        y1 = np.maximum(a[:, 1], b[1])
        x2 = np.minimum(a[:, 2], b[2])
        y2 = np.minimum(a[:, 3], b[3])
        w = x2 - x1 + 1
        h = y2 - y1 + 1
        inter = w * h
        aarea = (a[:, 2] - a[:, 0] + 1) * (a[:, 3] - a[:, 1] + 1)
        barea = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        o = inter / (aarea + barea - inter)
        o[w <= 0] = 0
        o[h <= 0] = 0
        return o

    @unittest.skipUnless(test_level() >= 1, 'skip test in current test level')
    def test_run_with_dataset(self):
        input_location = [
            'data/test/images/mog_face_detection.jpg',
            'data/test/images/mog_face_detection.jpg'
        ]

        dataset = MsDataset.load(input_location, target='image')
        val_set = MsDataset.load(
            'widerface_mini_train_val',
            namespace='ly261666',
            split='validation'
        )  #, download_mode=DownloadMode.FORCE_REDOWNLOAD)
        img_base_path = next(iter(val_set))[1]
        img_dir = osp.join(img_base_path, 'val_data')
        img_gt = osp.join(img_base_path, 'val_label.txt')
        gt_info = self.gen_gt_info(img_gt)
        pred_info = {}
        iou_th = 0.5
        thresh_num = 1000
        face_detection_func = pipeline(Tasks.face_detection,
                                       model=self.model_id)
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2)).astype('float')
        for img_name in os.listdir(img_dir):
            abs_img_name = osp.join(img_dir, img_name)
            result = face_detection_func(abs_img_name)
            pred_info = np.concatenate(
                [result['boxes'],
                 np.array(result['scores'])[:, np.newaxis]],
                axis=1)
            gt_box = np.array(gt_info[img_name])
            pred_recall, proposal_list = self.image_eval(
                pred_info, gt_box, iou_th)
            _img_pr_info, fp = self.img_pr_info(thresh_num, pred_info,
                                                proposal_list, pred_recall)
            pr_curve += _img_pr_info
            count_face += gt_box.shape[0]

        pr_curve = self.dataset_pr_info(thresh_num, pr_curve, count_face)
        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]
        for srecall in np.arange(0.1, 1.0001, 0.1):
            rindex = len(np.where(recall <= srecall)[0]) - 1
            rthresh = 1.0 - float(rindex) / thresh_num
            print('Recall-Precision-Thresh:', recall[rindex], propose[rindex],
                  rthresh)
        ap = self.voc_ap(recall, propose)
        print('ap: %.5f, iou_th: %.2f' % (ap, iou_th))
        self.show_result(abs_img_name, result)
        import pdb
        pdb.set_trace()

    #@unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    #def test_run_modelhub(self):
    #    face_detection = pipeline(Tasks.face_detection, model=self.model_id)

    #    result = face_detection(self.img_path)
    #    self.show_result(self.img_path, result)

    @unittest.skipUnless(test_level() >= 0, 'skip test in current test level')
    def test_demo_compatibility(self):
        self.compatibility_check()


if __name__ == '__main__':
    unittest.main()
