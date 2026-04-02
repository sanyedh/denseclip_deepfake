import json
import os.path as osp
import numpy as np
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset
import random
import cv2

# @PIPELINES.register_module()
# class RandomJPEGCompression(object):
#     """Deepfake 专属：随机进行 JPEG 压缩，模拟网络传播中的降质"""
#     def __init__(self, quality_lower=60, quality_upper=95, p=0.5):
#         self.quality_lower = quality_lower
#         self.quality_upper = quality_upper
#         self.p = p
#
#     def __call__(self, results):
#         if random.random() < self.p:
#             img = results['img']
#             quality = random.randint(self.quality_lower, self.quality_upper)
#             encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
#             success, encimg = cv2.imencode('.jpg', img, encode_param)
#             if success:
#                 decimg = cv2.imdecode(encimg, 1)
#                 results['img'] = decimg
#         return results
#
# @PIPELINES.register_module()
# class RandomGaussianBlur(object):
#     """Deepfake 专属：随机高斯模糊，模拟低质量媒体"""
#     def __init__(self, kernel_size_list=[3, 5], p=0.5):
#         self.kernel_size_list = kernel_size_list
#         self.p = p
#
#     def __call__(self, results):
#         if random.random() < self.p:
#             img = results['img']
#             ksize = random.choice(self.kernel_size_list)
#             results['img'] = cv2.GaussianBlur(img, (ksize, ksize), 0)
#         return results

@PIPELINES.register_module()
class ForceBinaryLabels(object):
    """强制将掩码二值化，过滤低亮度噪声"""
    def __init__(self, threshold=10):
        self.threshold = threshold
        self.has_printed = False

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            gt_seg = results[key]

            if not self.has_printed:
                unique_vals = np.unique(gt_seg)
                print(f"\n[DEBUG] Processing Mask. Unique values: {unique_vals}")
                print(f"[DEBUG] Applying threshold {self.threshold}: Values > {self.threshold} -> 1 (Fake), Others -> 0 (Real)")
                self.has_printed = True

            # 创建全 0 矩阵 (默认 Real)
            new_seg = np.zeros_like(gt_seg, dtype=np.uint8)
            # 将大于阈值的像素设为 1 (Fake)
            new_seg[gt_seg > self.threshold] = 1

            results[key] = new_seg

        return results

@DATASETS.register_module()
class DeepfakeDataset(CustomDataset):
    """
    Deepfake Dataset.
    Class 0: Real (Background)
    Class 1: Fake (Forged)
    """
    CLASSES = ('real', 'fake')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, text_json_path, **kwargs):
        self.text_json_path = text_json_path
        with open(text_json_path, 'r', encoding='utf-8') as f:
            self.text_mapping = json.load(f)

        super(DeepfakeDataset, self).__init__(
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        filename = osp.basename(img_info['filename'])
        fake_desc = self.text_mapping.get(filename, "A forged face.")

        # 【修改策略】：Prompt Dropout (50% 概率退化为通用描述)
        # 让模型既能吸收详细知识，又能适应盲测
        if random.random() < 0.4:
            fake_desc = "A forged face."

        results['text_prompts'] = ["A real authentic face.", fake_desc]

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)

        # 【修改策略】：测试/验证时，必须剥离上帝视角，强制使用通用 Prompt
        results['text_prompts'] = ["A real authentic face.", "A forged face."]

        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        gt_seg = results['gt_semantic_seg']

        # 【修改】强制验证集二值化与训练集(threshold=10)保持一致
        new_seg = np.zeros_like(gt_seg, dtype=np.uint8)
        new_seg[gt_seg > 10] = 1

        return new_seg

    def get_gt_seg_maps(self, efficient_test=None, **kwargs):
        """
        重写底层方法：强制 evaluate 评估阶段不直接读硬盘，
        而是调用我们的 get_gt_seg_map_by_idx 方法获取 (0, 1) 二值化后的标签。
        """
        for idx in range(len(self)):
            yield self.get_gt_seg_map_by_idx(idx)