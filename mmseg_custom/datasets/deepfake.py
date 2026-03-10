import json
import os.path as osp
import numpy as np
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset

@PIPELINES.register_module()
class ForceBinaryLabels(object):
    """强制将掩码二值化，过滤低亮度噪声"""
    def __init__(self, threshold=127):
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
            # 将大于阈值 (127) 的像素设为 1 (Fake)
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
        results['text_prompts'] = ["A real authentic face.", fake_desc]

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)

        filename = osp.basename(img_info['filename'])
        fake_desc = self.text_mapping.get(filename, "A forged face.")
        results['text_prompts'] = ["A real authentic face.", fake_desc]

        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        gt_seg = results['gt_semantic_seg']

        # 强制验证集二值化评估
        new_seg = np.zeros_like(gt_seg, dtype=np.uint8)
        new_seg[gt_seg > 127] = 1

        return new_seg