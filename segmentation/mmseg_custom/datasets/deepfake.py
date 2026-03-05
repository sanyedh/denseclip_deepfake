import json
import os.path as osp
import numpy as np
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset
from mmseg.datasets.pipelines import Compose

# --- 新增: 强制二值化标签的 Transform ---
# 修改 ForceBinaryLabels 类
@PIPELINES.register_module()
class ForceBinaryLabels(object):
    def __init__(self, threshold=127):  # 【修改1】阈值改为 127，过滤掉 0-14 的噪声
        self.threshold = threshold
        self.has_printed = False

    def __call__(self, results):
        for key in results.get('seg_fields', []):
            gt_seg = results[key]

            # 调试打印（保留用于验证）
            if not self.has_printed:
                unique_vals = np.unique(gt_seg)
                print(f"\n[DEBUG] Processing Mask. Unique values: {unique_vals}")
                print(f"[DEBUG] Applying threshold {self.threshold}: Values > {self.threshold} -> 1 (Fake), Others -> 0 (Real)")
                self.has_printed = True

            # 【核心逻辑修改】
            # 1. 创建全 0 矩阵 (默认 Real)
            new_seg = np.zeros_like(gt_seg, dtype=np.uint8)

            # 2. 将大于阈值 (127) 的像素设为 1 (Fake)
            # 这会将 242~255 全部归为 Fake，同时将 0~127 全部归为 Real
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
        # 加载文本映射
        with open(text_json_path, 'r', encoding='utf-8') as f:
            self.text_mapping = json.load(f)

        super(DeepfakeDataset, self).__init__(
            img_suffix='.jpg',
            # 建议：如果可能，尽量将数据集转为 png 格式以避免压缩噪声
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def prepare_train_img(self, idx):
        """训练时读取图片并注入文本"""
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        # 注入动态文本
        filename = img_info['filename']
        fake_desc = self.text_mapping.get(filename, "A forged face.")
        results['text_prompts'] = ["A real authentic face.", fake_desc]

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        """测试时读取图片并注入文本"""
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)

        filename = img_info['filename']
        fake_desc = self.text_mapping.get(filename, "A forged face.")
        results['text_prompts'] = ["A real authentic face.", fake_desc]

        self.pre_pipeline(results)
        return self.pipeline(results)