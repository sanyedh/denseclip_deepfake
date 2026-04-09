import json
import os
import os.path as osp
import numpy as np
from mmseg.datasets.builder import DATASETS, PIPELINES
from mmseg.datasets.custom import CustomDataset
import random


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
                print(
                    f"[DEBUG] Applying threshold {self.threshold}: Values > {self.threshold} -> 1 (Fake), Others -> 0 (Real)")
                self.has_printed = True

            # 创建全 0 矩阵 (默认 Real)
            new_seg = np.zeros_like(gt_seg, dtype=np.uint8)
            # 将大于阈值的像素设为 1 (Fake)
            new_seg[gt_seg > self.threshold] = 1

            results[key] = new_seg

        return results


@DATASETS.register_module()
class DeepfakeDataset(CustomDataset):
    CLASSES = ('real', 'fake')
    PALETTE = [[0, 0, 0], [255, 255, 255]]

    def __init__(self, data_root, dataset_split='training', use_detailed_text=True, **kwargs):
        """
        联合加载 FF++ 四个类别的图像，并利用 unique_key 解决同名文件文本覆盖问题。
        """
        self.dataset_split = dataset_split
        self.use_detailed_text = use_detailed_text

        # FF++ 的四个主要类别文件夹
        self.categories = ['Deepfakes', 'NeuralTextures', 'Face2Face', 'FaceSwap']

        # ================= 【核心修复：安全提取参数】 =================
        # 弹出 test.py 可能强行注入的参数，避免与 super() 冲突
        img_dir = kwargs.pop('img_dir', '')
        ann_dir = kwargs.pop('ann_dir', '')
        text_json_path = kwargs.pop('text_json_path', None)  # <--- 新增这一行：拦截 test.py 塞进来的临时 JSON 路径
        # ==============================================================

        # 1. 初始化并聚合四个类别的 JSON 文本映射
        self.text_mapping = {}

        # 如果 test.py 传了临时的 text_json_path，测试集就优先加载它
        if text_json_path is not None and osp.exists(text_json_path):
            with open(text_json_path, 'r', encoding='utf-8') as f:
                self.text_mapping = json.load(f)
        else:
            # 否则走原来的 FF++ 训练加载逻辑
            json_name = 'train_text.json' if dataset_split == 'training' else 'val_text.json'
            for cat in self.categories:
                json_path = osp.join(data_root, cat, 'text_infos', json_name)
                if osp.exists(json_path):
                    with open(json_path, 'r', encoding='utf-8') as f:
                        cat_dict = json.load(f)
                        for filename, text_desc in cat_dict.items():
                            unique_key = f"{cat}/{filename}"
                            self.text_mapping[unique_key] = text_desc

        super(DeepfakeDataset, self).__init__(
            data_root=data_root,
            img_dir=img_dir,
            ann_dir=ann_dir,
            img_suffix='.jpg',
            seg_map_suffix='.png',
            reduce_zero_label=False,
            **kwargs)

    def load_annotations(self, img_dir, img_suffix, ann_dir, seg_map_suffix, split):
        """兼容常规训练和自定义测试目录的扫描逻辑"""
        img_infos = []

        # ================= 【核心修复：防范 MMSeg 的路径自动拼接机制】 =================
        is_custom_test = False
        if img_dir and img_dir.strip() != '':
            norm_img_dir = osp.normpath(img_dir)
            norm_data_root = osp.normpath(self.data_root)
            if norm_img_dir != norm_data_root:
                is_custom_test = True

        if is_custom_test:
            print(f"\n[Dataset] Custom Test Mode Active! Scanning: {img_dir}\n")
            for img_name in os.listdir(img_dir):
                if img_name.endswith(img_suffix):
                    img_info = dict(
                        filename=osp.join(img_dir, img_name),
                        text_key="custom_test_image"
                    )
                    if ann_dir is not None and osp.exists(ann_dir):
                        seg_map = img_name.replace(img_suffix, seg_map_suffix)
                        img_info['ann'] = dict(seg_map=osp.join(ann_dir, seg_map))
                    img_infos.append(img_info)
            return img_infos
        # ====================================================================

        # 清理了多余的冗余代码，只保留一个干净的 for 循环
        for cat in self.categories:
            cat_img_dir = osp.join(self.data_root, cat, 'images', self.dataset_split)
            cat_ann_dir = osp.join(self.data_root, cat, 'annotations', self.dataset_split)

            if not osp.exists(cat_img_dir):
                print(f"[Warning] Directory not found, skipping: {cat_img_dir}")
                continue

            for img_name in os.listdir(cat_img_dir):
                if img_name.endswith(img_suffix):
                    if cat == 'FaceSwap' and self.dataset_split == 'training':
                        if random.random() < 0:  # 丢弃 FS 训练数据
                            continue
                    img_info = dict(
                        filename=osp.join(cat_img_dir, img_name),
                        text_key=f"{cat}/{img_name}"
                    )
                    seg_map = img_name.replace(img_suffix, seg_map_suffix)
                    img_info['ann'] = dict(seg_map=osp.join(cat_ann_dir, seg_map))
                    img_infos.append(img_info)

        # ================= 【极度关键：已修复缩进】 =================
        # 确保下面的统计和 return 逻辑在所有的 for 循环彻底结束之后才执行
        print(f"\n[Dataset] Split: {self.dataset_split}. Loaded {len(img_infos)} images from 4 categories.\n")

        # 类别统计
        cat_counts = {}
        for info in img_infos:
            cat = info['text_key'].split('/')[0]
            cat_counts[cat] = cat_counts.get(cat, 0) + 1

        print(f"--- 每个类别成功加载的数量统计 ---")
        for c, count in cat_counts.items():
            print(f"[{c}]: 加载了 {count} 张")
        print("=========================================\n")

        return img_infos

    def prepare_train_img(self, idx):
        img_info = self.img_infos[idx]
        ann_info = self.get_ann_info(idx)
        results = dict(img_info=img_info, ann_info=ann_info)

        # ================= 消融实验逻辑 =================
        if self.use_detailed_text:
            text_key = img_info.get('text_key')
            fake_desc = self.text_mapping.get(text_key, "A forged face.")

            # Prompt Dropout (40% 概率退化为通用描述)
            if random.random() < 0.5:
                fake_desc = "A forged face."
        else:
            fake_desc = "A forged face."
        # ===============================================

        results['text_prompts'] = ["A real authentic face.", fake_desc]

        self.pre_pipeline(results)
        return self.pipeline(results)

    def prepare_test_img(self, idx):
        img_info = self.img_infos[idx]
        results = dict(img_info=img_info)

        # 测试/验证时，强制剥离上帝视角
        results['text_prompts'] = ["A real authentic face.", "A forged face."]

        self.pre_pipeline(results)
        return self.pipeline(results)

    def get_gt_seg_map_by_idx(self, index):
        ann_info = self.get_ann_info(index)
        results = dict(ann_info=ann_info)
        self.pre_pipeline(results)
        self.gt_seg_map_loader(results)
        gt_seg = results['gt_semantic_seg']

        # 强制验证集二值化与训练集保持一致
        new_seg = np.zeros_like(gt_seg, dtype=np.uint8)
        new_seg[gt_seg > 10] = 1

        return new_seg

    def get_gt_seg_maps(self, efficient_test=None, **kwargs):
        for idx in range(len(self)):
            yield self.get_gt_seg_map_by_idx(idx)