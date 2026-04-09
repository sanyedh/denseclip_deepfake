import argparse
import os
import json  # <--- 新增导入
import warnings
warnings.filterwarnings(
    'ignore',
    message='On January 1, 2023, MMCV will release v2.0.0, in which it will remove components related to the training process',
    category=UserWarning,
    module='mmcv'
)
import mmcv
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint
from mmcv.utils import DictAction
from mmseg.apis import multi_gpu_test, single_gpu_test
from mmseg.datasets import build_dataloader, build_dataset
from mmseg.models import build_segmentor
import denseclip

def parse_args():
    parser = argparse.ArgumentParser(
        description='mmseg test (and eval) a model')

    # ==================== 【默认权重与配置】 ====================
    parser.add_argument(
        '--config',
        default=r'F:\python_program\deepfake\DenseCLIP-master\segmentation\configs\denseclip_deepfake_r50.py',
        help='test config file path')

    parser.add_argument(
        '--checkpoint',
        default=r'F:\python_program\deepfake\DenseCLIP-master\segmentation\work_dirs\denseclip_deepfake_r50\baseline_10k.pth',
        help='checkpoint file')
    # ========================================================

    # ==================== 【自定义测试集路径】 ====================
    parser.add_argument('--custom-img-dir', type=str, default=None,
                        help='[可选] 自定义测试图片文件夹的绝对路径')
    parser.add_argument('--custom-ann-dir', type=str, default=None,
                        help='[可选] 自定义测试掩码(Ground Truth)文件夹的绝对路径')
    parser.add_argument('--custom-json', type=str, default=None,
                        help='[可选] 自定义测试集文本JSON的绝对路径 (现在可以不用传了！)')
    # ====================================================================

    parser.add_argument('--eval', type=str, nargs='+', default=['mIoU'], help='evaluation metrics')
    parser.add_argument('--show-dir', default=r'F:\python_program\deepfake\DenseCLIP-master\segmentation\work_dirs\denseclip_deepfake_r50\visual_results', help='预测结果的保存目录')
    parser.add_argument('--aug-test', action='store_true', help='Use Flip and Multi scale aug')
    parser.add_argument('--out', help='output result file in pickle format')
    parser.add_argument('--format-only', action='store_true', help='Format the output results without perform evaluation.')
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--gpu-collect', action='store_true', help='whether to use gpu to collect results.')
    parser.add_argument('--tmpdir', help='tmp directory used for collecting results from multiple workers')
    parser.add_argument('--options', nargs='+', action=DictAction, help='custom options')
    parser.add_argument('--eval-options', nargs='+', action=DictAction, help='custom options for evaluation')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--opacity', type=float, default=0.5, help='Opacity of painted segmentation map. In (0, 1] range.')
    parser.add_argument('--local_rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    cfg = mmcv.Config.fromfile(args.config)
    if args.options is not None:
        cfg.merge_from_dict(args.options)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    if args.aug_test:
        cfg.data.test.pipeline[1].img_ratios = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
        cfg.data.test.pipeline[1].flip = True

    cfg.model.pretrained = None

    # 修复找不到 test 配置的问题
    if not hasattr(cfg.data, 'test') or cfg.data.test is None:
        cfg.data.test = cfg.data.val

    cfg.data.test.test_mode = True

    # ==================== 【动态覆盖路径 & 自动生成 JSON 核心逻辑】 ====================
    if args.custom_img_dir:
        print(f"\n[*] 检测到自定义图片路径: {args.custom_img_dir}")
        cfg.data.test.data_root = ''  # 清空默认的根目录前缀
        cfg.data.test.img_dir = args.custom_img_dir

        # 自动免 JSON 逻辑
        if args.custom_json:
            print(f"[*] 检测到自定义JSON路径: {args.custom_json}")
            cfg.data.test.text_json_path = args.custom_json
        else:
            print("[*] 未提供 JSON，系统正在自动扫描图片并生成后台临时 JSON...")
            # 在图片同目录下生成一个临时文件
            dummy_json_path = os.path.join(args.custom_img_dir, '.temp_auto_test.json')
            dummy_dict = {}
            # 扫描支持的图片格式
            for filename in os.listdir(args.custom_img_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    dummy_dict[filename] = "A face." # 填入无意义敷衍文本

            if len(dummy_dict) == 0:
                raise ValueError(f"[!] 报错：在 {args.custom_img_dir} 中没有找到任何图片，请检查路径！")

            with open(dummy_json_path, 'w', encoding='utf-8') as f:
                json.dump(dummy_dict, f)

            cfg.data.test.text_json_path = dummy_json_path
            print(f"[*] 临时 JSON 生成完毕，共扫描到 {len(dummy_dict)} 张测试图片！\n")

    if args.custom_ann_dir:
        print(f"[*] 检测到自定义掩码路径: {args.custom_ann_dir}")
        cfg.data.test.ann_dir = args.custom_ann_dir
    else:
        # 如果没有提供 Ground Truth 掩码，强制关闭 eval
        if args.custom_img_dir and args.eval:
            print("[!] 警告：未提供掩码(ann-dir)，无法计算 mIoU，自动关闭评价指标打印。")
            args.eval = None
    # ====================================================================================

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)

    # build the dataloader
    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=cfg.data.workers_per_gpu,
        dist=distributed, shuffle=False)

    cfg.model.train_cfg = None

    if 'DenseCLIP' in cfg.model.type:
        cfg.model.class_names = list(dataset.CLASSES)

    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    print(f"正在加载权重文件: {args.checkpoint}")
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu')

    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model.PALETTE = checkpoint.get('meta', {}).get('PALETTE', dataset.PALETTE)

    torch.cuda.empty_cache()
    eval_kwargs = {} if args.eval_options is None else args.eval_options
    efficient_test = eval_kwargs.get('efficient_test', False)

    print("开始进行推理测试...")
    if not distributed:
        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, args.show, args.show_dir,
                                  efficient_test, args.opacity)
    else:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()], broadcast_buffers=False)
        outputs = multi_gpu_test(model, data_loader, args.tmpdir, args.gpu_collect, efficient_test)

    print('\n推理完成！')
    rank, _ = get_dist_info()
    if rank == 0:
        kwargs = {} if args.eval_options is None else args.eval_options
        if args.format_only:
            dataset.format_results(outputs, **kwargs)

        if args.eval:
            print("\n正在计算评价指标 (mIoU)...")
            dataset.evaluate(outputs, args.eval, **kwargs)

        if args.show_dir:
            print(f"\n=======================================================")
            print(f"🎉 可视化掩码图像已成功保存至: {args.show_dir}")
            print(f"=======================================================")

        # 打扫战场：删掉刚才临时生成的 JSON，做到无痕测试
        if args.custom_img_dir and not args.custom_json:
            temp_json = os.path.join(args.custom_img_dir, '.temp_auto_test.json')
            if os.path.exists(temp_json):
                os.remove(temp_json)

if __name__ == '__main__':
    main()