import cv2
import numpy as np
import dlib
import os
import random
from tqdm import tqdm
from glob import glob
import skimage.feature
import skimage.metrics
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from scipy.ndimage import binary_erosion, binary_dilation
from scipy.fftpack import fft2, fftshift
import traceback
import sys

# ================= 配置区域 =================
PREDICTOR_PATH = r"F:\python_program\deepfake\VLFFD-main\shape_predictor_68_face_landmarks .dat"
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # 禁用GPU，减少内存占用

# ================= 常量定义 =================
FACE_POINTS = list(range(17, 68))
LANDMARKS_INDEX = {
    "face": list(range(17, 68)),
    "mouth": list(range(48, 61)),
    "eyes": list(range(36, 48)),
    "nose": list(range(27, 35)),
}

PAPER_THRESHOLDS = {
    "color_mean": 1.0,
    "color_var": 0.5,
    "blur": 100,
    "structure_ssim": 0.97,
    "texture_contrast": 0.7,
    "blending_gradient": 0.3,
    "blending_edge": 0.3,
    "blending_frequency": 0.2,
    "forgery_region_theta": 0.05
}

FORGERY_TYPE_TO_TEXT = {
    "color_difference": "has inconsistent colors",
    "blur": "lacks sharp details (blurry)",
    "structure_abnormal": "exhibits structural distortion",
    "texture_abnormal": "lacks natural texture",
    "blend_boundary": "has blending artifacts at boundaries"
}

def dist(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)

# ================= 工具函数 =================
def load_landmark_detector() -> Tuple[dlib.fhog_object_detector, dlib.shape_predictor]:
    if not os.path.exists(PREDICTOR_PATH):
        print(f"\n❌ 关键点检测器文件不存在！")
        print(f"  路径：{PREDICTOR_PATH}")
        print(f"  下载链接：")
        print(f"    官方：http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
        print(f"    备用：https://pan.baidu.com/s/1Z98sL8t3yWf6Ck88v4c3PA（提取码：dlib）")
        sys.exit(1)

    if not os.access(PREDICTOR_PATH, os.R_OK):
        print(f"\n❌ 无权限读取文件：{PREDICTOR_PATH}")
        print(f"  解决方案：1. 取消文件只读属性 2. 以管理员身份运行脚本")
        sys.exit(1)

    try:
        face_detector = dlib.get_frontal_face_detector()
        shape_predictor = dlib.shape_predictor(PREDICTOR_PATH)
        print(f"✅ 成功加载关键点检测器：{os.path.basename(PREDICTOR_PATH)}")
        return face_detector, shape_predictor
    except Exception as e:
        print(f"\n❌ 加载失败：{str(e)}")
        print(f"  建议：安装 dlib==19.24.0")
        sys.exit(1)

# ================= 类定义 =================
class BlendingDetector:
    def __init__(self):
        self.gradient_threshold = PAPER_THRESHOLDS["blending_gradient"]
        self.edge_threshold = PAPER_THRESHOLDS["blending_edge"]
        self.frequency_threshold = PAPER_THRESHOLDS["blending_frequency"]
        self.boundary_width = 10

    def get_boundary_region(self, mask: np.ndarray, width: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        binary_mask = (mask > 0).astype(np.uint8)
        kernel = np.ones((3,3), np.uint8)
        inner_mask = cv2.erode(binary_mask, kernel, iterations=width)
        outer_mask = cv2.dilate(binary_mask, kernel, iterations=width)
        inner_boundary = binary_mask - inner_mask
        outer_boundary = outer_mask - binary_mask
        return inner_boundary, outer_boundary

    def detect_edge_artifacts(self, image: np.ndarray, mask: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        inner_boundary, outer_boundary = self.get_boundary_region(mask, self.boundary_width)
        total_boundary = inner_boundary + outer_boundary

        if np.sum(total_boundary) == 0:
            return 0.0

        edges = cv2.Canny(gray_image, 100, 200)
        boundary_edges = edges * total_boundary
        edge_density = np.sum(boundary_edges) / (np.sum(total_boundary) + 1e-6)
        return edge_density

    def detect_gradient_discontinuity(self, image: np.ndarray, mask: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        inner_boundary, outer_boundary = self.get_boundary_region(mask, self.boundary_width)

        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)

        inner_gradient = np.mean(gradient_magnitude[inner_boundary > 0]) if np.sum(inner_boundary) > 0 else 0
        outer_gradient = np.mean(gradient_magnitude[outer_boundary > 0]) if np.sum(outer_boundary) > 0 else 0

        gradient_diff = abs(inner_gradient - outer_gradient) / (outer_gradient + 1e-6)
        return gradient_diff

    def detect_frequency_domain_change(self, image: np.ndarray, mask: np.ndarray) -> float:
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
        if len(mask.shape) == 3:
            mask = mask[..., 0]

        inner_boundary, outer_boundary = self.get_boundary_region(mask, self.boundary_width)
        total_boundary = inner_boundary + outer_boundary

        if np.sum(total_boundary) == 0:
            return 0.0

        fft = fft2(gray_image)
        fft_shift = fftshift(fft)
        magnitude = 20 * np.log(np.abs(fft_shift) + 1e-6)

        boundary_coords = np.where(total_boundary > 0)
        if len(boundary_coords[0]) == 0:
            return 0.0

        boundary_magnitude = magnitude[boundary_coords]
        inner_coords = np.where(inner_boundary > 0)
        inner_magnitude = magnitude[inner_coords] if len(inner_coords[0]) > 0 else boundary_magnitude
        outer_coords = np.where(outer_boundary > 0)
        outer_magnitude = magnitude[outer_coords] if len(outer_coords[0]) > 0 else boundary_magnitude

        def calc_freq_ratio(mag, img_shape):
            h, w = img_shape
            center = (h//2, w//2)
            radius = min(center) // 4
            y, x = np.indices(mag.shape[:2])
            dist_from_center = np.sqrt((x - center[1])**2 + (y - center[0])**2)
            low_mask = (dist_from_center <= radius).astype(float)
            low_energy = np.sum(mag * low_mask)
            high_energy = np.sum(mag * (1 - low_mask))
            return low_energy / (high_energy + 1e-6)

        inner_ratio = calc_freq_ratio(inner_magnitude.reshape(-1, 1), gray_image.shape)
        outer_ratio = calc_freq_ratio(outer_magnitude.reshape(-1, 1), gray_image.shape)
        freq_diff = abs(inner_ratio - outer_ratio)
        return freq_diff

    def detect_blending_artifacts(self, image: np.ndarray, mask: np.ndarray) -> bool:
        grad_diff = self.detect_gradient_discontinuity(image, mask)
        edge_density = self.detect_edge_artifacts(image, mask)
        freq_diff = self.detect_frequency_domain_change(image, mask)

        count = 0
        if grad_diff > self.gradient_threshold:
            count += 1
        if edge_density > self.edge_threshold:
            count += 1
        if freq_diff > self.frequency_threshold:
            count += 1

        return count >= 2

@dataclass
class ForgeryFeatures:
    COLOR_DIFFERENCE = "color_difference"
    BLUR = "blur"
    STRUCTURE_ABNORMAL = "structure_abnormal"
    TEXTURE_ABNORMAL = "texture_abnormal"
    BLEND_BOUNDARY = "blend_boundary"

class ForgeryDetector:
    def __init__(self, face_detector: dlib.fhog_object_detector, shape_predictor: dlib.shape_predictor):
        self.detector = face_detector
        self.predictor = shape_predictor
        self.blending_detector = BlendingDetector()
        self.thresholds = PAPER_THRESHOLDS

    def get_landmarks(self, image: np.ndarray) -> Optional[np.ndarray]:
        rects = self.detector(image, 1)
        if len(rects) == 0:
            return None
        largest_rect = max(rects, key=lambda r: (r.right() - r.left()) * (r.bottom() - r.top()))
        points = self.predictor(image, largest_rect).parts()
        landmarks = np.array([[p.x, p.y] for p in points])
        return landmarks

    def check_aspect_ratio_match(self, img1: np.ndarray, img2: np.ndarray, tol: float = 0.05) -> bool:
        h1, w1 = img1.shape[:2]
        h2, w2 = img2.shape[:2]
        ratio1 = w1 / h1
        ratio2 = w2 / h2
        return abs(ratio1 - ratio2) < tol

    def generate_mask_from_diff(self, real_image: np.ndarray, fake_image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if real_image.shape != fake_image.shape:
            real_image = cv2.resize(real_image, (fake_image.shape[1], fake_image.shape[0]))

        diff_per_channel = cv2.absdiff(real_image, fake_image)
        mask_normalized = diff_per_channel / 255.0
        mask_normalized = np.max(mask_normalized, axis=2)
        mask_uint8 = (mask_normalized * 255).astype(np.uint8)
        return mask_uint8, mask_normalized

    def extract_region(self, image: np.ndarray, landmarks: np.ndarray,
                       forgery_mask: np.ndarray, region: str) -> Tuple[np.ndarray, float]:
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        if region not in LANDMARKS_INDEX:
            return np.zeros_like(image), 0.0

        region_points = landmarks[LANDMARKS_INDEX[region]]
        hull = cv2.convexHull(region_points)
        cv2.fillConvexPoly(mask, hull, 1)
        mask = binary_dilation(mask, iterations=5).astype(np.uint8)

        if forgery_mask.shape[:2] != mask.shape[:2]:
            forgery_mask = cv2.resize(forgery_mask, (mask.shape[1], mask.shape[0]))

        region_pixels = forgery_mask[mask > 0]
        avg_mask_value = np.mean(region_pixels) if len(region_pixels) > 0 else 0.0

        y, x = np.where(mask > 0)
        if len(y) == 0 or len(x) == 0:
            return np.zeros((32, 32, 3), dtype=np.uint8), avg_mask_value
        y1, y2, x1, x2 = y.min(), y.max(), x.min(), x.max()
        cropped = image[y1:y2+1, x1:x2+1]

        if cropped.size == 0:
            cropped = np.zeros((32, 32, 3), dtype=np.uint8)
        return cropped, avg_mask_value

    def select_forgery_region(self, mask_normalized: np.ndarray, landmarks: np.ndarray) -> Optional[str]:
        L_f = []
        regions = ["mouth", "eyes", "nose", "face"]

        for region in regions:
            _, avg_mask_value = self.extract_region(
                np.zeros_like(mask_normalized), landmarks, mask_normalized, region
            )
            if avg_mask_value > self.thresholds["forgery_region_theta"]:
                L_f.append(region)

        if not L_f:
            return None
        return random.choice(L_f)

    def detect_color_inconsistency(self, real_part: np.ndarray, fake_part: np.ndarray) -> bool:
        if real_part.size == 0 or fake_part.size == 0:
            return False
        if real_part.shape != fake_part.shape:
            fake_part = cv2.resize(fake_part, (real_part.shape[1], real_part.shape[0]))

        real_lab = cv2.cvtColor(real_part, cv2.COLOR_RGB2LAB)
        fake_lab = cv2.cvtColor(fake_part, cv2.COLOR_RGB2LAB)

        real_mean = np.mean(real_lab, axis=(0, 1))
        fake_mean = np.mean(fake_lab, axis=(0, 1))
        mean_dist = np.linalg.norm(real_mean - fake_mean)

        real_var = np.var(real_lab, axis=(0, 1))
        fake_var = np.var(fake_lab, axis=(0, 1))
        var_dist = np.linalg.norm(real_var - fake_var)

        return mean_dist > self.thresholds["color_mean"] or var_dist > self.thresholds["color_var"]

    def detect_blur(self, real_part: np.ndarray, fake_part: np.ndarray) -> bool:
        if real_part.size == 0 or fake_part.size == 0:
            return False
        if real_part.shape != fake_part.shape:
            fake_part = cv2.resize(fake_part, (real_part.shape[1], real_part.shape[0]))

        real_gray = cv2.cvtColor(real_part, cv2.COLOR_RGB2GRAY) if len(real_part.shape) == 3 else real_part
        fake_gray = cv2.cvtColor(fake_part, cv2.COLOR_RGB2GRAY) if len(fake_part.shape) == 3 else fake_part

        real_var = cv2.Laplacian(real_gray, cv2.CV_64F).var()
        fake_var = cv2.Laplacian(fake_gray, cv2.CV_64F).var()

        return (real_var > fake_var) and (real_var - fake_var) > self.thresholds["blur"]

    def detect_structural_deformation(self, real_part: np.ndarray, fake_part: np.ndarray) -> bool:
        if real_part.size == 0 or fake_part.size == 0:
            return False
        if real_part.shape != fake_part.shape:
            fake_part = cv2.resize(fake_part, (real_part.shape[1], real_part.shape[0]))

        ssim = skimage.metrics.structural_similarity(
            real_part, fake_part, win_size=min(7, min(real_part.shape[:2])),
            channel_axis=2 if len(real_part.shape) == 3 else None,
            data_range=255
        )

        return ssim < self.thresholds["structure_ssim"]

    def detect_texture_anomaly(self, real_part: np.ndarray, fake_part: np.ndarray) -> bool:
        if real_part.size == 0 or fake_part.size == 0:
            return False
        if real_part.shape != fake_part.shape:
            fake_part = cv2.resize(fake_part, (real_part.shape[1], real_part.shape[0]))

        real_gray = cv2.cvtColor(real_part, cv2.COLOR_RGB2GRAY) if len(real_part.shape) == 3 else real_part
        fake_gray = cv2.cvtColor(fake_part, cv2.COLOR_RGB2GRAY) if len(fake_part.shape) == 3 else fake_part

        def glcm_contrast(img):
            img = (img // 32).astype(np.uint8)
            glcm = skimage.feature.graycomatrix(
                img, distances=[1], angles=[0], levels=8, symmetric=True, normed=True
            )
            return skimage.feature.graycoprops(glcm, 'contrast')[0, 0]

        real_contrast = glcm_contrast(real_gray)
        fake_contrast = glcm_contrast(fake_gray)

        return (real_contrast - fake_contrast) > self.thresholds["texture_contrast"]

class RawAnnotationGenerator:
    def __init__(self, detector: ForgeryDetector):
        self.detector = detector

    def generate(self, real_image: np.ndarray, fake_image: np.ndarray, fake_mask: np.ndarray,
                 fake_landmarks: np.ndarray, real_landmarks: np.ndarray) -> Optional[str]:
        selected_region = self.detector.select_forgery_region(fake_mask, fake_landmarks)
        if selected_region is None:
            return None

        real_region, _ = self.detector.extract_region(real_image, real_landmarks, fake_mask, selected_region)
        fake_region, _ = self.detector.extract_region(fake_image, fake_landmarks, fake_mask, selected_region)

        detected_types = []
        if self.detector.detect_color_inconsistency(real_region, fake_region):
            detected_types.append(ForgeryFeatures.COLOR_DIFFERENCE)
        if self.detector.detect_blur(real_region, fake_region):
            detected_types.append(ForgeryFeatures.BLUR)
        if self.detector.detect_structural_deformation(real_region, fake_region):
            detected_types.append(ForgeryFeatures.STRUCTURE_ABNORMAL)
        if self.detector.detect_texture_anomaly(real_region, fake_region):
            detected_types.append(ForgeryFeatures.TEXTURE_ABNORMAL)
        if self.detector.blending_detector.detect_blending_artifacts(fake_image, fake_mask):
            detected_types.append(ForgeryFeatures.BLEND_BOUNDARY)

        if not detected_types:
            return None

        text_descriptions = [FORGERY_TYPE_TO_TEXT[ft] for ft in detected_types]
        descriptions_str = " and ".join(text_descriptions)
        raw_annotation = f"This is a fake face. The {selected_region} region {descriptions_str}."
        return raw_annotation

class ForgeryDescriptionGenerator:
    def __init__(self, detector: ForgeryDetector):
        self.detector = detector
        self.raw_generator = RawAnnotationGenerator(detector)

    def generate_clip_prompt(self, raw_annotation: str) -> str:
        if "mouth" in raw_annotation:
            region = "mouth"
        elif "eyes" in raw_annotation:
            region = "eyes"
        elif "nose" in raw_annotation:
            region = "nose"
        else:
            region = "face"

        if "inconsistent colors" in raw_annotation:
            feature = "color inconsistency"
        elif "blurry" in raw_annotation:
            feature = "blurry details"
        elif "structural distortion" in raw_annotation:
            feature = "structural distortion"
        elif "natural texture" in raw_annotation:
            feature = "unnatural texture"
        elif "blending artifacts" in raw_annotation:
            feature = "blending artifacts"
        else:
            feature = "digital manipulation artifacts"

        return f"A forged face image with {feature} in the {region} region."

# ================= 单进程处理函数 =================
def process_single_frame(fake_path, image_root, mask_root, detector, generator):
    """单进程处理单帧，避免多进程内存溢出"""
    try:
        fake_path = fake_path.replace('\\', '/')
        fake_image = cv2.imread(fake_path)
        if fake_image is None:
            print(f"❌ 无法读取伪造图像：{fake_path}")
            return None
        fake_image_rgb = cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB)

        # 关键点检测
        fake_landmarks = detector.get_landmarks(fake_image_rgb)
        if fake_landmarks is None:
            print(f"❌ 未检测到伪造图像人脸：{fake_path}")
            return None

        # 真实图像路径匹配
        try:
            path_parts = fake_path.split("/")
            seq_idx = path_parts.index("manipulated_sequences")
            path_parts[seq_idx] = "original_sequences_nt"
            method_idx = seq_idx + 1
            path_parts[method_idx] = "youtube"
            vid_id_idx = -2
            vid_id = path_parts[vid_id_idx]
            if '_' in vid_id:
                path_parts[vid_id_idx] = vid_id.split('_')[0]
            real_path = "/".join(path_parts)
            real_paths = [real_path, real_path.replace(".jpg", ".png"), real_path.replace(".png", ".jpg")]
        except Exception as e:
            fake_folder = os.path.dirname(fake_path)
            fake_folder_name = os.path.basename(fake_folder)
            original_id = fake_folder_name.split('_')[0] if '_' in fake_folder_name else fake_folder_name
            real_folder = fake_folder.replace("manipulated_sequences", "original_sequences_nt")
            real_folder = real_folder.replace(os.path.basename(real_folder), original_id)
            real_folder = real_folder.replace(os.path.basename(os.path.dirname(real_folder)), "youtube")
            real_path = os.path.join(real_folder, os.path.basename(fake_path))
            real_paths = [real_path, real_path.replace(".jpg", ".png"), real_path.replace(".png", ".jpg")]

        # 读取真实图像
        real_image = None
        for rp in real_paths:
            if os.path.exists(rp):
                real_image = cv2.imread(rp)
                if real_image is not None:
                    real_path = rp
                    break
        if real_image is None:
            print(f"❌ 无法读取真实图像：{real_paths[0]}")
            return None
        real_image_rgb = cv2.cvtColor(real_image, cv2.COLOR_BGR2RGB)

        # 真实图像关键点
        real_landmarks = detector.get_landmarks(real_image_rgb)
        if real_landmarks is None:
            real_landmarks = fake_landmarks

        # 生成掩码
        mask_uint8, mask_normalized = detector.generate_mask_from_diff(real_image_rgb, fake_image_rgb)
        if np.max(mask_normalized) < 0.01:
            return None

        # 生成注释
        raw_annotation = generator.raw_generator.generate(
            real_image_rgb, fake_image_rgb, mask_normalized, fake_landmarks, real_landmarks
        )
        if raw_annotation is None:
            return None

        clip_prompt = generator.generate_clip_prompt(raw_annotation)
        result = f"{fake_path}*{raw_annotation}*{clip_prompt}\n"
        print(f"✅ 生成注释：{raw_annotation}")
        return result

    except Exception as e:
        traceback.print_exc()
        print(f"❌ 处理 {fake_path} 时出错: {str(e)}")
        return None

# ================= 主控制类 =================
class PFIG:
    def __init__(self, image_root: str, mask_root: str):
        self.image_root = image_root
        self.mask_root = mask_root

    def sample_frames(self, video_folder: str, num_frames: int = 3) -> List[str]:
        frame_paths = []
        for ext in ['*.jpg', '*.png']:
            frame_paths.extend(sorted(glob(os.path.join(video_folder, ext))))

        if not frame_paths:
            print(f"⚠️  视频文件夹 {video_folder} 无有效图像")
            return []

        total = len(frame_paths)
        if total <= num_frames:
            return frame_paths

        indices = np.linspace(0, total-1, num_frames, dtype=int)
        sampled = [frame_paths[i] for i in indices]
        return sampled

    def generate_dataset_singleprocess(self, face_detector, shape_predictor, methods: List[str] = None, frames_per_video: int = 3):
        """单进程生成数据集，避免内存溢出"""
        if methods is None:
            methods = ['Deepfakes', 'Face2Face', 'FaceSwap', 'NeuralTextures']

        output_dir = './output'
        os.makedirs(output_dir, exist_ok=True)

        # 预加载检测器和生成器（全局唯一，节省内存）
        detector = ForgeryDetector(face_detector, shape_predictor)
        generator = ForgeryDescriptionGenerator(detector)

        # 扫描所有任务
        all_tasks = []
        task_method_map = {}
        print("\n🔍 正在扫描数据集...")
        for method in methods:
            search_pattern = os.path.join(self.image_root, method, "c23", "*", "")
            video_folders = sorted(glob(search_pattern))

            print(f"  - {method}: 找到 {len(video_folders)} 个视频文件夹")

            for video_folder in video_folders:
                frames = self.sample_frames(video_folder, frames_per_video)
                for frame_path in frames:
                    all_tasks.append(frame_path)
                    task_method_map[frame_path] = method

        print(f"\n📦 总任务数: {len(all_tasks)}")
        if not all_tasks:
            print("❌ 未找到任何待处理图像")
            return

        # 打开文件句柄
        files = {}
        for method in methods:
            file_path = os.path.join(output_dir, f'{method}_annotations.txt')
            files[method] = open(file_path, 'w', encoding='utf-8')
            files[method].write("fake_path*raw_annotation*clip_prompt\n")

        success_count = 0
        # 单进程循环处理所有任务
        for frame_path in tqdm(all_tasks, total=len(all_tasks)):
            method = task_method_map[frame_path]
            result = process_single_frame(frame_path, self.image_root, self.mask_root, detector, generator)
            if result:
                files[method].write(result)
                files[method].flush()
                success_count += 1

        # 关闭文件
        for f in files.values():
            f.close()

        print(f"\n✅ 处理完成！")
        print(f"📊 统计：总任务数 {len(all_tasks)}，成功生成 {success_count} 条注释")
        for method in methods:
            file_path = os.path.join(output_dir, f'{method}_annotations.txt')
            if os.path.exists(file_path):
                line_count = sum(1 for _ in open(file_path, encoding='utf-8')) - 1
                print(f"  - {method}: {line_count} 条注释")

def main():
    image_root = r"F:\python_program\deepfake\VLFFD-main\dataset\manipulated_sequences"
    mask_root = r"F:\python_program\deepfake\VLFFD-main\dataset\manipulated_sequences"

    # 预加载检测器（全局唯一，避免重复加载）
    face_detector, shape_predictor = load_landmark_detector()

    pfig = PFIG(image_root, mask_root)
    # 单进程运行，避免内存溢出
    pfig.generate_dataset_singleprocess(face_detector, shape_predictor, frames_per_video=3)

if __name__ == "__main__":
    main()
