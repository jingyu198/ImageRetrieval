import os
from PIL import Image, UnidentifiedImageError
import torchvision.transforms as transforms
from torch.utils.data import Dataset

# 自定义Dataset类来递归处理图像数据
class ImageDataset(Dataset):
    def __init__(self, image_dir, init_count_plus = 0, crop = True):
        self.init_count_plus = init_count_plus
        self.bbox = (30, 80, 1800, 1020)  # 裁剪框
        self.image_dir = image_dir
        self.crop = crop  # 新增裁剪参数
        self.image_files = self.get_image_files(image_dir)
        
        # 定义变换
        transform_list = []
        if self.crop:
            transform_list.append(transforms.Lambda(lambda img: img.crop(self.bbox).convert("RGB")))
        
        transform_list.extend([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        self.transform = transforms.Compose(transform_list)

    # 递归查找所有图片文件
    def get_image_files(self, dir):
        image_files = []
        if dir is None: return []
        for root, _, files in os.walk(dir):
            for file in files:
                if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    image_files.append(os.path.join(root, file))
        print(f"一共递归检测到{len(image_files)}张图片...")
        return image_files[self.init_count_plus:]

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        try:
            image_path = self.image_files[idx]
            image = Image.open(image_path)
            image = self.transform(image)
        except (OSError, UnidentifiedImageError) as e:
            print(f"Warning: Could not load image {image_path} - {e}")
            image_path = self.image_files[0]
            image = Image.open(image_path)
            image = self.transform(image)
            return image, image_path  # 或者返回一个默认图像或替代数据

        return image, image_path