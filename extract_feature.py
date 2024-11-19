import os
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from ImageData import ImageDataset
from load_model import load_dino_model


device = torch.device("cuda")

# 加载DINO模型
root_dir = "/data/disk16T/jewelrys/data/screenshot/screenshot_with_2view/"
dino = load_dino_model(root_dir = root_dir)

# 输出地址，数据地址
dataset_path = os.path.join(root_dir, 'unzip')
features_dir = os.path.join(root_dir, 'outputs_2/features')

batch_size = 128  
save_interval = 50000 # 每处理50000张图片保存一次feature
init_count_plus = 0 #从第(init_count_plus + 1)张图片开始提取特征


# 定义函数来提取特征并分批保存
def extract_features(model, dataloader):
    all_features = []
    all_image_paths = []
    model.eval()
    first = 1
    init_count = 1 + init_count_plus # 记录每轮起始数目
    save_count = 0 + init_count_plus # 记录已经处理的图片数量
    current_round = 0

    with torch.no_grad():
        # 使用 tqdm 添加进度条
        for images, image_paths in tqdm(dataloader, desc="Extracting features"):
            images = images.to(device)
            features = model(images)
            all_features.append(features.to(device))
            all_image_paths.extend(image_paths)
            save_count += len(image_paths)
            current_round += len(image_paths)

            if current_round >= save_interval:  # 每200000个数据保存一次
                start_idx = init_count
                end_idx = save_count
                feature_chunk_path = os.path.join(features_dir, f"feature_{start_idx}_{end_idx}.pt")

                # 保存特征和路径
                torch.save({"features": torch.cat(all_features, dim=0), "paths": all_image_paths}, feature_chunk_path)
                print(f"特征已保存到: {feature_chunk_path}")

                # 重置计数器和列表
                all_features = []
                all_image_paths = []
                current_round = 0
                init_count = end_idx + 1
        
        # 保存剩余的特征
        if len(all_image_paths) > 0:
            start_idx = init_count
            end_idx = save_count 
            feature_chunk_path = os.path.join(features_dir, f"feature_{start_idx}_{end_idx}.pt")
            torch.save({"features": torch.cat(all_features, dim=0), "paths": all_image_paths}, feature_chunk_path)
            print(f"特征已保存到: {feature_chunk_path}")

if __name__ == "__main__":
    # 创建保存特征的目录（如果不存在）
    if not os.path.exists(features_dir):
        os.makedirs(features_dir)
        print(f"创建特征保存目录: {features_dir}")
    print("提取新的特征...")

    # 创建DataLoader
    image_dataset = ImageDataset(dataset_path, init_count_plus = init_count_plus)
    image_dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=False, num_workers=8)
    extract_features(dino, image_dataloader)



