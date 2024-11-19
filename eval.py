import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from PIL import Image
from ImageData import ImageDataset
import random
from image_retrieval import return_nearest_neighbor_paths
import torch


def process_image(input_image_path):
    input_image = Image.open(input_image_path)
    nearest_neighbor_image_paths = return_nearest_neighbor_paths(input_image)
    
    correct_label = input_image_path  # 假设图像名称的前缀是类别标签
    topK_count = {k: 0 for k in Ks}

    for i, neighbor_path in enumerate(nearest_neighbor_image_paths):
        neighbor_label = neighbor_path
        if neighbor_label == correct_label:
            for k in Ks:
                if i < k:
                    topK_count[k] += 1
            break

    return topK_count

def calculate_topK_accuracy_test(input_images):
    overall_topK_accuracy = {k: 0 for k in Ks}

    # 使用 ThreadPoolExecutor 进行并行计算
    with ThreadPoolExecutor() as executor:
        future_to_image = {executor.submit(process_image, img): img for img in input_images}
        for future in tqdm(as_completed(future_to_image), total=len(future_to_image), desc="Calculating Top-K Accuracy"):
            result = future.result()
            for k in Ks:
                overall_topK_accuracy[k] += result[k]

    # 打印结果
    for k in Ks:
        print(f"Top-{k} Accuracy for synthesized data: {overall_topK_accuracy[k]}, {overall_topK_accuracy[k] / total_inputs}")

if __name__ == "__main__":
    # 加载DINO模型
    total_inputs = 5000
    root_dir = "/data/disk16T/jewelrys/data/screenshot/screenshot_with_2view/"
    dataset_path = os.path.join(root_dir, 'unzip')
    input_images = ImageDataset(dataset_path).image_files
    input_images = random.sample(input_images, total_inputs)

    # 记录准确率和平均精度
    Ks = [5, 10, 20, 50]
    calculate_topK_accuracy_test(input_images)