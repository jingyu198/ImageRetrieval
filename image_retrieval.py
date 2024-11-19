import os
from PIL import Image
import torch
import faiss
import gradio as gr
from tqdm import tqdm
from ImageData import ImageDataset
from load_model import load_dino_model

device = torch.device("cuda")
# load DINO model
root_dir = "/data/disk16T/jewelrys/data/screenshot/screenshot_with_2view/"
dino = load_dino_model(root_dir = root_dir)

show_demo = 'part'  # whole or part
features_dir = os.path.join(root_dir, 'outputs/features')
k = 50

all_image_paths = []
index = None

# Load all feature files and merge
def build_feature_space(features_dir):
    global all_image_paths, index
    first = True

    # Display the load progress using tqdm
    feature_files = sorted([f for f in os.listdir(features_dir) if f.endswith(".pt")])
    for feature_file in tqdm(feature_files[:], desc="loading feature files ..."):
        feature_data = torch.load(os.path.join(features_dir, feature_file))
        #print("已加载特征：", os.path.join(features_dir, feature_file))
        
        if first:
            d = feature_data["features"].shape[1]
            nlist = 500
            index = faiss.IndexFlatL2(d)
            index = faiss.IndexIVFFlat(index, d, nlist)
            first = False
            index.train(feature_data["features"].cpu().numpy())  # train only for the first time

        index.add(feature_data["features"].cpu().numpy())
        all_image_paths.extend(feature_data["paths"])

    if first:
        print("No feature detected")
    else:
        print("total number of dataset: ", index.ntotal)
        print("feature dimension: ", feature_data["features"].shape[1])

# Use FAISS for nearest neighbor search
def nearest_neighbor_search(query_features, k):
    global index
    index.nprobe = 1 # default 
    distances, indices = index.search(query_features.cpu().numpy(), k)
    return indices

# Extract the features of a single image
def extract_single_image_features(query_image_tensor):
    with torch.no_grad():
        query_image_tensor = query_image_tensor.to(device)
        features = dino(query_image_tensor).float()
    return features.to(device)


def convert_local_path2tos_path(local_png_path):
    dir_path = "/".join(local_png_path.split("/")[9:]).replace(".png", ".jcd")
    tos_jcd_path = f"tos://mm-jcds-data/jcds_models_all/{dir_path}"
    dirname, basename = os.path.split(tos_jcd_path)
    basename = basename[:-6] + ".jcd"
    new_path = os.path.join(dirname, basename)
    return new_path

def extract_tos_url(tos_url):
    if tos_url.startswith("tos://"):
        tos_url = tos_url[6:]
    
    index = tos_url.find("/")
    bucket_name = tos_url[:index]  
    file_key = tos_url[index+1:]   
    return f"http://172.31.0.3:9099/model/public/mm/api/internal/ve-cloud/tos/pre-signed-url?fileKey={file_key}&bucketName={bucket_name}"



# 返回最近邻图片的地址
# def return_nearest_neighbor_paths(query_image):
#     global all_image_paths

#     imagedata = ImageDataset(None, crop = False)
#     query_image_tensor = imagedata.transform(query_image).unsqueeze(0)
#     query_features = extract_single_image_features(query_image_tensor)
#     indices = nearest_neighbor_search(query_features, k)
#     nearest_neighbor_image_paths = [all_image_paths[i] for i in indices[0]]

#     return nearest_neighbor_image_paths


# Return the nearest neighbor picture
def return_nearest_neighbor_images(query_image):
    global all_image_paths
    
    # PIL image converted to pyTorch tensor
    if show_demo == 'part':
        query_image = query_image['composite'].convert("RGB")
    elif show_demo == 'whole':
        query_image = query_image.convert("RGB")

    imagedata = ImageDataset(None, crop = False)
    query_image_tensor = imagedata.transform(query_image).unsqueeze(0)
    query_features = extract_single_image_features(query_image_tensor)
    indices = nearest_neighbor_search(query_features, k)
    nearest_neighbor_image_paths = [all_image_paths[i] for i in indices[0]]

    # 对最近邻图像进行裁剪并返�?
    nearest_neighbor_images, download_links = [],[]
    for path in nearest_neighbor_image_paths:
        img = Image.open(path).crop((30, 80, 1800, 1020))
        nearest_neighbor_images.append(img)
        tos_jcd_path = convert_local_path2tos_path(path)
        download_link = extract_tos_url(tos_jcd_path)
        download_links.append(download_link)

    return [(image, f"No.{i+1}") for i, image in enumerate(nearest_neighbor_images)], download_links

print("Load the saved feature file...")
build_feature_space(features_dir)

if __name__ == "__main__":
    
    def update_buttons(links):
        test = 'http://172.31.0.3:9099/model/public/mm/api/internal/ve-cloud/tos/pre-signed-url?fileKey=jcds_models_all/gz_data/disk_0/艺美/宝石吊咀/15-水滴石天鹅吊�?jcd&bucketName=mm-jcds-data'
        print(links[:5])
        return [gr.update(link=links[i], visible=True) for i in range(50)]

    _HEADER_ = '''
    <h2><b>🤗jcd file retrieval system</b></h2>
    '''
    with gr.Blocks() as demo:
        gr.Markdown(_HEADER_)
        with gr.Row():
            with gr.Row():
                with gr.Column():
                    with gr.Row():
                        if show_demo == 'whole': 
                            input_img = gr.Image(type="pil", label="upload image")
                        elif show_demo == 'part': 
                            input_img = gr.ImageEditor(
                                label="upload image",
                                type="pil",
                                crop_size="1:1",
                                height= 500,
                                width= 800,)
                    with gr.Row():
                        submit = gr.Button("Search",variant="primary")
                    
                with gr.Column(variant="panel"):
                    results = gr.Gallery(label="Results")
        with gr.Row():
            gr.Markdown('''点击下方对应编号下载jcd文件''')
        with gr.Row("panel"):
            buttons = [gr.Button(value=f'{i+1}', size='sm') for i in range(50)]
        
        links = gr.State()

        submit.click(
            fn=return_nearest_neighbor_images,
            inputs=input_img,
            outputs=[results, links]
        ).success(
            fn=update_buttons,
            inputs=links,
            outputs=buttons,
        )


    demo.launch(server_name="221.194.175.112", server_port=23336)
