import os
import torch
import torch.nn as nn

# 定义一个函数来加载DINO模型
def load_dino_model(model_name="dinov2_vitg14_reg_1", 
                    checkpoint_path='checkpoints/dinov2_vitg14_reg4_pretrain.pth', 
                    root_dir="/data/disk16T/jewelrys/data/screenshot/screenshot_with_2view/"):
    
    device = torch.device("cuda")
    # 设置torch.hub的目录
    torch.hub.set_dir(root_dir)
    
    dino = torch.hub.load("/home/jingyu/DINO/dinov2-main", model_name, source='local')
    dino.load_state_dict(torch.load(os.path.join(root_dir, checkpoint_path)))

    dino.to(device)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        dino = nn.DataParallel(dino)

    return dino

if __name__ == "__main__":
    dino_model = load_dino_model()
    print("DINO model loaded successfully!")