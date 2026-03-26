import torch
import numpy as np
from models.dark_net import Darknet

def load_and_save_weights():
    cfg_path = '../datasets/darknet/yolo_v3.cfg'
    weights_path = '../datasets/darknet/yolo_v3.weights'
    save_path = '../datasets/darknet/yolo_v3.pth'

    model = Darknet(cfg_path, 224)
    
    # 解析二进制文件
    fp = open(weights_path, "rb")
    _ = np.fromfile(fp, dtype=np.int32, count=5) # 跳过文件头
    weights = np.fromfile(fp, dtype=np.float32)
    fp.close()

    ptr = 0
    for m in model.module_list:
        module_type = m[0].__class__.__name__
        if module_type == "Sequential": # 包含 Conv2d 的层
            for i in range(len(m)):
                if m[i].__class__.__name__ == "Conv2d":
                    conv = m[i]
                    if i + 1 < len(m) and m[i+1].__class__.__name__ == "BatchNorm2d":
                        bn = m[i+1]
                        # 加载 BN 参数
                        num_b = bn.bias.numel()
                        bn.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn.bias))
                        ptr += num_b
                        bn.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn.weight))
                        ptr += num_b
                        bn.running_mean.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn.running_mean))
                        ptr += num_b
                        bn.running_var.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn.running_var))
                        ptr += num_b
                    else:
                        # 加载 Conv 偏置
                        num_b = conv.bias.numel()
                        conv.bias.data.copy_(torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv.bias))
                        ptr += num_b
                    # 加载 Conv 权重
                    num_w = conv.weight.numel()
                    conv.weight.data.copy_(torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv.weight))
                    ptr += num_w

    # 包装成代码想要的 ['model'] 格式
    torch.save({'model': model.state_dict()}, save_path)
    print(f"成功转换并保存至: {save_path}")

if __name__ == "__main__":
    load_and_save_weights()