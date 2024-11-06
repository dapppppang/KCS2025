import torch
from model import MobileNetV1
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the pre-trained model
model = MobileNetV1(num_classes=10).to(device)
model.load_state_dict(torch.load('weight_binary_files/fp32/model_fp32_parameters.pth'))

# Filter and save weights only
weights = {k: v for k, v in model.state_dict().items() if 'weight' in k}
torch.save(weights, 'weight_binary_files/fp32/model_fp32_weights.pth')
print('Model weights saved.\n')

# Reload the saved weights for extraction
weight_pth = torch.load('weight_binary_files/fp32/model_fp32_weights.pth')

def save_weights_as_bin(param, layer_name):
    """Converts weights to binary format (fp16) and saves to a .bin file."""
    flattened_tensor = param.flatten().cpu().numpy()
    float32_value = np.float32(flattened_tensor)

    flattened_numpy_bin = [
        f"{np.frombuffer(float32_value[i].tobytes(), dtype=np.uint32)[0]:032b}"
        for i in range(len(float32_value))
    ]

    with open(f'weight_binary_files/fp32/{layer_name}_weight_bin.bin', 'w') as f:
        for b in flattened_numpy_bin:
            f.write(b + '\n')

# Iterate through model parameters
for name, param in model.named_parameters():
    if 'depthwise.weight' in name:
        layer_num = name.split('.')[0].replace('conv', '')  # Extract conv layer number correctly
        save_weights_as_bin(param.data, f'dwcv{layer_num}')  # Call function to save weights as binary


# 검증을 위해 float32 숫자를 직접 출력해보자

# .pth 파일 경로
path = 'weight_binary_files/fp32/model_fp32_weights.pth'  # 실제 파일 경로로 변경하세요
checkpoint = torch.load(path)

# state_dict 추출
if "state_dict" in checkpoint:  # checkpoint가 'state_dict' 형태일 경우
    state_dict = checkpoint["state_dict"]
else:                           # checkpoint가 바로 state_dict인 경우
    state_dict = checkpoint

# depthwise 가중치 출력
for layer_name, weights in state_dict.items():
    if 'depthwise' in layer_name and 'weight' in layer_name:  # depthwise 레이어의 가중치만 필터링
        print(f"Layer: {layer_name} | Shape: {weights.shape}")
        #print(f'{weights.to(float32):.9f}')  # depthwise 가중치 텐서를 출력
        print(weights)  # depthwise 가중치 텐서를 출력
