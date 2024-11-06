import torch
from model import MobileNetV1
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Load the pre-trained model
model = MobileNetV1(num_classes=10).to(device)
model.load_state_dict(torch.load('weight_binary_files/fixed32/model_fixed32_parameters.pth'))

# Filter and save weights only
weights = {k: v for k, v in model.state_dict().items() if 'weight' in k}
torch.save(weights, 'weight_binary_files/fixed32/model_fixed32_weights.pth')
print('Model weights saved.\n')

# Reload the saved weights for extraction
weight_pth = torch.load('weight_binary_files/fixed32/model_fixed32_weights.pth')

# fixed32 형식으로 변환하는 함수

# def float_to_fixed32(tensor, scale_factor):
#     return (tensor * scale_factor).round().to(torch.int32)  # round() 추가

# Fixed point 변환 함수
def float_to_fixed32(tensor, fraction_bits=16, total_bits=32):
    for value in tensor.flatten():
        scale = 2 ** fraction_bits
        max_val = (2 ** (total_bits - 1) - 1) / scale
        min_val = -(2 ** (total_bits - 1)) / scale
        fixed_value = np.round(value * scale)
        fixed_value = np.clip(fixed_value, min_val * scale, max_val * scale)

        return fixed_value / scale

def save_weights_as_bin(param, layer_name):
    """Converts weights to binary format (fixed32) and saves to a .bin file."""
    scale_factor = 2**16
    flattened_tensor = param.flatten()
    fixed32_value = float_to_fixed32(flattened_tensor, scale_factor).cpu().numpy()

    # Convert each int32 element to binary string representation
    flattened_numpy_bin = [
        f"{np.frombuffer(fixed32_value[i].tobytes(), dtype=np.uint32)[0]:032b}"
        for i in range(len(fixed32_value))
    ]

    # Save the binary values to a file
    with open(f'weight_binary_files/fixed32/{layer_name}_weight_bin.bin', 'w') as f:
        for b in flattened_numpy_bin:
            f.write(b + '\n')

# Iterate through model parameters
for name, param in model.named_parameters():
    if 'depthwise.weight' in name:
        layer_num = name.split('.')[0].replace('conv', '')  # Extract conv layer number correctly
        save_weights_as_bin(param.data, f'dwcv{layer_num}')  # Call function to save weights as binary

# 검증을 위해 float32 숫자를 직접 출력해보자
path = 'weight_binary_files/fixed32/model_fixed32_weights.pth'
checkpoint = torch.load(path)

# state_dict 추출
state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint

# depthwise 가중치 출력
for layer_name, weights in state_dict.items():
    if 'depthwise' in layer_name and 'weight' in layer_name:
        print(f"Layer: {layer_name} | Shape: {weights.shape}")
        print(weights/(2**16))  # depthwise 가중치 텐서를 출력
