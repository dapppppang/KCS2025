import torch
from model import MobileNetV1
from dataset import get_dataloaders
from train import train, test
from tqdm import tqdm
import torch.quantization

# def save_model(model, path1, path2, path3, path4):
#     # 모델의 state_dict 가져오기
#     state_dict = model.state_dict()
#
#     # 모든 파라미터를 bfloat16 형식으로 변환
# ##################
#     scale_factor = 2 ** 16  # 고정 소수점에서 소수 부분 비트를 16비트로 사용한다고 가정
#
#     # fixed32 형식으로 변환하는 함수
#     def float_to_fixed32(tensor, scale_factor):
#         return (tensor * scale_factor).to(torch.int32)
#
#     # state_dict의 모든 값들을 fixed32로 변환
#     fixed32_state_dict = {k: float_to_fixed32(v, scale_factor) for k, v in state_dict.items()}
#
#     # 모델 저장
#     torch.save(fixed32_state_dict, path4)
#     print(f"Model saved in fixed32 format at: {path4}")
# #######################
# #    fixed32_state_dict = {k: v.to(torch.fixed32) for k, v in state_dict.items()}
# #    torch.save(fixed32_state_dict, path4)
# #    print(f"Model saved in fixed32 format at: {path4}")
#
#     bf16_state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
#     torch.save(bf16_state_dict, path1)
#     print(f"Model saved in bfloat16 format at: {path1}")
#
#     fp16_state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
#     torch.save(fp16_state_dict, path2)
#     print(f"Model saved in fp16 format at: {path2}")
#
#     fp32_state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
#     torch.save(fp32_state_dict, path3)
#     print(f"Model saved in fp32 format at: {path3}")



if __name__ == '__main__':
    # 디바이스 설정 (GPU 사용 여부 확인)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 데이터 로더 준비
    trainloader, testloader = get_dataloaders(batch_size=128)

    # 모델 생성 및 이동
    model = MobileNetV1(num_classes=10).to(device)

    # 손실 함수 및 옵티마이저 설정
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # 학습 및 테스트 루프
    num_epochs = 10
    for epoch in range(1, num_epochs + 1):
        train(model, device, trainloader, optimizer, criterion, epoch)
        test(model, device, testloader)

    # # 학습이 완료된 후 모델 가중치 저장
    # # 나중에 모델을 재사용할 수 있도록 함.
    # save_model(model, "model_bf16_parameters.pth", "model_fp16_parameters.pth", "model_fp32_parameters.pth", "model_fixed32_parameters.pth")
