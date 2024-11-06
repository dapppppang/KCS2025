import torch
from model import MobileNetV1
from dataset import get_dataloaders
from train import train, test
from infer import infer_and_evaluate
from tqdm import tqdm
import torch.quantization

def save_model(model, path1, path2, path3):
    # 모델의 state_dict 가져오기
    state_dict = model.state_dict()

    # 모든 파라미터를 bfloat16 형식으로 변환
    bf16_state_dict = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

    # 변환된 state_dict를 지정된 경로에 저장
    torch.save(bf16_state_dict, path1)
    print(f"Model saved in bfloat16 format at: {path1}")

    fp16_state_dict = {k: v.to(torch.float16) for k, v in state_dict.items()}
    torch.save(fp16_state_dict, path2)
    print(f"Model saved in fp16 format at: {path2}")

    fp32_state_dict = {k: v.to(torch.float32) for k, v in state_dict.items()}
    torch.save(fp32_state_dict, path3)
    print(f"Model saved in fp32 format at: {path3}")


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

    # 학습이 완료된 후 모델 가중치 저장
    # 나중에 모델을 재사용할 수 있도록 함.
    save_model(model, "model_bf16_parameters.pth", "model_fp16_parameters.pth", "model_fp32_parameters.pth")
