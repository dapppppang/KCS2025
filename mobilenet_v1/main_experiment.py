# 앞서 저장된 모델을 이용해 추론하기

import torch
from model import MobileNetV1
from dataset import get_dataloaders                                     # 데이터셋을 로드하고 훈련 및 테스트 데이터로 나누는 함수
from evaluate import evaluate_model
from sklearn.metrics import accuracy_score, classification_report       # 추론 성능 분석을 위한 정확도 및 분류 보고서를 제공하는 라이브러리
from tqdm import tqdm                                                   # 진행 상황을 시각적으로 표시하는 라이브러리


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    # 학습된 모델 불러오기
    model = MobileNetV1(num_classes=10).to(device)

    # 저장된 가중치 불러오기
    model.load_state_dict(torch.load('/content/model_fp32_parameters.pth'))

    # 딕셔너리 model.state_dict()의 weight 파라미터만 필터링 후 저장
    weights = {k: v for k, v in model.state_dict().items() if 'weight' in k}
    path = 'model_fp32_weights.pth'
    torch.save(weights, path)
    print(f'Model weights saved to {path}\n')

    # 모델을 평가 모드로 설정 (드롭아웃, 배치 정규화 비활성화)
    model.eval()


    # 데이터 로더 가져오기
    # train이나 test 할때와는 달리, FPGA에 보내기 위해 batch size를 1로 한다.
    trainloader, testloader = get_dataloaders(batch_size=1)

    # 테스트 데이터셋에 대한 예측 및 성능 평가
    all_labels, all_preds = evaluate_model(model, testloader, device)

    # 정확도 출력
    # accuracy_score : 전체 정확도 계산
    accuracy = accuracy_score(all_labels, all_preds)
    print(f"Accuracy: {accuracy:.4f}")

    # 분류 보고서 출력 (CIFAR-10의 클래스별 이름 설정)
    # classification_report : 각 클래스별 성능 지표 출력
    class_names = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
    report = classification_report(all_labels, all_preds, target_names=class_names)
    print(report)