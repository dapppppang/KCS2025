import torch
import numpy as np
from sklearn.metrics import classification_report

def infer_and_evaluate(model, device, test_loader):
    model.eval()  # 평가 모드로 설정
    all_labels = []
    all_predictions = []

    # 데이터 이터레이터 설정
    data_iter = iter(test_loader)

    # 데이터로더를 반복하여 배치 가져오기
    for images, labels in data_iter:
        images, labels = images.to(device), labels.to(device)

        # 예측 수행
        with torch.no_grad():  # 그래디언트 계산 비활성화
            outputs = model(images)

        _, predicted = torch.max(outputs, 1)  # 예측 클래스 결정

        # 모든 예측과 라벨을 리스트에 저장
        all_labels.extend(labels.cpu().numpy())  # CPU로 이동 후 NumPy 배열로 변환
        all_predictions.extend(predicted.cpu().numpy())

    # 정확도 출력
    accuracy = sum(np.array(all_labels) == np.array(all_predictions)) / len(all_labels)
    print(f'Accuracy: {accuracy * 100:.2f}%')

    # 분류 보고서 출력
    report = classification_report(all_labels, all_predictions, target_names=[str(i) for i in range(10)])
    print(report)

