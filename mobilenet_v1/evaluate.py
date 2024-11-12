import torch
from torch.cuda.amp import autocast

# 모델 평가 함수 정의
def evaluate_model(model, dataloader, device):
    all_labels = []
    all_preds = []

    # 평가 모드로 설정
    model.eval()

    with torch.no_grad():
        for images, labels in dataloader:
            # device로 데이터 이동
            images = images.to(device)
            labels = labels.to(device)

            # # float32을 사용하여 추론 수행
            # with torch.cuda.amp.autocast(dtype=torch.float32):
            #     outputs = model(images)
            #     _, preds = torch.max(outputs, 1)


            # autocast 제거 (양자화된 모델에는 필요하지 않음)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return all_labels, all_preds
